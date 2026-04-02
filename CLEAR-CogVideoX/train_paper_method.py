"""
CogVideoX-2b CLEAR Paper Method Training (Stage 1)

Video-to-video subtitle removal with:
- Channel-concatenation conditioning (subtitle latent as condition)
- Focal loss with mask guidance from Stage 1 adapter
- Multi-scale temporal consistency loss
- LoRA fine-tuning on CogVideoXTransformer3DModel

Architecture modification:
  patch_embed.proj: Conv2d(16→1920) expanded to Conv2d(32→1920)
  First 16 channels: noisy clean latent (pretrained weights preserved)
  Last 16 channels: subtitle condition latent (zero-initialized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
import argparse
import numpy as np
import time
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model_state_dict

from transformers import AutoTokenizer, T5EncoderModel
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)
from diffusers.training_utils import cast_training_params


# ====================== Model Modification ======================

def expand_transformer_input_channels(transformer):
    """
    Expand patch_embed.proj from Conv2d(16, 1920) to Conv2d(32, 1920).
    First 16 channels keep pretrained weights (noisy latent).
    Last 16 channels zero-initialized (subtitle condition latent).
    """
    old_proj = transformer.patch_embed.proj
    assert isinstance(old_proj, nn.Conv2d), f"Expected Conv2d, got {type(old_proj)}"

    in_ch = old_proj.in_channels
    out_ch = old_proj.out_channels
    ks = old_proj.kernel_size
    stride = old_proj.stride
    has_bias = old_proj.bias is not None

    new_proj = nn.Conv2d(
        in_ch * 2, out_ch,
        kernel_size=ks, stride=stride, bias=has_bias,
    )

    with torch.no_grad():
        new_proj.weight[:, :in_ch] = old_proj.weight
        new_proj.weight[:, in_ch:] = 0.0
        if has_bias:
            new_proj.bias.copy_(old_proj.bias)

    transformer.patch_embed.proj = new_proj
    print(f"Expanded patch_embed.proj: Conv2d({in_ch}→{out_ch}) → Conv2d({in_ch*2}→{out_ch})")
    return transformer


# ====================== Mask Predictor ======================

def create_mask_predictor(checkpoint_path, device, adapter_code_path=None):
    """Load Stage 1 adapter for mask prediction (frozen, float32).

    Args:
        checkpoint_path: path to adapter checkpoint (.pt)
        device: torch device
        adapter_code_path: path to the directory containing models/dual_encoder.py
                           (the MultiscaleDisentangledAdapter implementation)
    """
    if adapter_code_path:
        sys.path.insert(0, adapter_code_path)

    from models.dual_encoder import MultiscaleDisentangledAdapter

    adapter = MultiscaleDisentangledAdapter(
        backbone='resnet50',
        encoder_output_dim=512,
        content_dim=256,
        subtitle_dim=256,
        use_reconstruction=True,
        use_multiscale=True,
        fusion_layer='layer2',
        pretrained=False,
        backbone_weight_path=None,
        use_adaptive_loss_weights=False,
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        sd = ckpt['model']
    elif 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt

    if any(k.startswith('module.') for k in sd.keys()):
        sd = {k.replace('module.', ''): v for k, v in sd.items()}

    adapter.load_state_dict(sd, strict=False)
    adapter.eval().float().to(device)
    for p in adapter.parameters():
        p.requires_grad = False

    print(f"Mask predictor loaded from {checkpoint_path}")
    return adapter


def predict_masks_batch(mask_predictor, subtitle_frames_01, dilation_kernel=3):
    """
    Predict subtitle masks from raw frames.

    Args:
        mask_predictor: Stage 1 adapter
        subtitle_frames_01: [B, F, 3, H, W] in [0, 1]
        dilation_kernel: dilation size

    Returns:
        masks: [B, F, H, W] binary-ish masks in [0, 1]
    """
    B, T, C, H, W = subtitle_frames_01.shape
    device = subtitle_frames_01.device

    with torch.no_grad():
        frames_norm = subtitle_frames_01 * 2 - 1
        flat = frames_norm.reshape(B * T, C, H, W).float()
        _, _, mask_logits, _ = mask_predictor(flat, return_all=True)
        masks = torch.sigmoid(mask_logits)

        if dilation_kernel > 0:
            pad = dilation_kernel // 2
            masks = F.max_pool2d(masks, kernel_size=dilation_kernel, stride=1, padding=pad)

        masks = masks.reshape(B, T, H, W)
    return masks


def align_mask_to_latent(mask, target_frames, target_h, target_w):
    """
    Align mask from pixel space to latent space dimensions.

    Args:
        mask: [B, F_pixel, H_pixel, W_pixel]
        target_frames: latent temporal dim
        target_h, target_w: latent spatial dims
    """
    B, F_pix, H, W = mask.shape

    if F_pix != target_frames:
        chunk_size = F_pix / target_frames
        aligned = []
        for i in range(target_frames):
            s = int(i * chunk_size)
            e = min(int((i + 1) * chunk_size), F_pix)
            e = max(e, s + 1)
            chunk = mask[:, s:e]
            aligned.append(chunk.max(dim=1)[0])
        mask = torch.stack(aligned, dim=1)

    F_cur = mask.shape[1]
    if mask.shape[2] != target_h or mask.shape[3] != target_w:
        kh = mask.shape[2] // target_h
        kw = mask.shape[3] // target_w
        frames = []
        for t in range(F_cur):
            fr = mask[:, t:t+1]
            fr = F.max_pool2d(fr, kernel_size=(kh, kw), stride=(kh, kw))
            frames.append(fr)
        mask = torch.cat(frames, dim=1)

    return mask


# ====================== Loss Functions ======================

def compute_temporal_loss(pred, mask, scales=[1, 2, 4], lambda_t=0.1):
    """
    Multi-scale temporal consistency loss on predicted velocity.
    Subtitle regions are down-weighted to 0.15 to avoid penalizing
    necessary temporal changes in inpainted areas.
    """
    B, T, C, H, W = pred.shape
    if T < 2:
        return torch.tensor(0.0, device=pred.device)

    total = 0.0
    count = 0
    for s in scales:
        if T <= s:
            continue
        diff = (pred[:, :-s] - pred[:, s:]) ** 2
        diff = diff.mean(dim=2)

        if mask is not None:
            sub = (mask[:, :-s] + mask[:, s:]).clamp(0, 1)
            weight = 1.0 - sub * 0.85
            diff = diff * weight

        total += diff.mean() / s
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=pred.device)
    return lambda_t * total / count


# ====================== Dataset ======================

class SubtitleRemovalDataset(Dataset):
    """
    Paired video dataset for subtitle removal training.

    Directory structure:
        clean_dir/
            video_name.mp4          (clean, no subtitles)
        subtitle_dir/
            video_name_with_subtitle.mp4  (same video with subtitles)

    Pairs are matched by filename: remove '_with_subtitle' suffix to find clean counterpart.

    Optional filtering via non_text.json (whitelist of allowed video IDs).
    """

    def __init__(
        self,
        clean_dirs,
        subtitle_dirs,
        non_text_json=None,
        num_samples_per_dir=50,
        num_frames=49,
        height=480,
        width=720,
        random_seed=2026,
    ):
        self.num_frames = num_frames
        self.height = height
        self.width = width

        allowed = self._load_filter(non_text_json)
        self.pairs = self._collect_pairs(clean_dirs, subtitle_dirs, allowed)

        self.pairs = self._sample_per_dir(
            clean_dirs, subtitle_dirs, allowed,
            num_samples_per_dir, random_seed,
        )
        print(f"Dataset: {len(self.pairs)} video pairs")

    def _load_filter(self, json_path):
        if not json_path or not os.path.exists(json_path):
            return None
        with open(json_path, 'r') as f:
            items = json.load(f)
        return set(s.rsplit('.', 1)[0] for s in items)

    def _collect_pairs(self, clean_dirs, subtitle_dirs, allowed):
        pairs = []
        for c_dir, s_dir in zip(clean_dirs, subtitle_dirs):
            if not os.path.exists(c_dir) or not os.path.exists(s_dir):
                continue
            clean_map = {}
            for fn in os.listdir(c_dir):
                if fn.endswith('.mp4'):
                    name = fn[:-4]
                    clean_map[name] = os.path.join(c_dir, fn)

            dir_has_filter_coverage = False
            if allowed is not None:
                dir_has_filter_coverage = any(name in allowed for name in clean_map)
            use_filter = allowed is not None and dir_has_filter_coverage

            dir_count = 0
            for fn in os.listdir(s_dir):
                if fn.endswith('.mp4'):
                    base = fn[:-4].replace('_with_subtitle', '')
                    if base in clean_map:
                        if not use_filter or base in allowed:
                            pairs.append((clean_map[base], os.path.join(s_dir, fn), c_dir))
                            dir_count += 1

            filter_status = "filtered by non_text.json" if use_filter else "no filter"
            print(f"  {c_dir}: {dir_count} pairs found ({filter_status})")
        return pairs

    def _sample_per_dir(self, clean_dirs, subtitle_dirs, allowed, n_per_dir, seed):
        rng = np.random.RandomState(seed)
        all_sampled = []
        for c_dir, s_dir in zip(clean_dirs, subtitle_dirs):
            dir_pairs = [p for p in self.pairs if p[2] == c_dir]
            if len(dir_pairs) > n_per_dir:
                idxs = rng.choice(len(dir_pairs), size=n_per_dir, replace=False)
                dir_pairs = [dir_pairs[i] for i in idxs]
            all_sampled.extend(dir_pairs)
            print(f"  {c_dir}: {len(dir_pairs)} pairs sampled")
        return [(p[0], p[1]) for p in all_sampled]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_path, subtitle_path = self.pairs[idx]

        clean_frames = self._load_video(clean_path)
        sub_frames = self._load_video(subtitle_path)

        total = min(len(clean_frames), len(sub_frames))
        if total >= self.num_frames:
            start = np.random.randint(0, total - self.num_frames + 1)
            clean_frames = clean_frames[start:start + self.num_frames]
            sub_frames = sub_frames[start:start + self.num_frames]
        else:
            clean_frames = clean_frames[:total]
            sub_frames = sub_frames[:total]

        nf = len(clean_frames)
        remainder = (3 + (nf % 4)) % 4
        if remainder != 0 and nf > remainder:
            clean_frames = clean_frames[:nf - remainder]
            sub_frames = sub_frames[:nf - remainder]

        clean_t = self._frames_to_tensor(clean_frames)
        sub_t = self._frames_to_tensor(sub_frames)

        return {
            'clean_video': clean_t,
            'subtitle_video': sub_t,
            'subtitle_raw': (sub_t + 1) / 2,
        }

    def _load_video(self, path):
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(path)
        total = len(vr)
        return vr.get_batch(list(range(total)))

    def _frames_to_tensor(self, frames):
        """Convert decord frames [F, H, W, C] uint8 to [F, C, H, W] float [-1, 1]."""
        frames = frames.float() / 255.0
        frames = frames.permute(0, 3, 1, 2)

        if frames.shape[2] != self.height or frames.shape[3] != self.width:
            frames = F.interpolate(frames, size=(self.height, self.width), mode='bilinear', align_corners=False)

        frames = frames * 2 - 1
        return frames


# ====================== Prompt Encoding ======================

def encode_prompt(tokenizer, text_encoder, prompt, device, dtype, max_len=226):
    inputs = tokenizer(
        prompt, padding="max_length", max_length=max_len,
        truncation=True, add_special_tokens=True, return_tensors="pt",
    )
    with torch.no_grad():
        embeds = text_encoder(inputs.input_ids.to(device))[0]
    return embeds.to(dtype=dtype)


# ====================== Main ======================

def parse_args():
    p = argparse.ArgumentParser(description='CogVideoX-CLEAR Stage 1 Training (Paper Method)')
    p.add_argument('--model_path', type=str, required=True,
                   help='Path to CogVideoX-2b pretrained model')
    p.add_argument('--adapter_checkpoint', type=str, required=True,
                   help='Path to Stage 1 mask predictor checkpoint (.pt)')
    p.add_argument('--adapter_code_path', type=str, default=None,
                   help='Path to directory containing models/dual_encoder.py for mask predictor')
    p.add_argument('--non_text_json', type=str, default=None,
                   help='Path to non_text.json filter list (optional)')
    p.add_argument('--clean_dirs', type=str, nargs='+', required=True,
                   help='Directories containing clean (subtitle-free) videos')
    p.add_argument('--subtitle_dirs', type=str, nargs='+', required=True,
                   help='Directories containing subtitle videos (must match clean_dirs order)')
    p.add_argument('--output_dir', type=str, required=True,
                   help='Directory to save checkpoints')
    p.add_argument('--num_samples_per_dir', type=int, default=50)
    p.add_argument('--num_frames', type=int, default=49,
                   help='Frames per video clip (must satisfy 4k+1 for CogVideoX VAE)')
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--width', type=int, default=720)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--lora_rank', type=int, default=64)
    p.add_argument('--focal_alpha', type=float, default=5.0,
                   help='Focal loss weight for subtitle regions')
    p.add_argument('--temporal_weight', type=float, default=0.1,
                   help='Temporal consistency loss weight')
    p.add_argument('--num_epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--save_interval', type=int, default=25)
    p.add_argument('--log_interval', type=int, default=1)
    p.add_argument('--gradient_checkpointing', action='store_true', default=True)
    p.add_argument('--enable_slicing', action='store_true', default=True)
    p.add_argument('--enable_tiling', action='store_true', default=True)
    p.add_argument('--mask_dilation_kernel', type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='fp16',
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print("=" * 70)
        print("CogVideoX-2b CLEAR Paper Method Training (Stage 1)")
        print("=" * 70)

    # ========== Load Models ==========
    model_path = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.float16,
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # ========== Expand Transformer for V2V ==========
    expand_transformer_input_channels(transformer)

    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(lora_config)
    transformer.patch_embed.proj.requires_grad_(True)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    weight_dtype = torch.float16
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if accelerator.mixed_precision == 'fp16':
        cast_training_params([transformer], dtype=torch.float32)

    # ========== Load Mask Predictor ==========
    mask_predictor = create_mask_predictor(
        args.adapter_checkpoint, accelerator.device,
        adapter_code_path=args.adapter_code_path,
    )

    # ========== Trainable Params ==========
    lora_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    proj_params = list(transformer.patch_embed.proj.parameters())
    all_trainable = []
    seen_ids = set()
    for p in lora_params + proj_params:
        if id(p) not in seen_ids:
            all_trainable.append(p)
            seen_ids.add(id(p))

    n_trainable = sum(p.numel() for p in all_trainable)
    n_total = sum(p.numel() for p in transformer.parameters())
    if accelerator.is_main_process:
        print(f"Trainable: {n_trainable:,} / {n_total:,} ({n_trainable/n_total*100:.2f}%)")

    # ========== Dataset ==========
    dataset = SubtitleRemovalDataset(
        clean_dirs=args.clean_dirs,
        subtitle_dirs=args.subtitle_dirs,
        non_text_json=args.non_text_json,
        num_samples_per_dir=args.num_samples_per_dir,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        random_seed=args.seed,
    )

    def collate_fn(batch):
        return {
            k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()
        }

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, collate_fn=collate_fn,
    )

    # ========== Pre-encode prompt ==========
    prompt = "Please remove the subtitle text from the video while preserving the original content."
    prompt_embeds = encode_prompt(
        tokenizer, text_encoder, prompt, accelerator.device, weight_dtype,
    )

    del text_encoder
    torch.cuda.empty_cache()

    # ========== Optimizer & Scheduler ==========
    optimizer = torch.optim.AdamW(
        all_trainable, lr=args.learning_rate,
        betas=(0.9, 0.95), weight_decay=1e-4,
    )

    from torch.optim.lr_scheduler import CosineAnnealingLR
    total_steps = args.num_epochs * len(dataloader)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.learning_rate * 0.1)

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler,
    )

    # ========== Training Loop ==========
    if accelerator.is_main_process:
        print(f"\nStarting training: {len(dataset)} samples, {args.num_epochs} epochs")
        print(f"Steps per epoch: {len(dataloader)}")

    global_step = 0
    for epoch in range(args.num_epochs):
        transformer.train()
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)

        for batch in progress:
            with accelerator.accumulate(transformer):
                clean_video = batch['clean_video'].to(dtype=weight_dtype)
                sub_video = batch['subtitle_video'].to(dtype=weight_dtype)
                sub_raw = batch['subtitle_raw']

                B = clean_video.shape[0]
                device = clean_video.device

                with torch.no_grad():
                    clean_vae = clean_video.permute(0, 2, 1, 3, 4)
                    sub_vae = sub_video.permute(0, 2, 1, 3, 4)

                    clean_latent_dist = vae.encode(clean_vae).latent_dist
                    sub_latent_dist = vae.encode(sub_vae).latent_dist

                    clean_latent = clean_latent_dist.sample() * vae.config.scaling_factor
                    sub_latent = sub_latent_dist.sample() * vae.config.scaling_factor

                    clean_latent = clean_latent.permute(0, 2, 1, 3, 4)
                    sub_latent = sub_latent.permute(0, 2, 1, 3, 4)

                _, F_lat, C_lat, H_lat, W_lat = clean_latent.shape

                with torch.no_grad():
                    masks_pixel = predict_masks_batch(
                        mask_predictor, sub_raw, dilation_kernel=args.mask_dilation_kernel,
                    )
                    masks_latent = align_mask_to_latent(
                        masks_pixel, F_lat, H_lat, W_lat,
                    ).to(device=device, dtype=weight_dtype)

                noise = torch.randn_like(clean_latent)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (B,), device=device,
                ).long()

                noisy_latent = scheduler.add_noise(
                    clean_latent.to(weight_dtype), noise.to(weight_dtype), timesteps,
                )

                model_input = torch.cat([noisy_latent, sub_latent], dim=2)

                batch_prompt = prompt_embeds.expand(B, -1, -1)

                model_output = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=batch_prompt,
                    timestep=timesteps,
                    return_dict=False,
                )[0]

                model_pred = scheduler.get_velocity(model_output, noisy_latent, timesteps)
                target = clean_latent.to(weight_dtype)

                alphas_cumprod = scheduler.alphas_cumprod[timesteps.cpu()].to(device=device)
                ts_weights = 1.0 / (1.0 - alphas_cumprod)
                while len(ts_weights.shape) < len(model_pred.shape):
                    ts_weights = ts_weights.unsqueeze(-1)

                base_err = (model_pred - target) ** 2 * ts_weights

                err_spatial = base_err.mean(dim=2)
                subtitle_region = (masks_latent > 0.5).float()
                bg_region = 1.0 - subtitle_region

                spatial_weight = 1.0 + args.focal_alpha * subtitle_region
                focal_term = (err_spatial.detach() + 1e-6).pow(1.5)
                weight_map = torch.clamp(spatial_weight * focal_term, 0.1, 20.0)

                focal_loss = (err_spatial * weight_map).mean()

                t_loss = compute_temporal_loss(
                    model_pred, masks_latent,
                    scales=[1, 2, 4], lambda_t=args.temporal_weight,
                )

                loss = focal_loss + t_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(all_trainable, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % args.log_interval == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        sub_l = (err_spatial * subtitle_region).sum() / (subtitle_region.sum() + 1e-6)
                        bg_l = (err_spatial * bg_region).sum() / (bg_region.sum() + 1e-6)
                    print(
                        f"[Step {global_step}] loss={loss.item():.6f} "
                        f"focal={focal_loss.item():.6f} temporal={t_loss.item():.6f} "
                        f"sub={sub_l.item():.6f} bg={bg_l.item():.6f} "
                        f"mask={subtitle_region.mean().item()*100:.1f}%"
                    )

                if global_step % args.save_interval == 0 and accelerator.is_main_process:
                    save_checkpoint(accelerator, transformer, args.output_dir, global_step)

    if accelerator.is_main_process:
        save_checkpoint(accelerator, transformer, args.output_dir, global_step, final=True)
        print(f"\nTraining complete. Output: {args.output_dir}")


def save_checkpoint(accelerator, transformer, output_dir, step, final=False):
    """Save LoRA weights and expanded patch_embed.proj weights."""
    unwrapped = accelerator.unwrap_model(transformer)

    lora_sd = get_peft_model_state_dict(unwrapped)

    proj_sd = {
        'patch_embed.proj.weight': unwrapped.patch_embed.proj.weight.cpu(),
        'patch_embed.proj.bias': unwrapped.patch_embed.proj.bias.cpu()
            if unwrapped.patch_embed.proj.bias is not None else None,
    }

    tag = "final" if final else f"step_{step}"
    ckpt_path = os.path.join(output_dir, f"checkpoint_{tag}.pt")
    torch.save({
        'lora_state_dict': lora_sd,
        'proj_state_dict': proj_sd,
        'step': step,
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == '__main__':
    main()
