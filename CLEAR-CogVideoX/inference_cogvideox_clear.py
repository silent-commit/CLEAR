"""
CogVideoX-CLEAR Inference Script

Loads CogVideoX-2b base model with trained LoRA weights,
takes subtitle videos as input, outputs subtitle-free videos.

Architecture: Channel-concatenation V2V conditioning
  patch_embed.proj: Conv2d(16->1920) expanded to Conv2d(32->1920)
  First 16 channels: noisy clean latent (diffusion)
  Last 16 channels: subtitle condition latent (frozen)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import argparse
import time
import numpy as np
from peft import LoraConfig, set_peft_model_state_dict

from transformers import AutoTokenizer, T5EncoderModel
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)


def expand_transformer_input_channels(transformer):
    """Expand patch_embed.proj from Conv2d(16,1920) to Conv2d(32,1920)."""
    old_proj = transformer.patch_embed.proj
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
    print(f"Expanded patch_embed.proj: Conv2d({in_ch}->{out_ch}) -> Conv2d({in_ch*2}->{out_ch})")


def load_checkpoint(transformer, ckpt_path):
    """Load LoRA + condition_proj weights from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    proj_sd = ckpt['proj_state_dict']
    transformer.patch_embed.proj.weight.data.copy_(proj_sd['patch_embed.proj.weight'])
    if proj_sd.get('patch_embed.proj.bias') is not None:
        transformer.patch_embed.proj.bias.data.copy_(proj_sd['patch_embed.proj.bias'])
    print("Loaded condition_proj weights")

    lora_sd = ckpt['lora_state_dict']
    incompatible = set_peft_model_state_dict(transformer, lora_sd, adapter_name="default")
    if incompatible and getattr(incompatible, 'unexpected_keys', None):
        print(f"Warning: unexpected LoRA keys: {incompatible.unexpected_keys}")
    print(f"Loaded LoRA weights (step {ckpt.get('step', '?')})")


def encode_prompt(tokenizer, text_encoder, prompt, device, dtype, max_len=226):
    inputs = tokenizer(
        prompt, padding="max_length", max_length=max_len,
        truncation=True, add_special_tokens=True, return_tensors="pt",
    )
    with torch.no_grad():
        embeds = text_encoder(inputs.input_ids.to(device))[0]
    return embeds.to(dtype=dtype)


def probe_video_resolution(path):
    """Read the first frame to get original video dimensions."""
    import decord
    vr = decord.VideoReader(path)
    frame = vr[0]
    return frame.shape[0], frame.shape[1], len(vr), vr.get_avg_fps()


def auto_detect_processing_resolution(orig_h, orig_w, base_short=480, base_long=720):
    """Choose processing resolution based on aspect ratio.
    Portrait (H > W): height=base_long, width=base_short
    Landscape (H <= W): height=base_short, width=base_long
    """
    if orig_h > orig_w:
        return base_long, base_short
    else:
        return base_short, base_long


def load_video_frames(path, height=480, width=720):
    """Load all video frames and resize to target resolution.

    Returns:
        frames: [F, C, H, W] float tensor in [-1, 1]
        fps: original video fps
        original_count: original frame count
        orig_h, orig_w: original dimensions
    """
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(path)
    total = len(vr)
    fps = vr.get_avg_fps()

    frames = vr.get_batch(list(range(total)))
    frames = frames.float() / 255.0
    frames = frames.permute(0, 3, 1, 2)

    orig_h, orig_w = frames.shape[2], frames.shape[3]

    if frames.shape[2] != height or frames.shape[3] != width:
        frames = F.interpolate(frames, size=(height, width), mode='bilinear', align_corners=False)

    frames = frames * 2 - 1
    return frames, fps, total, orig_h, orig_w


def ensure_4kp1_frames(n):
    """Return largest valid frame count <= n satisfying CogVideoX VAE (4k+1)."""
    remainder = (3 + (n % 4)) % 4
    if remainder != 0 and n > remainder:
        return n - remainder
    return n


def make_chunks(total_frames, chunk_size=49, overlap=8):
    """Generate (start, end) indices for overlapping chunks."""
    if total_frames <= chunk_size:
        return [(0, total_frames)]

    chunks = []
    step = chunk_size - overlap
    start = 0

    while start + chunk_size <= total_frames:
        chunks.append((start, start + chunk_size))
        start += step

    if chunks[-1][1] < total_frames:
        last_start = total_frames - chunk_size
        if last_start != chunks[-1][0]:
            chunks.append((last_start, total_frames))

    return chunks


def blend_chunks(chunks_data, total_frames):
    """Blend overlapping chunks by weighted averaging."""
    if len(chunks_data) == 1:
        start, frames = chunks_data[0]
        n = min(len(frames), total_frames)
        if start == 0:
            return frames[:n]
        out = torch.zeros(total_frames, *frames.shape[1:])
        out[start:start + n] = frames[:n]
        return out

    chunks_data.sort(key=lambda x: x[0])
    C, H, W = chunks_data[0][1].shape[1:]
    accum = torch.zeros(total_frames, C, H, W)
    count = torch.zeros(total_frames)

    for start, frames in chunks_data:
        n = min(len(frames), total_frames - start)
        accum[start:start + n] += frames[:n]
        count[start:start + n] += 1

    count = count.clamp(min=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    return accum / count


def save_video(frames, output_path, fps=24.0):
    """Save tensor frames [F, C, H, W] in [-1,1] to mp4."""
    import cv2
    frames = ((frames + 1) / 2).clamp(0, 1) * 255
    frames = frames.byte().permute(0, 2, 3, 1).cpu().numpy()

    H, W = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved: {output_path} ({len(frames)} frames, {fps:.1f} fps)")


@torch.no_grad()
def denoise_chunk(transformer, scheduler, sub_latent, prompt_embeds,
                  num_steps, seed, device, dtype):
    """Run reverse diffusion for one latent chunk using DPM-Solver++ 2M."""
    scheduler_copy = CogVideoXDPMScheduler.from_config(scheduler.config)
    scheduler_copy.set_timesteps(num_steps, device=device)
    timesteps = scheduler_copy.timesteps

    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(
        sub_latent.shape, generator=generator, device=device, dtype=dtype,
    )

    B = latent.shape[0]
    old_pred_original_sample = None
    timestep_back = None

    for t in timesteps:
        latent_input = scheduler_copy.scale_model_input(latent, t)

        model_input = torch.cat([latent_input, sub_latent], dim=2)
        timestep_batch = t.unsqueeze(0).expand(B)

        model_output = transformer(
            hidden_states=model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep_batch,
            return_dict=False,
        )[0]

        latent, old_pred_original_sample = scheduler_copy.step(
            model_output, old_pred_original_sample, t, timestep_back, latent,
            return_dict=False,
        )
        timestep_back = t

    return latent


def parse_args():
    p = argparse.ArgumentParser(description='CogVideoX-CLEAR Inference')
    p.add_argument('--model_path', type=str, required=True,
                   help='Path to CogVideoX-2b base model')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to trained LoRA checkpoint (.pt)')
    p.add_argument('--input_video', type=str, required=True,
                   help='Path to input subtitle video')
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--video_name', type=str, default=None,
                   help='Output video name (without extension)')
    p.add_argument('--prompt', type=str,
                   default="Please remove the subtitle text from the video while preserving the original content.")
    p.add_argument('--num_steps', type=int, default=50,
                   help='Number of denoising steps')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lora_rank', type=int, default=64)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--width', type=int, default=720)
    p.add_argument('--auto_resolution', action='store_true',
                   help='Auto-detect resolution from input aspect ratio')
    p.add_argument('--match_input_size', action='store_true',
                   help='Resize output to match original input dimensions')
    p.add_argument('--chunk_size', type=int, default=49,
                   help='Frames per chunk (must be 4k+1)')
    p.add_argument('--chunk_overlap', type=int, default=9,
                   help='Overlapping frames between chunks')
    p.add_argument('--enable_slicing', action='store_true', default=True)
    p.add_argument('--enable_tiling', action='store_true', default=True)
    p.add_argument('--copy_source', action='store_true',
                   help='Copy source video to output dir')
    return p.parse_args()


def main():
    args = parse_args()
    video_name = args.video_name or os.path.splitext(os.path.basename(args.input_video))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    orig_h, orig_w, orig_nframes, orig_fps = probe_video_resolution(args.input_video)

    if args.auto_resolution:
        args.height, args.width = auto_detect_processing_resolution(orig_h, orig_w)

    print("=" * 70)
    print(f"CogVideoX-CLEAR Inference: {video_name}")
    print(f"  Model:      {args.model_path}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Input:      {args.input_video}")
    print(f"  Input res:  {orig_h}x{orig_w} ({'portrait' if orig_h > orig_w else 'landscape'})")
    print(f"  Process at: {args.height}x{args.width}")
    if args.match_input_size:
        print(f"  Output res: {orig_h}x{orig_w} (match input)")
    print(f"  Steps:      {args.num_steps}")
    print(f"  Chunk:      {args.chunk_size} frames, overlap={args.chunk_overlap}")
    print("=" * 70)

    # ========== Load Models ==========
    print("\n[1/5] Loading models...")
    t_load_start = time.time()

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype,
    ).to(device)

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype,
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype,
    ).to(device)
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        model_path, subfolder="scheduler",
    )

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # ========== Modify Architecture + Load LoRA ==========
    print("\n[2/5] Setting up LoRA and loading checkpoint...")
    expand_transformer_input_channels(transformer)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(lora_config)
    load_checkpoint(transformer, args.checkpoint)

    transformer.eval().to(device, dtype=dtype)

    # ========== Encode Prompt ==========
    prompt_embeds = encode_prompt(tokenizer, text_encoder, args.prompt, device, dtype)
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    t_load_end = time.time()
    print(f"Models loaded in {t_load_end - t_load_start:.1f}s")

    # ========== Load Video ==========
    print(f"\n[3/5] Loading video: {args.input_video}")
    all_frames, fps, original_total, vid_orig_h, vid_orig_w = load_video_frames(
        args.input_video, height=args.height, width=args.width,
    )
    total_frames = len(all_frames)
    print(f"  {original_total} frames ({vid_orig_h}x{vid_orig_w}) -> process at {args.height}x{args.width}, fps={fps:.1f}")

    # ========== Process in Chunks ==========
    chunks = make_chunks(total_frames, args.chunk_size, args.chunk_overlap)
    print(f"\n[4/5] Processing {len(chunks)} chunks...")

    t_infer_start = time.time()
    chunks_output = []

    for ci, (c_start, c_end) in enumerate(chunks):
        chunk_frames = all_frames[c_start:c_end]
        n_valid = ensure_4kp1_frames(len(chunk_frames))
        chunk_frames = chunk_frames[:n_valid]
        n_chunk = len(chunk_frames)

        chunk_t_start = time.time()
        print(f"  Chunk {ci+1}/{len(chunks)}: frames[{c_start}:{c_start + n_chunk}] ({n_chunk} frames)")

        chunk_input = chunk_frames.unsqueeze(0).to(device=device, dtype=dtype)
        chunk_vae_input = chunk_input.permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            sub_latent = vae.encode(chunk_vae_input).latent_dist.sample()
            sub_latent = sub_latent * vae.config.scaling_factor
            sub_latent = sub_latent.permute(0, 2, 1, 3, 4)

        del chunk_vae_input
        torch.cuda.empty_cache()

        clean_latent = denoise_chunk(
            transformer, scheduler, sub_latent, prompt_embeds,
            args.num_steps, args.seed + ci, device, dtype,
        )

        del sub_latent
        torch.cuda.empty_cache()

        with torch.no_grad():
            decode_input = clean_latent.permute(0, 2, 1, 3, 4) / vae.config.scaling_factor
            decoded = vae.decode(decode_input).sample
            decoded = decoded.permute(0, 2, 1, 3, 4)

        chunk_output = decoded.squeeze(0).cpu().float()

        if chunk_output.shape[0] > n_chunk:
            chunk_output = chunk_output[:n_chunk]

        chunks_output.append((c_start, chunk_output))

        del clean_latent, decode_input, decoded
        torch.cuda.empty_cache()

        chunk_t_end = time.time()
        print(f"    Done in {chunk_t_end - chunk_t_start:.1f}s")

    t_infer_end = time.time()
    inference_time = t_infer_end - t_infer_start

    # ========== Blend & Save ==========
    print(f"\n[5/5] Blending and saving...")
    output_frames = blend_chunks(chunks_output, total_frames)

    if args.match_input_size and (vid_orig_h != args.height or vid_orig_w != args.width):
        print(f"  Resizing output: {args.height}x{args.width} -> {vid_orig_h}x{vid_orig_w}")
        output_frames = F.interpolate(
            output_frames, size=(vid_orig_h, vid_orig_w),
            mode='bilinear', align_corners=False,
        )

    output_video_path = os.path.join(args.output_dir, f"{video_name}_result.mp4")
    save_video(output_frames, output_video_path, fps=fps)

    if args.copy_source:
        import shutil
        src_copy = os.path.join(args.output_dir, f"{video_name}_source.mp4")
        shutil.copy2(args.input_video, src_copy)

    time_per_frame = inference_time / total_frames
    out_h, out_w = output_frames.shape[2], output_frames.shape[3]
    timing_info = {
        'video_name': video_name,
        'total_time_seconds': round(inference_time, 2),
        'time_per_frame_seconds': round(time_per_frame, 4),
        'num_frames': total_frames,
        'num_chunks': len(chunks),
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'num_steps': args.num_steps,
        'seed': args.seed,
        'model_load_time_seconds': round(t_load_end - t_load_start, 2),
        'input_resolution': f'{vid_orig_h}x{vid_orig_w}',
        'processing_resolution': f'{args.height}x{args.width}',
        'output_resolution': f'{out_h}x{out_w}',
        'auto_resolution': args.auto_resolution,
        'match_input_size': args.match_input_size,
    }
    timing_path = os.path.join(args.output_dir, 'timing_info.json')
    with open(timing_path, 'w') as f:
        json.dump(timing_info, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Inference Complete: {video_name}")
    print(f"  Output:       {output_video_path}")
    print(f"  Frames:       {total_frames}")
    print(f"  Total time:   {inference_time:.1f}s")
    print(f"  Per frame:    {time_per_frame:.3f}s")
    print(f"  Timing info:  {timing_path}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
