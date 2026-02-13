"""
CLEAR Stage II: Adaptive Weighting Learning

Trains LoRA-adapted diffusion model with a lightweight occlusion head that
learns context-dependent weighting strategies. The occlusion head predicts
adaptive weights from DiT encoder features, modulating generation through
focal weighting with gradient backflow.

Key components:
1. LoRA adaptation on frozen pre-trained diffusion model (0.77% parameters)
2. Context-dependent occlusion head for adaptive weight prediction
3. Joint optimization: L_distill + L_gen + 0.1 * L_sparse
4. Dynamic alpha scheduling for exploration

Reference: Section 3.3-3.4 in the CLEAR paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys, json, argparse
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import time
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import random
import torchvision.transforms.functional as TF

# DiffSynth-Studio imports (requires DiffSynth-Studio to be installed or in PYTHONPATH)
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import ModelLogger, DiffusionTrainingModule
from diffsynth.schedulers.flow_match import FlowMatchScheduler

# CLEAR model imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.dual_encoder import MultiscaleDisentangledAdapter
from models.occlusion_head import (
    OcclusionHead,
    compute_adaptive_weights,
    dynamic_alpha_schedule,
    compute_context_distillation_loss,
    compute_context_consistency_loss,
)

# Alignment utilities
try:
    from utils.vae_temporal_alignment import align_mask_to_vae_temporal
except ImportError:
    print("Warning: vae_temporal_alignment not found, mask alignment may not work")


# ========== Focal Loss & Temporal Loss（从旧脚本复制）==========

def get_dynamic_alpha(global_step, initial_alpha=5.0, max_alpha=15.0, cycle_length=20):
    """
    动态Alpha调度器 - 小三角波（每20步）
    
    🎯 策略：每20步做一次 5→15→5 的三角波
    - Step 0-10:  alpha从5线性增加到15
    - Step 10-20: alpha从15线性下降到5
    - Step 20-40: 重复...
    
    Args:
        global_step: 当前训练步数
        initial_alpha: 起始alpha值（默认5.0）
        max_alpha: 峰值alpha值（默认15.0）
        cycle_length: 周期长度（默认20步）
    
    Returns:
        当前step对应的alpha值
    """
    step_in_cycle = global_step % cycle_length
    half_cycle = cycle_length / 2.0
    
    if step_in_cycle <= half_cycle:
        # 前半周期：5 → 15
        alpha = initial_alpha + (max_alpha - initial_alpha) * (step_in_cycle / half_cycle)
    else:
        # 后半周期：15 → 5
        alpha = max_alpha - (max_alpha - initial_alpha) * ((step_in_cycle - half_cycle) / half_cycle)
    
    return alpha


def compute_timestep_weight(timestep):
    """
    计算timestep的训练权重（基于FlowMatchScheduler的实现）
    
    🎯 核心思想：中间timestep（困难）权重高，两端（简单）权重低
    使用高斯权重: exp(-2 * ((t - 0.5) / 1.0)^2)
    
    Args:
        timestep: [B] 连续值，范围[0, 1]
    
    Returns:
        weight: [B] 每个timestep的权重
    """
    # 高斯权重：中心0.5权重最高，两端权重低
    # 公式：exp(-2 * ((t - 0.5) / 1.0)^2)
    weight = torch.exp(-2.0 * ((timestep - 0.5) ** 2))
    
    # 归一化（可选，保持loss尺度一致）
    # weight = weight / weight.mean()
    
    return weight


def compute_multi_scale_temporal_loss(
    pred_frames,
    mask_frames,
    scales=[1, 4, 8, 16],  # 🔥修改：适合81帧长视频，原[1,2,4]
    lambda_temporal=0.1,
    use_full_frame=True,  # 🔥新增：全帧计算（推荐）vs只背景
    subtitle_weight=0.15,  # 🔥改进：从0.5降到0.15，减少字幕区域时序约束
    use_charbonnier=True,  # 🔥新增：使用Charbonnier替代L2
    charbonnier_eps=1e-3,  # 🔥Charbonnier的epsilon参数
):
    """
    多尺度时序一致性Loss
    
    🔥 改进（基于ProPainter/E2FGVI等最新研究）：
    1. scales=[1,4,8,16]: 适合81帧长视频，捕捉短中长期依赖
    2. use_full_frame=True: 推荐全帧计算，避免字幕区域时序断裂  
    3. subtitle_weight=0.15: 🔥降低字幕区域约束（原0.5），避免残影锁定
    4. use_charbonnier=True: 🔥使用Charbonnier替代L2，对大变化更鲁棒
    
    Args:
        pred_frames: [B, T, C, H, W] 预测的帧序列
        mask_frames: [B, T, H, W] 字幕mask序列（1=字幕，0=背景）
        scales: 时间尺度列表
        lambda_temporal: 总权重
        use_full_frame: True=全帧，False=只背景（旧方法）
        subtitle_weight: 字幕区域权重（0-1，🔥推荐0.15，原0.5）
        use_charbonnier: 是否使用Charbonnier loss（更鲁棒）
        charbonnier_eps: Charbonnier的epsilon参数
    """
    B, T, C, H, W = pred_frames.shape
    
    if T < 2:
        return torch.tensor(0.0, device=pred_frames.device)
    
    total_loss = 0.0
    total_pairs = 0
    
    for scale in scales:
        if T <= scale:
            continue
        
        scale_weight = 1.0 / scale
        scale_loss = 0.0
        num_pairs = 0
        
        for t in range(T - scale):
            frame_t = pred_frames[:, t]
            frame_t_s = pred_frames[:, t+scale]
            
            # 🔥 计算帧差：Charbonnier loss（更鲁棒）或L2
            if use_charbonnier:
                # Charbonnier: sqrt(diff^2 + eps^2)
                # 对大变化不过度惩罚，对小抖动仍敏感
                diff_squared = (frame_t - frame_t_s) ** 2
                frame_diff = torch.sqrt(diff_squared + charbonnier_eps ** 2)  # [B, C, H, W]
            else:
                # 传统L2
                frame_diff = (frame_t - frame_t_s) ** 2
            
            if use_full_frame:
                # 🔥 方案A：全帧计算（推荐）
                if mask_frames is not None and subtitle_weight < 1.0:
                    mask_t = mask_frames[:, t]
                    mask_t_s = mask_frames[:, t+scale]
                    
                    # 上采样mask
                    if mask_t.shape[-2:] != (H, W):
                        mask_t_resized = F.interpolate(
                            mask_t.unsqueeze(1),
                            size=(H, W),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)
                        mask_t_s_resized = F.interpolate(
                            mask_t_s.unsqueeze(1),
                            size=(H, W),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)
                    else:
                        mask_t_resized = mask_t
                        mask_t_s_resized = mask_t_s
                    
                    # 权重图：背景=1.0，字幕=subtitle_weight
                    mask_union = torch.clamp(mask_t_resized + mask_t_s_resized, 0, 1)
                    weight_map = 1.0 - mask_union * (1.0 - subtitle_weight)
                    weight_map = weight_map.unsqueeze(1)  # [B, 1, H, W]
                    
                    weighted_diff = frame_diff * weight_map
                    pair_loss = weighted_diff.sum() / (weight_map.sum() * C + 1e-6)
                else:
                    # 完全等权重
                    pair_loss = frame_diff.mean()
            else:
                # 🔥 方案B：只背景（旧方法）
                mask_t = mask_frames[:, t]
                mask_t_s = mask_frames[:, t+scale]
                
                # 上采样mask
                if mask_t.shape[-2:] != (H, W):
                    mask_t_resized = F.interpolate(
                        mask_t.unsqueeze(1),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                    mask_t_s_resized = F.interpolate(
                        mask_t_s.unsqueeze(1),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                else:
                    mask_t_resized = mask_t
                    mask_t_s_resized = mask_t_s
                
                # 只背景
                mask_union = torch.clamp(mask_t_resized + mask_t_s_resized, 0, 1)
                background_mask = 1.0 - mask_union
                background_mask = background_mask.unsqueeze(1)
                
                weighted_diff = frame_diff * background_mask
                pair_loss = weighted_diff.sum() / (background_mask.sum() * C + 1e-6)
            
            scale_loss += pair_loss * scale_weight
            num_pairs += 1
        
        if num_pairs > 0:
            total_loss += scale_loss / num_pairs
            total_pairs += 1
    
    if total_pairs == 0:
        return torch.tensor(0.0, device=pred_frames.device)
    
    temporal_loss = total_loss / total_pairs
    return lambda_temporal * temporal_loss


# ========== Mask Predictor（复用旧的Adapter）==========

def create_mask_predictor(adapter_checkpoint_path, device):
    """创建mask预测器（使用旧的Stage1 Adapter）"""
    print(f"\n加载Mask预测器: {adapter_checkpoint_path}")
    
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
    
    # 加载权重
    checkpoint = torch.load(adapter_checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 处理DDP前缀
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    adapter.load_state_dict(state_dict, strict=False)
    adapter.eval()
    
    # 冻结所有参数
    for param in adapter.parameters():
        param.requires_grad = False
    
    # 🔥 确保adapter使用float32（与训练时一致）
    # adapter不参与梯度计算，保持float32可以避免类型不匹配
    adapter = adapter.float()
    
    print("✓ Mask预测器加载完成（冻结，float32）")
    return adapter


def save_mask_visualization(
    original_frame,
    original_mask,
    dilated_mask,
    save_path,
    step
):
    """
    保存mask可视化：原图、原始mask、dilate后的mask三张图并列
    
    Args:
        original_frame: [3, H, W] 原始图像，范围[0, 1]
        original_mask: [H, W] 原始mask，范围[0, 1]
        dilated_mask: [H, W] dilate后的mask，范围[0, 1]
        save_path: 保存目录
        step: 当前训练步数
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 转换为numpy
    frame_np = (original_frame.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    original_mask_np = (original_mask.cpu().numpy() * 255).astype(np.uint8)
    dilated_mask_np = (dilated_mask.cpu().numpy() * 255).astype(np.uint8)
    
    # 创建彩色mask（红色表示mask区域）
    original_mask_color = np.zeros((original_mask_np.shape[0], original_mask_np.shape[1], 3), dtype=np.uint8)
    original_mask_color[:, :, 0] = original_mask_np  # 红色通道
    
    dilated_mask_color = np.zeros((dilated_mask_np.shape[0], dilated_mask_np.shape[1], 3), dtype=np.uint8)
    dilated_mask_color[:, :, 0] = dilated_mask_np  # 红色通道
    
    # 横向拼接三张图
    combined = np.hstack([frame_np, original_mask_color, dilated_mask_color])
    
    # 添加标签
    label_height = 30
    label_width = combined.shape[1]
    label_area = np.ones((label_height, label_width, 3), dtype=np.uint8) * 255
    
    # 添加文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    
    # 计算每个区域的中心位置
    section_width = frame_np.shape[1]
    
    cv2.putText(label_area, "Original", (section_width//2 - 40, 20), 
                font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(label_area, "Original Mask", (section_width + section_width//2 - 60, 20), 
                font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(label_area, "Dilated Mask", (2*section_width + section_width//2 - 55, 20), 
                font, font_scale, (0, 0, 0), font_thickness)
    
    # 添加分隔线
    cv2.line(label_area, (section_width, 0), (section_width, label_height), (200, 200, 200), 2)
    cv2.line(label_area, (2*section_width, 0), (2*section_width, label_height), (200, 200, 200), 2)
    
    # 拼接标签和图像
    final_img = np.vstack([label_area, combined])
    
    # 保存
    save_file = os.path.join(save_path, f"mask_vis_step{step:06d}.jpg")
    cv2.imwrite(save_file, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    
    return save_file


def align_mask_to_latent_precise(mask, target_frames):
    """
    精确对齐mask到VAE latent的时间维度（使用chunk+max）
    
    🔥 修复：Wan2.1 VAE的实际压缩比例是3:1（而非4:1）
    - 81帧 → 27帧（81/3 = 27）
    - 使用均匀分块：每个latent帧对应约3个pixel帧
    
    对于mask，使用max（并集）确保不遗漏任何字幕
    
    Args:
        mask: [B, T_pixel, H, W] 原始mask
        target_frames: T_latent
    
    Returns:
        [B, T_latent, H, W] 对齐后的mask
    """
    B, T_pixel, H, W = mask.shape
    
    # 🔥 使用实际的压缩比例计算chunk大小
    # chunk_size是平均每个latent帧对应多少个pixel帧
    chunk_size = T_pixel / target_frames
    
    aligned_frames = []
    
    # 🔥 均匀分块映射（更通用，支持任意压缩比例）
    for i in range(target_frames):
        # 计算当前latent帧对应的pixel帧范围
        start_idx = int(i * chunk_size)
        end_idx = int((i + 1) * chunk_size)
        end_idx = min(end_idx, T_pixel)  # 避免越界
        
        # 确保至少有1帧
        if start_idx >= T_pixel:
            start_idx = T_pixel - 1
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        
        # 取chunk的最大值（并集）
        chunk_mask = mask[:, start_idx:end_idx, :, :]  # [B, ~3, H, W]
        
        # 安全检查：确保chunk不为空
        if chunk_mask.shape[1] == 0:
            # 如果chunk为空，使用最后一帧
            max_mask = mask[:, -1, :, :]
        else:
            max_mask = chunk_mask.max(dim=1, keepdim=False)[0]  # [B, H, W]
        
        aligned_frames.append(max_mask)
    
    aligned_mask = torch.stack(aligned_frames, dim=1)  # [B, T_latent, H, W]
    return aligned_mask


def align_mask_to_latent(mask, target_height, target_width, target_frames=None):
    """
    对齐mask到latent空间维度
    
    Args:
        mask: [B, T, H, W] 原始mask
        target_height: latent空间高度
        target_width: latent空间宽度  
        target_frames: latent空间帧数（如果None则不对齐时间维度）
    
    Returns:
        [B, T_out, H_out, W_out] 对齐后的mask
    """
    B, T, H, W = mask.shape
    
    # Step 1: 时间对齐（如果需要）
    if target_frames is not None and T != target_frames:
        # 🔥 优先使用精确的chunk+max方法
        try:
            mask = align_mask_to_latent_precise(mask, target_frames)
        except Exception as e:
            print(f"⚠️ 精确对齐失败: {e}，尝试插值方法")
            try:
                # 备选方案1: 使用旧项目的对齐函数
                mask = align_mask_to_vae_temporal(
                    mask,
                    T_output=target_frames,
                    mode='causal_uniform',
                    threshold=0.5
                )
            except Exception as e2:
                print(f"⚠️ 时间对齐失败: {e2}，使用简单插值")
                # 备选方案2: 简单插值
                mask = F.interpolate(
                    mask.unsqueeze(1),  # [B, 1, T, H, W]
                    size=(target_frames, H, W),
                    mode='trilinear',
                    align_corners=False
                ).squeeze(1)
    
    # Step 2: 空间对齐（使用max_pool保证并集语义）
    T_current = mask.shape[1]
    if H != target_height or W != target_width:
        # 🔥 计算下采样因子（应该是整数，通常是8）
        spatial_factor_h = H // target_height
        spatial_factor_w = W // target_width
        
        # 验证是否是整数倍下采样
        if H == target_height * spatial_factor_h and W == target_width * spatial_factor_w:
            # 🔥 方法1: 精确的max_pool（推荐）
            # 每个latent位置对应spatial_factor × spatial_factor的patch
            aligned_frames = []
            for t in range(T_current):
                frame = mask[:, t, :, :]  # [B, H, W]
                # 使用max_pool2d: 每个patch取最大值（并集）
                frame_aligned = F.max_pool2d(
                    frame.unsqueeze(1),  # [B, 1, H, W]
                    kernel_size=(spatial_factor_h, spatial_factor_w),
                    stride=(spatial_factor_h, spatial_factor_w),
                    padding=0
                ).squeeze(1)  # [B, H_target, W_target]
                aligned_frames.append(frame_aligned)
            mask = torch.stack(aligned_frames, dim=1)  # [B, T, H_target, W_target]
        else:
            # 🔥 方法2: 非整数倍时的备用方案
            # 先用adaptive_max_pool确保不丢失，再微调
            print(f"⚠️ 非整数倍下采样: {H}→{target_height} ({spatial_factor_h:.2f}×), {W}→{target_width} ({spatial_factor_w:.2f}×)")
            aligned_frames = []
            for t in range(T_current):
                frame = mask[:, t, :, :]  # [B, H, W]
                # adaptive_max_pool2d自动调整
                frame_aligned = F.adaptive_max_pool2d(
                    frame.unsqueeze(1),  # [B, 1, H, W]
                    output_size=(target_height, target_width)
                ).squeeze(1)  # [B, H_target, W_target]
                aligned_frames.append(frame_aligned)
            mask = torch.stack(aligned_frames, dim=1)  # [B, T, H_target, W_target]
    
    return mask


def predict_and_process_masks(
    mask_predictor,
    subtitle_frames,
    dilation_kernel=3,
    enable_random_blackout=True,
    device='cuda',
    save_visualization=False,
    vis_save_path=None,
    global_step=0
):
    """
    预测mask并处理（dilate + 染色）
    
    🔥 优化：batch处理所有帧（而非逐帧循环）
    - 之前: 81次ResNet forward（逐帧）
    - 现在: 1次ResNet forward（batch=81）
    - 提升: 50-70%
    
    Args:
        mask_predictor: Stage1 Adapter
        subtitle_frames: [B, T, 3, H, W] RGB图像，范围[0, 1]
        dilation_kernel: dilate kernel大小
        enable_random_blackout: 是否启用随机染色
        save_visualization: 是否保存可视化
        vis_save_path: 可视化保存路径
        global_step: 当前训练步数
    
    Returns:
        processed_masks: [B, T, H, W] 处理后的mask
        processed_frames: [B, T, 3, H, W] 染色后的frames（如果染色）
    """
    B, T, C, H, W = subtitle_frames.shape
    
    with torch.no_grad():
        # 🔥 关键优化：reshape为 [B*T, C, H, W]，batch处理所有帧
        frames_normalized = subtitle_frames * 2 - 1  # [0,1] → [-1,1]
        frames_flat = frames_normalized.reshape(B * T, C, H, W)
        
        # 🔥 一次forward处理所有81帧！（而非81次循环）
        _, _, mask_logits, _ = mask_predictor(frames_flat, return_all=True)
        predicted_mask_flat = torch.sigmoid(mask_logits)  # (B*T, 1, H, W)
        
        # 保存原始mask（用于"只在dilate区染色"）
        original_mask_flat = predicted_mask_flat.clone()
        
        # 决定是否调整dilation_kernel
        base_dilation_kernel = dilation_kernel
        blackout_mode = "none"
        
        if enable_random_blackout:
            rand_val = torch.rand(1, device=device).item()
            
            if rand_val < 0.1:
                blackout_mode = "random_color"  # 20%：染彩色
            elif rand_val < 0.2:
                blackout_mode = "full"  # 20%：染黑色
            elif rand_val < 0.6:
                blackout_mode = "dilated_only_color"  # 50%：只在dilate区染色
                dilation_kernel = base_dilation_kernel + 10  # kernel从3变13
            else:
                blackout_mode = "none"  # 10%：不染色
        
        # Dilate mask
        if dilation_kernel > 0:
            padding = dilation_kernel // 2
            predicted_mask_flat = F.max_pool2d(
                predicted_mask_flat,
                kernel_size=dilation_kernel,
                stride=1,
                padding=padding
            )
        
        # Reshape masks
        predicted_mask = predicted_mask_flat.reshape(B, T, *predicted_mask_flat.shape[2:]).squeeze(2)  # [B, T, H, W]
        original_mask = original_mask_flat.reshape(B, T, *original_mask_flat.shape[2:]).squeeze(2)
        
        # 🎨 保存可视化（取第一个batch的第一帧）
        if save_visualization and vis_save_path is not None and B > 0 and T > 0:
            # 选择中间帧进行可视化（更有代表性）
            vis_frame_idx = T // 2
            save_mask_visualization(
                original_frame=subtitle_frames[0, vis_frame_idx],  # [3, H, W]
                original_mask=original_mask[0, vis_frame_idx],      # [H, W]
                dilated_mask=predicted_mask[0, vis_frame_idx],      # [H, W]
                save_path=vis_save_path,
                step=global_step
            )
        
        # 染色操作
        processed_frames = subtitle_frames.clone()
        
        if enable_random_blackout and blackout_mode != "none":
            # 选择要处理的区域
            if blackout_mode == "full":
                binary_mask = (predicted_mask > 0.5).float()
            elif blackout_mode == "dilated_only_color":
                dilated_binary = (predicted_mask > 0.5).float()
                original_binary = (original_mask > 0.5).float()
                binary_mask = torch.clamp(dilated_binary - original_binary, 0, 1)
            else:  # random_color
                binary_mask = (predicted_mask > 0.5).float()
            
            # 扩展mask到RGB通道 [B, T, 3, H, W]
            binary_mask_rgb = binary_mask.unsqueeze(2).expand(-1, -1, 3, -1, -1)
            
            if blackout_mode == "random_color":
                # 随机彩色
                random_color = torch.rand(B, 1, 3, 1, 1, device=device)
                random_color = random_color.expand(-1, T, -1, H, W)
                processed_frames = processed_frames * (1 - binary_mask_rgb) + random_color * binary_mask_rgb
            elif blackout_mode == "dilated_only_color":
                # 深灰到纯黑
                black_value = torch.rand(B, 1, 3, 1, 1, device=device) * 0.1
                black_color = black_value.expand(-1, T, -1, H, W)
                processed_frames = processed_frames * (1 - binary_mask_rgb) + black_color * binary_mask_rgb
            else:  # full
                # 纯黑
                processed_frames = processed_frames * (1 - binary_mask_rgb)
    
    return predicted_mask, processed_frames


# ========== Dataset（从non_text.json加载）==========

class VideoDataset(Dataset):
    """从non_text.json加载视频对（参考旧训练脚本的逻辑）"""
    def __init__(
        self,
        non_text_json_path,
        clean_base_dirs,
        subtitle_base_dirs,
        num_frames=81,
        max_samples=1000,
        random_seed=42,
    ):
        self.num_frames = num_frames
        self.random_seed = random_seed
        self.clean_base_dirs = clean_base_dirs if isinstance(clean_base_dirs, list) else [clean_base_dirs]
        self.subtitle_base_dirs = subtitle_base_dirs if isinstance(subtitle_base_dirs, list) else [subtitle_base_dirs]
        
        # 加载non_text.json作为过滤列表
        if non_text_json_path and os.path.exists(non_text_json_path):
            with open(non_text_json_path, 'r', encoding='utf-8') as f:
                items = json.load(f)
            
            def _to_video_name(s: str) -> str:
                # "1148_22175_0.jpg" -> "1148_22175_0"
                s = s.rsplit('.', 1)[0]
                return s
            
            allowed_samples = set(_to_video_name(s) for s in items)
            print(f"✓ 从non_text.json加载了{len(allowed_samples)}个允许的样本ID")
        else:
            allowed_samples = None
            print("⚠️ 未找到non_text.json，将使用所有视频对")
        
        # 收集所有视频对
        self.video_pairs = []
        self._collect_video_pairs(allowed_samples)
        
        # 随机抽取max_samples个样本
        if max_samples and len(self.video_pairs) > max_samples:
            rng = np.random.RandomState(random_seed)
            idxs = rng.choice(len(self.video_pairs), size=max_samples, replace=False)
            self.video_pairs = [self.video_pairs[i] for i in idxs]
        
        print(f"✓ 最终使用{len(self.video_pairs)}个视频对进行训练")
    
    def _collect_video_pairs(self, allowed_samples):
        """收集视频对（参考旧脚本的match_video_pairs逻辑）"""
        # 扫描clean和subtitle目录，匹配视频对
        for clean_dir in self.clean_base_dirs:
            for subtitle_dir in self.subtitle_base_dirs:
                if not os.path.exists(clean_dir) or not os.path.exists(subtitle_dir):
                    continue
                
                # 获取所有视频文件
                clean_videos = {}
                for fname in os.listdir(clean_dir):
                    if fname.endswith('.mp4'):
                        video_name = fname.rsplit('.', 1)[0]  # 去掉.mp4
                        clean_videos[video_name] = os.path.join(clean_dir, fname)
                
                subtitle_videos = {}
                for fname in os.listdir(subtitle_dir):
                    if fname.endswith('.mp4'):
                        # 🔥 关键修复：subtitle视频去掉"_with_subtitle"后缀来匹配
                        base_name = fname.rsplit('.', 1)[0]  # 去掉.mp4
                        base_name = base_name.replace('_with_subtitle', '')  # 去掉后缀
                        subtitle_videos[base_name] = os.path.join(subtitle_dir, fname)
                
                # 匹配视频对
                for video_name in clean_videos:
                    if video_name in subtitle_videos:
                        # 如果有过滤列表，检查是否在列表中
                        if allowed_samples is None or video_name in allowed_samples:
                            self.video_pairs.append((
                                clean_videos[video_name],
                                subtitle_videos[video_name]
                            ))
        
        print(f"✓ 收集到{len(self.video_pairs)}个视频对")
    
    def __len__(self):
        return len(self.video_pairs)
    
    def __getitem__(self, idx):
        clean_path, subtitle_path = self.video_pairs[idx]
        
        # 🔥 第一次加载时打印提示
        if not hasattr(self, '_first_load_done'):
            self._first_load_start = time.time()
            print(f"\n⏳ [GPU-{idx%8}] 正在加载第一个视频样本...")
            print(f"   Clean: {os.path.basename(clean_path)}")
            print(f"   Subtitle: {os.path.basename(subtitle_path)}")
            print(f"   需要加载: {self.num_frames}帧 × 720P")
        
        # 加载视频
        clean_frames = self.load_video(clean_path)
        subtitle_frames = self.load_video(subtitle_path)
        
        # 🔥 第一次加载完成
        if not hasattr(self, '_first_load_done'):
            print(f"✓ [GPU-{idx%8}] 视频加载完成: {time.time()-self._first_load_start:.2f}秒")
            self._first_load_done = True
        
        # 随机采样连续的num_frames帧
        total_frames = min(len(clean_frames), len(subtitle_frames))
        
        if total_frames >= self.num_frames:
            max_start_idx = total_frames - self.num_frames
            start_idx = np.random.randint(0, max_start_idx + 1)
            end_idx = start_idx + self.num_frames
            
            clean_frames = clean_frames[start_idx:end_idx]
            subtitle_frames = subtitle_frames[start_idx:end_idx]
        else:
            # 视频太短，从头开始取
            clean_frames = clean_frames[:total_frames]
            subtitle_frames = subtitle_frames[:total_frames]
        
        return {
            'video': clean_frames,  # 目标（无字幕）
            'control_video': subtitle_frames,  # 输入（有字幕）
            'prompt': "Please remove the subtitle text from the video while preserving the character appearance, background composition, and color style. Do not add any new elements."
        }
    
    def crop_and_resize_image(self, image, target_height, target_width):
        """
        🔥 参考DiffSynth-Studio的ImageCropAndResize实现
        
        智能裁剪和缩放：
        1. 计算scale使得图像覆盖目标尺寸（scale = max(tw/w, th/h)）
        2. resize到scale后的尺寸
        3. center crop到目标尺寸
        
        这样可以保持宽高比，避免图像拉伸变形
        """
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        
        # Resize to cover target size
        new_height = round(height * scale)
        new_width = round(width * scale)
        image = TF.resize(
            image,
            (new_height, new_width),
            interpolation=TF.InterpolationMode.BILINEAR
        )
        
        # Center crop to exact target size
        image = TF.center_crop(image, (target_height, target_width))
        return image
    
    def get_target_resolution(self, video_path, max_pixels=1280*720, division_factor=16):
        """
        🔥 参考DiffSynth-Studio的动态分辨率计算
        
        根据视频原始分辨率和max_pixels计算目标分辨率：
        1. 如果原始分辨率 <= max_pixels，使用原始（对齐到division_factor）
        2. 如果超过，缩放到max_pixels
        3. 确保尺寸是division_factor的倍数（VAE要求）
        4. 自动判断横竖屏，保持宽高比
        """
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        
        if not ret:
            # 🔥 改进：尝试从视频属性获取分辨率
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
        
            # 如果还是获取失败，根据文件名或默认横屏
            if orig_h == 0 or orig_w == 0:
                print(f"⚠️ 无法读取视频分辨率: {video_path}，使用默认720P横屏")
                return 720, 1280
        else:
            orig_h, orig_w = first_frame.shape[:2]
            cap.release()
        
        # 如果超过max_pixels，等比缩小
        if orig_w * orig_h > max_pixels:
            scale = (orig_w * orig_h / max_pixels) ** 0.5
            target_h = int(orig_h / scale)
            target_w = int(orig_w / scale)
        else:
            target_h = orig_h
            target_w = orig_w
        
        # 对齐到division_factor（VAE下采样要求）
        target_h = target_h // division_factor * division_factor
        target_w = target_w // division_factor * division_factor
        
        return target_h, target_w
    
    def load_video(self, video_path, use_smart_resize=True, max_pixels=1280*720):
        """
        加载视频，返回PIL Image列表
        
        Args:
            video_path: 视频路径
            use_smart_resize: 是否使用智能裁剪（DiffSynth-Studio方式）
            max_pixels: 最大像素数（默认720P）
        
        🔥 改进：参考DiffSynth-Studio的数据增强
        - 自动计算目标分辨率（保持宽高比）
        - 智能裁剪（不拉伸变形）
        - 对齐到16的倍数（VAE要求）
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # 计算目标分辨率
        target_h, target_w = self.get_target_resolution(video_path, max_pixels, division_factor=16)
        
        # 读取所有帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            
            # 应用智能裁剪或简单resize
            if use_smart_resize:
                pil_img = self.crop_and_resize_image(pil_img, target_h, target_w)
            else:
                pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
            
            frames.append(pil_img)
        
        cap.release()
        return frames


# ========== Training Module（融合DiffSynth-Studio + Focal Loss）==========

class WanTrainingModuleWithFocalLoss(nn.Module):
    """
    融合DiffSynth-Studio的WanVideoPipeline和旧的Focal Loss逻辑
    """
    def __init__(
        self,
        model_base_path,
        mask_predictor,
        lora_rank=64,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        focal_loss_alpha=5.0,
        temporal_loss_weight=0.1,
        mask_dilation_kernel=3,
        enable_random_blackout=True,
        use_gradient_checkpointing=True,
        vis_save_path=None,
        use_custom_loss=True,
        use_tiled_vae=False,
        vae_chunk_size=None,
        use_uniform_timestep_sampling=False,
    ):
        super().__init__()
        
        # 保存可视化路径
        self.vis_save_path = vis_save_path
        self.use_custom_loss = use_custom_loss
        self.use_uniform_timestep_sampling = use_uniform_timestep_sampling
        
        # 加载WanVideoPipeline
        print(f"\n加载Wan2.1-1.3B-Control模型: {model_base_path}")
        
        # 🔥 强制禁用modelscope下载
        os.environ['MODELSCOPE_CACHE'] = os.path.dirname(model_base_path)
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        # 🔥 关键修复：使用skip_download和path参数，直接加载本地文件
        import glob
        
        # 查找所有diffusion模型文件
        diffusion_files = sorted(glob.glob(os.path.join(model_base_path, "diffusion_pytorch_model*.safetensors")))
        if not diffusion_files:
            raise FileNotFoundError(f"No diffusion_pytorch_model*.safetensors found in {model_base_path}")
        
        print(f"\n📁 找到的模型文件:")
        print(f"  DiT: {len(diffusion_files)} files")
        for f in diffusion_files:
            print(f"    - {os.path.basename(f)}")
        
        # 构建model_configs - 使用path参数直接指定文件，跳过下载
        model_configs = []
        
        # DiT模型文件
        model_configs.append(ModelConfig(
            path=diffusion_files,  # 直接传入文件列表
            skip_download=True,  # 跳过下载
            local_model_path=model_base_path  # 明确本地路径
        ))
        
        # T5文本编码器
        t5_path = os.path.join(model_base_path, "models_t5_umt5-xxl-enc-bf16.pth")
        if os.path.exists(t5_path):
            print(f"  T5: {os.path.basename(t5_path)}")
            model_configs.append(ModelConfig(
                path=t5_path,
                skip_download=True,
                local_model_path=model_base_path
            ))
        else:
            raise FileNotFoundError(f"T5 model not found: {t5_path}")
        
        # VAE
        vae_path = os.path.join(model_base_path, "Wan2.1_VAE.pth")
        if os.path.exists(vae_path):
            print(f"  VAE: {os.path.basename(vae_path)}")
            model_configs.append(ModelConfig(
                path=vae_path,
                skip_download=True,
                local_model_path=model_base_path
            ))
        else:
            raise FileNotFoundError(f"VAE not found: {vae_path}")
        
        # CLIP文本编码器
        clip_path = os.path.join(model_base_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
        if os.path.exists(clip_path):
            print(f"  CLIP: {os.path.basename(clip_path)}")
            model_configs.append(ModelConfig(
                path=clip_path,
                skip_download=True,
                local_model_path=model_base_path
            ))
        else:
            raise FileNotFoundError(f"CLIP model not found: {clip_path}")
        
        print(f"✓ 配置了{len(model_configs)}个模型文件（全部本地加载，跳过下载）")
        
        # 🔥 Monkey patch: 完全禁用download_if_necessary
        original_download = ModelConfig.download_if_necessary
        ModelConfig.download_if_necessary = lambda self, use_usp=False: None
        
        try:
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,  # 使用bf16数据类型
                device="cpu",
                model_configs=model_configs
            )
        finally:
            # 恢复原函数
            ModelConfig.download_if_necessary = original_download
        
        print("✓ WanVideoPipeline加载完成")
        # 检查模型数据类型
        try:
            print(f"  模型数据类型: {next(self.pipe.dit.parameters()).dtype}")
        except:
            print(f"  模型数据类型: bfloat16 (设定)")
        
        # 🔥 初始化 prompter（text_encoder + tokenizer）
        # 参考官方 from_pretrained 的实现（第409-410行）
        if self.pipe.text_encoder is not None:
            self.pipe.prompter.fetch_models(text_encoder=self.pipe.text_encoder)
            print("✓ Prompter已连接text_encoder")
        
        # 🔥 加载tokenizer（从本地模型目录）
        tokenizer_path = os.path.join(model_base_path, "google/umt5-xxl")
        if os.path.exists(tokenizer_path):
            self.pipe.prompter.fetch_tokenizer(tokenizer_path)
            print(f"✓ Prompter已加载tokenizer: {tokenizer_path}")
        else:
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        # 注入LoRA
        print(f"\n注入LoRA (rank={lora_rank}, targets={lora_target_modules})")
        from peft import LoraConfig, inject_adapter_in_model
        
        # 解析target modules
        target_list = [m.strip() for m in lora_target_modules.split(',')]
        
        # 创建LoRA配置
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,  # alpha = rank
            target_modules=target_list,
            lora_dropout=0.0,
            bias="none",
        )
        
        # 对DiT模型注入LoRA
        self.pipe.dit = inject_adapter_in_model(lora_config, self.pipe.dit)
        
        print("✓ LoRA注入完成")
        
        # 🔥 【步骤1】先冻结DiT的base model参数（只保留LoRA可训练）
        for name, param in self.pipe.dit.named_parameters():
            if 'lora' not in name.lower():
                param.requires_grad = False
        
        # 🔥 【步骤2】冻结其他所有组件
        for component_name, component in [
            ('text_encoder', self.pipe.text_encoder),
            ('vae', self.pipe.vae), 
            ('image_encoder', self.pipe.image_encoder)
        ]:
            if component is not None:
                component.eval()
                for param in component.parameters():
                    param.requires_grad = False
                frozen_count = sum(p.numel() for p in component.parameters())
                print(f"  🔒 {component_name} 冻结: {frozen_count:,} 参数")
        
        # 设置DiT训练模式
        self.pipe.dit.train()
        
        # 🔥 【步骤3】现在再统计参数（确保冻结生效）
        total_params = sum(p.numel() for p in self.pipe.dit.parameters())
        trainable_params = sum(p.numel() for p in self.pipe.dit.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for n, p in self.pipe.dit.named_parameters() if 'lora' in n.lower())
        
        # 🔥 统计额外的可学习参数（如temporal_loss_weight）
        # 使用id()比较对象标识，避免张量值比较导致的形状不匹配错误
        dit_param_ids = {id(p) for p in self.pipe.dit.parameters()}
        extra_trainable_params = [p for p in self.parameters() if p.requires_grad and id(p) not in dit_param_ids]
        extra_params = sum(p.numel() for p in extra_trainable_params)
        
        print(f"\n📊 参数统计:")
        print(f"  总参数: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"  可训练参数 (DiT): {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"  LoRA参数: {lora_params:,} ({lora_params/1e6:.2f}M)")
        print(f"  额外可学习参数: {extra_params} 🎓")
        
        # 🔥 调试：如果额外参数异常多，打印详情
        if extra_params > 100:
            print(f"\n⚠️  警告：检测到异常多的额外参数 ({extra_params:,})，预期只有1个")
            print(f"   正在检查各组件的冻结状态...")
            
            # 检查各组件
            components_to_check = [
                ('text_encoder', self.pipe.text_encoder),
                ('vae', self.pipe.vae),
                ('image_encoder', self.pipe.image_encoder),
                ('mask_predictor', self.mask_predictor),
            ]
            
            for name, component in components_to_check:
                if component is not None:
                    trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
                    total = sum(p.numel() for p in component.parameters())
                    if trainable > 0:
                        print(f"   ❌ {name}: {trainable:,}/{total:,} 参数可训练（应全部冻结！）")
                    else:
                        print(f"   ✓ {name}: 已正确冻结")
        
        print(f"  LoRA占比: {lora_params/total_params*100:.4f}%")
        
        # 🔥 验证只有LoRA参数和额外参数可训练
        if trainable_params != lora_params:
            print(f"\n⚠️  警告：可训练参数({trainable_params/1e6:.2f}M) != LoRA参数({lora_params/1e6:.2f}M)")
            print(f"   差异: {(trainable_params - lora_params):,} 个参数")
            if extra_params > 0:
                print(f"   其中 {extra_params} 个是额外的可学习参数（如temporal_loss_weight）✓")
                if trainable_params - lora_params > extra_params:
                    print(f"   ⚠️  还有 {trainable_params - lora_params - extra_params} 个未知可训练参数！")
            else:
                print(f"   可能有base model参数未冻结！这会导致：")
                print(f"   - Backward极慢（更新{trainable_params/1e9:.2f}B参数）")
                print(f"   - 显存爆炸（存储{trainable_params/1e9:.2f}B梯度）")
        else:
            print(f"✓ 验证通过：只有LoRA参数可训练")
            if extra_params > 0:
                print(f"  （+ {extra_params} 个额外可学习参数）")
        
        # 保存配置（冻结操作已在统计前完成）
        self.mask_predictor = mask_predictor
        self.focal_loss_alpha = focal_loss_alpha  # 初始值（会被动态调度覆盖）
        self.initial_focal_alpha = focal_loss_alpha  # 🔥保存初始alpha用于动态调度
        
        # 🔥 将temporal_loss_weight改为可学习参数
        self.temporal_loss_weight = nn.Parameter(
            torch.tensor(temporal_loss_weight, dtype=torch.float32),
            requires_grad=True
        )
        print(f"  🎓 Temporal loss weight设为可学习参数，初始值: {temporal_loss_weight}")
        
        self.mask_dilation_kernel = mask_dilation_kernel
        self.enable_random_blackout = enable_random_blackout
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_dynamic_alpha = True  # 🔥启用动态alpha调度
        self.use_tiled_vae = use_tiled_vae  # 🔥 AMD GPU推荐
        self.vae_chunk_size = vae_chunk_size  # 🔥 自定义chunk大小
        
        # 🔥 创建Scheduler（使用与pipeline相同的配置）
        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=1000,
            shift=5,
            sigma_min=0.0,
            extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)  # 训练模式
        
        print(f"\n训练配置:")
        print(f"  Focal loss alpha: {focal_loss_alpha}")
        print(f"  Temporal loss weight (可学习): {temporal_loss_weight} (初始值)")
        print(f"  Mask dilation kernel: {mask_dilation_kernel}")
        print(f"  Random blackout: {enable_random_blackout}")
        print(f"  Scheduler: FlowMatchScheduler (shift=5, num_train_timesteps=1000)")
        
        # 🔥 使用自定义loss（完整实现focal loss）
        self.use_custom_loss = True
    
    # 🔥 删除重复的forward_with_custom_loss定义
    # 保留下面统一的实现
    
    def forward(self, batch, global_step=0, save_visualization=False):
        """
        训练forward pass - 根据use_custom_loss标志选择实现
        """
        if self.use_custom_loss:
            return self.forward_with_custom_loss(batch, global_step, save_visualization)
        else:
            return self.forward_with_pipeline_loss(batch, global_step, save_visualization)
    
    def forward_with_custom_loss(self, batch, global_step=0, save_visualization=False):
        """
        🔥 优化版：减少格式转换，统一使用 [B, C, T, H, W] 格式
        
        格式约定：
        - VAE: 输入/输出 [B, C, T, H, W] ✅
        - DiT: 输入/输出 [B, C, T, H, W] ✅
        - Mask predictor: 需要 [B, T, C, H, W] ⚠️ (逐帧处理)
        
        总转换次数: 2次 (原来6+次)
        """
        # 🔥 第一个step详细时间统计
        if global_step == 0:
            import time
            step_start = time.time()
            print("\n" + "="*80)
            print("📊 第一个Step详细时间分析")
            print("="*80)
        
        # 🔥 动态Alpha调度
        if self.use_dynamic_alpha:
            current_alpha = get_dynamic_alpha(
                global_step,
                initial_alpha=self.initial_focal_alpha,
                max_alpha=15.0,
                cycle_length=20
            )
            self.focal_loss_alpha = current_alpha
        
        clean_frames = batch['video']  # List of PIL Images
        subtitle_frames_pil = batch['control_video']  # List of PIL Images
        prompts = batch['prompt']  # List of strings
        
        # ============ Step 1: 数据准备 (统一使用 [B, C, T, H, W]) ============
        if global_step == 0:
            t0 = time.time()
        
        # 🔥 直接生成 [B, C, T, H, W] 格式
        clean_video = self.pil_list_to_tensor(clean_frames, target_format='BCTHW').cuda()
        subtitle_video = self.pil_list_to_tensor(subtitle_frames_pil, target_format='BCTHW').cuda()
        B, C, T, H, W = clean_video.shape
        device = clean_video.device
        
        if global_step == 0:
            print(f"✓ Step 1 - PIL→Tensor转换: {time.time()-t0:.2f}秒 | 形状: [{B}, {C}, {T}, {H}, {W}]")
        
        # ============ Step 2: Mask预测 (临时转换1次) ============
        if global_step == 0:
            t1 = time.time()
        
        # 🔥 mask predictor需要 [B, T, C, H, W] 格式
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            subtitle_BTCHW = subtitle_video.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W] → [B,T,C,H,W]
            predicted_mask, processed_subtitle_BTCHW = predict_and_process_masks(
                self.mask_predictor,
                subtitle_BTCHW.float(),
                dilation_kernel=self.mask_dilation_kernel,
                enable_random_blackout=self.enable_random_blackout,
                device=device,
                save_visualization=save_visualization,
                vis_save_path=self.vis_save_path,
                global_step=global_step
            )
            # 🔥 转回 [B, C, T, H, W]
            processed_subtitle_video = processed_subtitle_BTCHW.permute(0, 2, 1, 3, 4)  # 转换2
        
        if global_step == 0:
            print(f"✓ Step 2 - Mask预测+膨胀+染色: {time.time()-t1:.2f}秒")
        
        # ============ Step 3: VAE编码 (保持 [B, C, T, H, W]) ============
        if global_step == 0:
            t2 = time.time()
        
        # 🔥 AMD GPU环境检测和自动配置
        is_amd_gpu = hasattr(torch.version, 'hip') or 'hip' in str(torch.version.cuda).lower()
        
        # 🔥 自动检测最优配置（基于实测数据）
        # Venv环境：BF16最快（0.35秒/3帧）
        # Conda环境：FP32最快（0.71秒/3帧）
        # 判断依据：检查是否在虚拟环境中
        in_virtualenv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        if global_step == 0 and is_amd_gpu:
            print("  🔥 检测到AMD GPU (ROCm)")
            print(f"  📍 环境类型: {'虚拟环境(venv)' if in_virtualenv else '系统/Conda环境'}")
        
        with torch.no_grad():
            # 🔥 VAE编码（参考 DiffSynth-Studio/examples/wanvideo/model_training/train.py）
            clean_for_vae = clean_video * 2 - 1  # [0,1] → [-1,1]
            subtitle_for_vae = processed_subtitle_video * 2 - 1
            
            if global_step == 0:
                t2_1 = time.time()
            
            # 🔥 AMD GPU 优化参数（根据实测数据自动配置）
            # 
            # 实测数据（AMD MI308X）：
            #   Venv环境：BF16 0.35秒/3帧 ✅ 最快！（81帧预估9.5秒）
            #   Conda环境：FP32 0.71秒/3帧 ✅ 较快 （81帧预估18.6秒）
            #   Venv环境：FP32 11.61秒/3帧 ❌ 很慢
            #   Conda环境：BF16 63.54秒/3帧 ❌ 极慢
            #
            # 自动选择策略：
            #   - Venv环境 → 使用 BF16（快33倍）
            #   - Conda/系统环境 → 使用 FP32
            
            use_bf16_for_vae = in_virtualenv  # Venv用BF16，Conda用FP32
            chunk_size = None  # 🔥 不再手动分块，使用pipeline默认
            tiled = True   # 🔥 改为True：与推理保持一致
            
            if global_step == 0:
                if is_amd_gpu:
                    vae_dtype = "BF16" if use_bf16_for_vae else "FP32"
                    print(f"  ⚙️  VAE配置: {vae_dtype}, tiled={tiled}")
            
            # 🔥 现在训练和推理使用相同的VAE配置：
            # - tiled=True: VAE空间分块
            # - 时间维度由VAE内部处理（pipeline默认机制）
            
            # 🎯 使用pipeline的VAE编码（与推理保持一致）
            # 不再手动时间分块，让VAE内部处理
            if global_step == 0:
                print(f"  ⚙️  VAE编码配置: tiled={tiled}, 使用pipeline默认")
            
            # 编码 clean video（使用pipeline的encode方法）
            # video_BCTHW: [B, C, T, H, W]
            B, C, T, H, W = clean_for_vae.shape
            clean_latent_list = []
            for i in range(B):
                video_single = clean_for_vae[i:i+1]  # [1, C, T, H, W]
                # 🔥 VAE.encode期望输入: List[Tensor[C, T, H, W]]
                # 不需要transpose，直接传入 [C, T, H, W] 格式
                
                if use_bf16_for_vae:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        latent = self.pipe.vae.encode(
                            [video_single[0]],  # [C, T, H, W] ✅
                            device=device,
                            tiled=tiled,
                            tile_size=(30, 52),
                            tile_stride=(15, 26)
                        )[0]  # 返回stacked tensor
                else:
                    latent = self.pipe.vae.encode(
                        [video_single[0]],  # [C, T, H, W] ✅
                        device=device,
                        tiled=tiled,
                        tile_size=(30, 52),
                        tile_stride=(15, 26)
                    )[0]
                
                clean_latent_list.append(latent)
            
            clean_latent_BCTHW = torch.stack(clean_latent_list, dim=0)
            # 🔥 确保latent在正确的设备上（VAE tiled模式可能返回CPU tensor）
            clean_latent_BCTHW = clean_latent_BCTHW.to(device=device, dtype=self.pipe.torch_dtype)
            
            if global_step == 0:
                vae_dtype_str = "BF16" if use_bf16_for_vae else "FP32"
                print(f"  ✓ Clean video编码完成: {time.time()-t2_1:.2f}秒 ({vae_dtype_str}, tiled={tiled})")
                t2_2 = time.time()
            
            # 编码 subtitle video（使用相同方式）
            subtitle_latent_list = []
            for i in range(B):
                video_single = subtitle_for_vae[i:i+1]  # [1, C, T, H, W]
                # 🔥 VAE.encode期望输入: List[Tensor[C, T, H, W]]
                
                if use_bf16_for_vae:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        latent = self.pipe.vae.encode(
                            [video_single[0]],  # [C, T, H, W] ✅
                            device=device,
                            tiled=tiled,
                            tile_size=(30, 52),
                            tile_stride=(15, 26)
                        )[0]
                else:
                    latent = self.pipe.vae.encode(
                        [video_single[0]],  # [C, T, H, W] ✅
                        device=device,
                        tiled=tiled,
                        tile_size=(30, 52),
                        tile_stride=(15, 26)
                    )[0]
                
                subtitle_latent_list.append(latent)
            
            subtitle_latent_BCTHW = torch.stack(subtitle_latent_list, dim=0)
            # 🔥 确保latent在正确的设备上（VAE tiled模式可能返回CPU tensor）
            subtitle_latent_BCTHW = subtitle_latent_BCTHW.to(device=device, dtype=self.pipe.torch_dtype)
            
            if global_step == 0:
                vae_dtype_str = "BF16" if use_bf16_for_vae else "FP32"
                print(f"  ✓ Subtitle video编码完成: {time.time()-t2_2:.2f}秒 ({vae_dtype_str}, tiled={tiled})")
        
        # 获取latent维度
        B, C_latent, T_latent, H_latent, W_latent = clean_latent_BCTHW.shape
        
        if global_step == 0:
            print(f"✓ Step 3 - VAE编码总计: {time.time()-t2:.2f}秒 | {T}帧→{T_latent}帧 | {H}×{W}→{H_latent}×{W_latent}")
        
        # ============ Step 4: Mask对齐 ============
        if global_step == 0:
            t3 = time.time()
        
        aligned_mask = None
        if predicted_mask is not None:
            aligned_mask = align_mask_to_latent(
                predicted_mask,
                target_height=H_latent,
                target_width=W_latent,
                target_frames=T_latent
            )  # [B, T, H, W]
        
        if global_step == 0:
            print(f"✓ Step 4 - Mask对齐: {time.time()-t3:.2f}秒")
        
        # ============ Step 5: Prompt编码 ============
        if global_step == 0:
            t4 = time.time()
        
        with torch.no_grad():
            # 🔥 使用 prompter 来编码 prompt（参考 WanVideoUnit_PromptEmbedder）
            prompt_emb = self.pipe.prompter.encode_prompt(
                prompts,
                device=device,
                positive=True
            )
        
        if global_step == 0:
            print(f"✓ Step 5 - Prompt编码: {time.time()-t4:.2f}秒")
        
        # ============ Step 6: Flow Matching (使用Scheduler方法) ============
        if global_step == 0:
            t5 = time.time()
        
        # 🔥 Timestep采样策略：随机采样 vs 均匀采样
        # 🔥 初始化timestep_id_avg（确保在所有情况下都被定义）
        timestep_id_avg = -1
        
        if self.use_uniform_timestep_sampling:
            # 🎯 均匀采样：使用质数步长遍历，保证均匀覆盖所有timesteps
            # 原理：prime_stride与num_train_timesteps互质，可以遍历所有timestep
            # 🔥 改进：从10%位置开始（timestep_id=100），避免第一个step权重过低
            #   例如：1000个timesteps，stride=103（质数）
            #   step 0 → timestep_id 100 (10%位置，t≈0.9，权重中等)
            #   step 1 → timestep_id 203
            #   step 2 → timestep_id 306
            #   ...
            #   经过1000步后，所有timesteps都会被均匀覆盖
            prime_stride = 103  # 质数步长，确保均匀覆盖
            start_offset = int(self.scheduler.num_train_timesteps * 0.1)  # 🔥 从10%位置开始（100）
            timestep_id = torch.tensor([(start_offset + global_step * prime_stride) % self.scheduler.num_train_timesteps for _ in range(B)])
            timestep_id_avg = timestep_id[0].item()  # 🔥 保存用于日志（均匀采样时是确定的）
            if global_step == 0:
                print(f"  ✅ 使用均匀采样timestep（质数步长={prime_stride}，从10%位置开始，均匀覆盖所有timesteps）")
        else:
            # 📊 随机采样：标准扩散模型训练方式
            timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (B,))  # CPU
            # timestep_id_avg 保持为 -1（随机采样时无法确定）
            if global_step == 0:
                print(f"  ✅ 使用随机采样timestep（标准方式）")
        
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=torch.float32, device=device)  # 然后移到GPU
        
        # 🔥 直接在 [B, C, T, H, W] 上操作
        noise = torch.randn_like(clean_latent_BCTHW)
        
        # 🔥 关键修复：对 clean_latent 加噪声（而非 subtitle_latent）
        # - noisy_latent (x): 噪声版本的 clean video（去噪目标）
        # - control_latent (y): subtitle video 作为控制信号
        noisy_latent = self.scheduler.add_noise(clean_latent_BCTHW, noise, timestep)
        control_latent = subtitle_latent_BCTHW  # 控制信号
        
        if global_step == 0:
            print(f"✓ Step 6 - Flow Matching准备: {time.time()-t5:.2f}秒")
        
        # ============ Step 7: DiT预测 (Wan2.1-Control架构) ============
        if global_step == 0:
            t6 = time.time()
        
        # 🔥 Wan2.1-Control 需要准备完整的控制信号
        # 参考 WanVideoUnit_FunControl 的实现（第774-780行）
        
        # 计算需要填充的通道数
        y_dim = self.pipe.dit.in_dim - control_latent.shape[1] - noisy_latent.shape[1]
        
        # 创建 CLIP feature 零向量（B, 257, 1280）
        clip_feature = torch.zeros(
            (B, 257, 1280), 
            dtype=self.pipe.torch_dtype, 
            device=device
        )
        
        # 创建额外的 y 通道零向量
        y_extra = torch.zeros(
            (B, y_dim, T_latent, H_latent, W_latent),
            dtype=self.pipe.torch_dtype,
            device=device
        )
        
        # 拼接：y = [control_latents, y_extra]
        y_full = torch.cat([control_latent, y_extra], dim=1)
        
        # 🔥 DiT forward
        pred_velocity = self.pipe.dit(
            noisy_latent,       # [B, 16, T, H, W] - 去噪目标
            timestep=timestep,
            context=prompt_emb,
            clip_feature=clip_feature,  # [B, 257, 1280] - CLIP零向量
            y=y_full,           # [B, 36, T, H, W] - 完整控制信号
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )  # 返回 [B, C, T, H, W]
        
        if global_step == 0:
            dit_time = time.time()-t6
            print(f"✓ Step 7 - DiT Forward: {dit_time:.2f}秒 ⚡ {'(WITH gradient_checkpointing)' if self.use_gradient_checkpointing else '(NO gradient_checkpointing)'}")
        
        # ============ Step 8: Focal Loss计算 (使用Scheduler方法) ============
        if global_step == 0:
            t7 = time.time()
        
        # 🔥 关键修复：training_target 应该从 clean_latent 计算（而非 subtitle_latent）
        # 因为模型的学习目标是：从 noisy → clean（在 subtitle 的控制下）
        target_velocity = self.scheduler.training_target(clean_latent_BCTHW, noise, timestep)  # [B, C, T, H, W]
        
        base_loss_tensor = F.mse_loss(pred_velocity, target_velocity, reduction='none')  # [B, C, T, H, W]
        err = base_loss_tensor.mean(dim=1)  # [B, T, H, W] - channel维度平均
        
        if aligned_mask is not None:
            subtitle_region = (aligned_mask > 0.5).float()  # [B, T, H, W]
            background_region = 1.0 - subtitle_region
            
            # Focal loss逻辑
            spatial_weight = 1.0 + self.focal_loss_alpha * subtitle_region
            focal_gamma = 1.5
            focal_term = (err.detach() + 1e-6).pow(focal_gamma)
            weight_map = torch.clamp(spatial_weight * focal_term, 0.1, 20.0)
            
            # 加权
            weighted_loss = err * weight_map
            
            # 🔥 移除这里的scheduler_weight应用，改为在total_loss上统一应用
            # 这样base_loss和temporal_loss使用相同的权重，更一致
            
            base_loss = weighted_loss.mean()
            
            # 统计
            subtitle_loss = (err * subtitle_region).sum() / (subtitle_region.sum() + 1e-6)
            background_loss = (err * background_region).sum() / (background_region.sum() + 1e-6)
            weight_min = weight_map.min()
            weight_max = weight_map.max()
            weight_mean = weight_map.mean()
            subtitle_weight_avg = (weight_map * subtitle_region).sum() / (subtitle_region.sum() + 1e-6)
            background_weight_avg = (weight_map * background_region).sum() / (background_region.sum() + 1e-6)
        else:
            # 无mask时，base_loss不应用scheduler权重（改为在total_loss上统一应用）
            base_loss = err.mean()
            
            subtitle_loss = base_loss
            background_loss = base_loss
            weight_min = torch.tensor(1.0, device=device)
            weight_max = torch.tensor(1.0, device=device)
            weight_mean = torch.tensor(1.0, device=device)
            subtitle_weight_avg = torch.tensor(1.0, device=device)
            background_weight_avg = torch.tensor(1.0, device=device)
        
        if global_step == 0:
            print(f"✓ Step 8 - Focal Loss计算: {time.time()-t7:.2f}秒")
        
        # ============ Step 9: Temporal Loss ============
        if global_step == 0:
            t8 = time.time()
        temporal_loss = torch.tensor(0.0, device=device)
        if self.temporal_loss_weight > 0 and T_latent >= 2:
            # 🔥 compute_multi_scale_temporal_loss期望 [B, T, C, H, W]
            # 目前pred_velocity是 [B, C, T, H, W]，需要转换
            pred_v_BTCHW = pred_velocity.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] → [B, T, C, H, W]
            
            temporal_loss = compute_multi_scale_temporal_loss(
                pred_v_BTCHW,
                aligned_mask,
                scales=[1, 4, 8, 16],
                lambda_temporal=self.temporal_loss_weight,
                use_full_frame=True,
                subtitle_weight=1,
                use_charbonnier=True,
                charbonnier_eps=1e-3
            )
        
        if global_step == 0:
            print(f"✓ Step 9 - Temporal Loss计算: {time.time()-t8:.2f}秒")
        
        # ============ Step 10: 统一应用Scheduler权重 ============
        # 🔥 在total_loss上统一应用scheduler_weight，确保base_loss和temporal_loss使用相同的权重
        # scheduler_weight表示"这个timestep的训练重要性"（中间timestep权重高）
        scheduler_weight = self.scheduler.training_weight(timestep)  # [B]
        scheduler_weight = scheduler_weight.to(device=device)  # 🔥 确保在正确的设备上
        scheduler_weight_mean = scheduler_weight.mean()  # 标量，用于加权total_loss
        
        # Total loss（统一应用scheduler权重）
        # 注意：timestep_id_avg已在Step 6中计算（均匀采样时显示遍历进度）
        total_loss = (base_loss + temporal_loss) * scheduler_weight_mean
        
        # 🔥 打印第一个step的总时间
        if global_step == 0:
            total_time = time.time() - step_start
            print("="*80)
            print(f"🎯 第一个Step总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
            print(f"   最慢步骤: DiT Forward ({dit_time:.2f}秒)")
            print(f"   预计每step: ~{total_time:.1f}秒")
            # 🔥 修正：动态计算实际的参数更新步数（考虑gradient accumulation）
            # 注意：这里无法直接访问dataloader长度，所以先不打印预测时间
            # 实际训练会在后续日志中显示真实进度
            print("="*80 + "\n")
        
        return {
            'loss': total_loss,
            'base_loss': base_loss.item() if isinstance(base_loss, torch.Tensor) else base_loss,
            'subtitle_loss': subtitle_loss.item() if isinstance(subtitle_loss, torch.Tensor) else subtitle_loss,
            'background_loss': background_loss.item() if isinstance(background_loss, torch.Tensor) else background_loss,
            'temporal_loss': temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else temporal_loss,
            'temporal_weight': self.temporal_loss_weight.item(),  # 🔥 记录当前学到的权重
            'mask_ratio': (predicted_mask > 0.5).float().mean().item() if predicted_mask is not None else 0.0,
            'weight_min': weight_min.item() if isinstance(weight_min, torch.Tensor) else weight_min,
            'weight_max': weight_max.item() if isinstance(weight_max, torch.Tensor) else weight_max,
            'weight_mean': weight_mean.item() if isinstance(weight_mean, torch.Tensor) else weight_mean,
            'subtitle_weight': subtitle_weight_avg.item() if isinstance(subtitle_weight_avg, torch.Tensor) else subtitle_weight_avg,
            'background_weight': background_weight_avg.item() if isinstance(background_weight_avg, torch.Tensor) else background_weight_avg,
            'timestep_avg': timestep.mean().item(),
            'timestep_id_avg': timestep_id_avg,  # 均匀采样时的timestep ID（-1表示随机采样）
            'scheduler_weight_avg': scheduler_weight_mean.item() if 'scheduler_weight_mean' in locals() else 1.0,
        }
    
    def forward_with_pipeline_loss(self, batch, global_step=0, save_visualization=False):
        """
        使用pipeline的training_loss（原实现，保留作为备用）
        
        ⚠️ 注意：此实现无法自定义focal loss
        """
        clean_frames = batch['video']
        subtitle_frames_pil = batch['control_video']
        prompts = batch['prompt']
        
        subtitle_frames = self.pil_list_to_tensor(subtitle_frames_pil)
        B, T, C, H, W = subtitle_frames.shape
        device = subtitle_frames.device
        
        # 预测mask
        with torch.no_grad():
            predicted_mask, processed_subtitle_frames = predict_and_process_masks(
                self.mask_predictor,
                subtitle_frames,
                dilation_kernel=self.mask_dilation_kernel,
                enable_random_blackout=self.enable_random_blackout,
                device=device,
                save_visualization=save_visualization,
                vis_save_path=self.vis_save_path,
                global_step=global_step
            )
        
        processed_subtitle_pil = self.tensor_to_pil_list(processed_subtitle_frames)
        
        # 使用pipeline进行训练
        inputs_posi = {"prompt": prompts}
        inputs_shared = {
            "input_video": processed_subtitle_pil,
            "height": H,
            "width": W,
            "num_frames": T,
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": 1.0,
            "min_timestep_boundary": 0.0,
        }
        
        inputs_nega = {}
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )
        
        inputs = {**inputs_shared, **inputs_posi}
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        
        base_loss = self.pipe.training_loss(**models, **inputs)
        
        return {
            'loss': base_loss,
            'base_loss': base_loss.item() if isinstance(base_loss, torch.Tensor) else base_loss,
            'subtitle_loss': 0.0,
            'background_loss': 0.0,
            'temporal_loss': 0.0,
            'temporal_weight': self.temporal_loss_weight.item(),  # 🔥 记录当前学到的权重
            'mask_ratio': (predicted_mask > 0.5).float().mean().item() if predicted_mask is not None else 0.0,
        }
    
    def pil_list_to_tensor(self, pil_list, target_format='BCTHW'):
        """
        将PIL Image列表转为tensor
        
        Args:
            pil_list: List[List[PIL.Image]] 或 List[PIL.Image]
            target_format: 'BCTHW' (DiT格式) 或 'BTCHW' (旧格式)
        
        Returns:
            [B, C, T, H, W] if target_format=='BCTHW'
            [B, T, C, H, W] if target_format=='BTCHW'
        """
        if isinstance(pil_list[0], list):
            # Batch mode
            batch = []
            for video in pil_list:
                frames = []
                for img in video:
                    arr = np.array(img).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]
                    frames.append(tensor)
                video_tensor = torch.stack(frames, dim=1)  # [C, T, H, W]
                batch.append(video_tensor)
            result = torch.stack(batch, dim=0)  # [B, C, T, H, W]
        else:
            # Single video mode
            frames = []
            for img in pil_list:
                arr = np.array(img).astype(np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]
                frames.append(tensor)
            result = torch.stack(frames, dim=1).unsqueeze(0)  # [1, C, T, H, W]
        
        # 如果需要旧格式，转换一次
        if target_format == 'BTCHW':
            result = result.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] → [B, T, C, H, W]
        
        return result
    
    def tensor_to_pil_list(self, tensor):
        """将tensor [B, T, 3, H, W]转回PIL Image列表"""
        B, T, C, H, W = tensor.shape
        
        result = []
        for b in range(B):
            video_frames = []
            for t in range(T):
                frame = tensor[b, t].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(frame)
                video_frames.append(pil_img)
            result.append(video_frames)
        
        if B == 1:
            return result[0]
        return result


# ========== Main Training Loop ==========

def main():
    parser = argparse.ArgumentParser()
    
    # 模型路径
    parser.add_argument('--model_base_path', type=str, required=True)
    parser.add_argument('--adapter_checkpoint', type=str, required=True)
    
    # 数据配置
    parser.add_argument('--non_text_json', type=str, required=True)
    parser.add_argument('--clean_dirs', type=str, nargs='+', required=True)
    parser.add_argument('--subtitle_dirs', type=str, nargs='+', required=True)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_frames', type=int, default=81)
    parser.add_argument('--random_seed', type=int, default=42)
    
    # LoRA配置
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_target_modules', type=str, default="q,k,v,o,ffn.0,ffn.2")
    
    # Loss配置
    parser.add_argument('--focal_loss_alpha', type=float, default=5.0)
    parser.add_argument('--temporal_loss_weight', type=float, default=0.1)
    parser.add_argument('--mask_dilation_kernel', type=int, default=3)
    parser.add_argument('--enable_random_blackout', action='store_true', default=True)
    parser.add_argument('--use_custom_loss', action='store_true', default=True,
                        help='使用自定义focal loss实现（推荐）vs pipeline loss')
    
    # 训练配置
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--use_gradient_checkpointing', action='store_true', default=True,
                        help='使用gradient checkpointing节省显存（会慢2倍）。如果显存充足，用--no_gradient_checkpointing禁用')
    parser.add_argument('--no_gradient_checkpointing', dest='use_gradient_checkpointing', action='store_false',
                        help='禁用gradient checkpointing，速度翻倍但显存增加10-15GB')
    parser.add_argument('--use_tiled_vae', action='store_true', default=False,
                        help='使用tiled VAE编码（AMD GPU推荐，慢但稳定）')
    parser.add_argument('--vae_chunk_size', type=int, default=None,
                        help='VAE chunk大小（None=自动检测，AMD建议5）')
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=1, help='基础日志间隔（打印总loss）')
    parser.add_argument('--log_detail_interval', type=int, default=5, help='详细日志间隔（打印loss分解）')
    parser.add_argument('--vis_interval', type=int, default=20, help='Mask可视化保存间隔（步数）')
    parser.add_argument('--use_uniform_timestep_sampling', action='store_true', default=False,
                        help='🔥 使用均匀采样timestep（质数步长，均匀覆盖所有timesteps），而非随机采样')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.random_seed)
    
    # 初始化accelerator（bf16混合精度）
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='bf16',  # 使用bf16混合精度训练
        log_with='tensorboard',
        project_dir=args.output_path,
    )
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 创建可视化目录
    vis_save_path = os.path.join(args.output_path, "mask_visualizations")
    os.makedirs(vis_save_path, exist_ok=True)
    
    if accelerator.is_main_process:
        print("=" * 80)
        print("SEDiT-WAN2.1 LoRA训练 - Mask引导（81帧版本）")
        print("=" * 80)
        print(f"\n配置:")
        print(f"  模型: {args.model_base_path}")
        print(f"  LoRA rank: {args.lora_rank}")
        print(f"  Focal loss alpha: {args.focal_loss_alpha}")
        print(f"  Temporal loss weight: {args.temporal_loss_weight}")
        print(f"  帧数: {args.num_frames}")
        print(f"  样本数: {args.num_samples}")
        print(f"  学习率: {args.learning_rate}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  Mask可视化目录: {vis_save_path}")
        print()
    
    # 创建mask预测器
    if accelerator.is_main_process:
        print("\n加载Mask预测器...")
    
    mask_predictor = create_mask_predictor(args.adapter_checkpoint, accelerator.device)
    mask_predictor = mask_predictor.to(accelerator.device)
    
    # 🔥 确保mask_predictor完全冻结，不参与DDP
    mask_predictor.eval()
    for param in mask_predictor.parameters():
        param.requires_grad = False
    
    # 创建数据集
    if accelerator.is_main_process:
        print("\n加载数据集...")
    
    dataset = VideoDataset(
        non_text_json_path=args.non_text_json,
        clean_base_dirs=args.clean_dirs,
        subtitle_base_dirs=args.subtitle_dirs,
        num_frames=args.num_frames,
        max_samples=args.num_samples,
        random_seed=args.random_seed,
    )
    
    # 自定义collate函数（保持PIL Image格式）
    def collate_fn(batch):
        """
        自定义collate函数，保持PIL Image列表格式
        
        Args:
            batch: List[dict], 每个dict包含 'video', 'control_video', 'prompt'
        
        Returns:
            dict with batched data
        """
        # batch是list of dicts，每个dict: {'video': List[PIL], 'control_video': List[PIL], 'prompt': str}
        return {
            'video': [item['video'] for item in batch],
            'control_video': [item['control_video'] for item in batch],
            'prompt': [item['prompt'] for item in batch],
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,  # 🔥 优化：使用2个workers加速数据加载（从0改为2）
        pin_memory=True,  # 🔥 优化：启用pin_memory加速GPU传输
        collate_fn=collate_fn,
        persistent_workers=True,  # 🔥 优化：保持workers常驻（避免重复创建）
        prefetch_factor=2,  # 🔥 优化：每个worker预加载2个batch
    )
    
    # 创建模型
    if accelerator.is_main_process:
        print("\n创建训练模型...")
    
    model = WanTrainingModuleWithFocalLoss(
        model_base_path=args.model_base_path,
        mask_predictor=mask_predictor,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
        focal_loss_alpha=args.focal_loss_alpha,
        temporal_loss_weight=args.temporal_loss_weight,
        mask_dilation_kernel=args.mask_dilation_kernel,
        enable_random_blackout=args.enable_random_blackout,
        use_gradient_checkpointing=args.use_gradient_checkpointing,  # 🔥 从命令行参数读取
        vis_save_path=vis_save_path,
        use_custom_loss=args.use_custom_loss,
        use_uniform_timestep_sampling=args.use_uniform_timestep_sampling,  # 🔥 循环均匀采样
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    
    # 学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    num_training_steps = args.num_epochs * len(dataloader)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
        eta_min=args.learning_rate * 0.1
    )
    
    # Accelerator准备
    if accelerator.is_main_process:
        print("\n⏳ Accelerator准备中（DDP分发模型到8个GPU，可能需要1-2分钟）...")
        prepare_start = time.time()
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    if accelerator.is_main_process:
        print(f"✓ Accelerator准备完成: {time.time()-prepare_start:.2f}秒")
    
    # 开始训练
    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("🚀 开始训练")
        print("=" * 80)
        print(f"  总样本: {len(dataset)}")
        print(f"  每GPU样本: {len(dataset)//8}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  Effective batch: {args.batch_size * 8 * args.gradient_accumulation_steps}")
        print(f"  Steps per epoch: {len(dataloader)}")
        print("=" * 80)
    
    global_step = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        
        if accelerator.is_main_process and epoch == 0:
            print("\n⏳ 开始迭代DataLoader（第一次会加载81帧视频，可能需要1-2分钟）...")
            first_batch_start = time.time()
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 🔥 第一个batch加载完成的时间统计
            if batch_idx == 0 and epoch == 0 and accelerator.is_main_process:
                print(f"✓ 第一个batch数据加载完成: {time.time()-first_batch_start:.2f}秒")
                print("🚀 开始Forward Pass...\n")
            
            with accelerator.accumulate(model):
                # Forward pass
                # 🎨 定期保存mask可视化（仅主进程）
                save_vis = (global_step % args.vis_interval == 0) and accelerator.is_main_process
                outputs = model(batch, global_step=global_step, save_visualization=save_vis)
                loss = outputs['loss']
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # 日志
            if accelerator.sync_gradients:
                global_step += 1
                
                # 每步打印总Loss（简洁）
                if global_step % args.log_interval == 0 and accelerator.is_main_process:
                    loss_val = outputs['loss'].item() if isinstance(outputs['loss'], torch.Tensor) else outputs['loss']
                    temporal_w = outputs.get('temporal_weight', 0)
                    print(f"[Step {global_step:4d}] Loss: {loss_val:.6f} | λ_temporal: {temporal_w:.6f} 🎓")
                
                # 每5步打印详细分解
                if global_step % args.log_detail_interval == 0 and accelerator.is_main_process:
                    print(f"\n{'='*60}")
                    print(f"[Step {global_step}] 详细Loss分解:")
                    print(f"{'='*60}")
                    loss_val = outputs['loss'].item() if isinstance(outputs['loss'], torch.Tensor) else outputs['loss']
                    print(f"  Total Loss:      {loss_val:.6f}")
                    print(f"  ├─ Base Loss:    {outputs['base_loss']:.6f}")
                    
                    # 显示focal loss分解
                    if outputs.get('subtitle_loss', 0) > 0:
                        print(f"  │  ├─ Subtitle:   {outputs['subtitle_loss']:.6f}")
                        print(f"  │  └─ Background: {outputs['background_loss']:.6f}")
                        ratio = outputs['subtitle_loss'] / (outputs['background_loss'] + 1e-8)
                        print(f"  │     (S/B比例: {ratio:.2f}×)")
                    
                    # 显示focal modulation统计
                    if 'weight_min' in outputs:
                        print(f"  │  🎯 Focal权重:")
                        print(f"  │     Min/Mean/Max: {outputs['weight_min']:.2f} / {outputs['weight_mean']:.2f} / {outputs['weight_max']:.2f}")
                        if 'subtitle_weight' in outputs:
                            print(f"  │     字幕区域平均权重: {outputs['subtitle_weight']:.2f}")
                            print(f"  │     背景区域平均权重: {outputs['background_weight']:.2f}")
                            weight_ratio = outputs['subtitle_weight'] / (outputs['background_weight'] + 1e-8)
                            print(f"  │     权重比例: {weight_ratio:.2f}×")
                    
                    if outputs.get('temporal_loss', 0) > 0:
                        temporal_weighted = outputs['temporal_loss']
                        # 从outputs中获取实际使用的权重
                        actual_weight = outputs.get('temporal_weight', args.temporal_loss_weight)
                        temporal_raw = temporal_weighted / actual_weight if actual_weight > 0 else 0
                        print(f"  └─ Temporal Loss:")
                        print(f"     ├─ 原始值:    {temporal_raw:.6f}")
                        print(f"     ├─ 学习权重:  {actual_weight:.6f} 🎓")
                        print(f"     └─ 加权后:    {temporal_weighted:.6f} (× {actual_weight:.6f})")
                        if outputs['base_loss'] > 0:
                            raw_ratio = temporal_raw / outputs['base_loss']
                            weighted_ratio = temporal_weighted / outputs['base_loss']
                            print(f"     📊 原始Temporal/Base: {raw_ratio:.1f}×")
                            print(f"     📊 加权Temporal/Base: {weighted_ratio:.1f}×")
                    
                    mask_pct = outputs['mask_ratio'] * 100 if outputs['mask_ratio'] < 1 else outputs['mask_ratio']
                    print(f"  Mask覆盖率:      {mask_pct:.2f}%")
                    
                    # 显示scheduler_weight和timestep信息
                    if 'scheduler_weight_avg' in outputs and 'timestep_avg' in outputs:
                        scheduler_w = outputs['scheduler_weight_avg']
                        timestep_avg = outputs['timestep_avg']
                        timestep_id = outputs.get('timestep_id_avg', -1)
                        
                        # 显示timestep信息
                        # 🔥 修复：如果timestep_avg > 1，说明是索引值，需要归一化显示
                        # FlowMatchScheduler的timesteps数组：索引0对应t=1.0（完全噪声），索引999对应t=0.0（完全真实）
                        # 但实际使用时，scheduler.timesteps[timestep_id]返回的可能是索引值或归一化值
                        if timestep_avg > 1.0:
                            # timestep是索引值（0-999），需要归一化
                            # FlowMatch: t = 1.0 - (timestep_id / num_train_timesteps)
                            timestep_normalized = 1.0 - (timestep_avg / 1000.0)
                        else:
                            # timestep已经是归一化值[0, 1]
                            timestep_normalized = timestep_avg
                        
                        # 🔥 计算用于显示的归一化timestep值
                        if timestep_id >= 0:
                            # 均匀采样模式：从timestep_id计算归一化值（0-1范围）
                            # FlowMatchScheduler: timestep_id=0对应t=1.0, timestep_id=999对应t=0.0
                            # 归一化公式: t_normalized = 1.0 - (timestep_id / num_train_timesteps)
                            timestep_normalized_display = 1.0 - (timestep_id / 1000.0)  # 🔥 从ID计算归一化值
                            progress_pct = (timestep_id / 1000.0) * 100
                            print(f"  ⏱️  Timestep:      {timestep_normalized_display:.4f} (ID={timestep_id}/1000, {progress_pct:.1f}%)")
                            print(f"  📊 Scheduler权重: {scheduler_w:.4f} (基于timestep动态计算，均匀采样模式)")
                        else:
                            # 随机采样模式：使用归一化后的timestep值
                            print(f"  ⏱️  Timestep:      {timestep_normalized:.4f} (平均)")
                            print(f"  📊 Scheduler权重: {scheduler_w:.4f} (基于timestep动态计算，随机采样模式)")
                        
                        # 解释权重含义（使用归一化的timestep值）
                        timestep_for_interpretation = timestep_normalized_display if timestep_id >= 0 else timestep_normalized
                        if timestep_for_interpretation < 0.3:
                            print(f"     (Timestep接近0，权重较低，训练简单样本)")
                        elif timestep_for_interpretation > 0.7:
                            print(f"     (Timestep接近1，权重较低，训练简单样本)")
                        else:
                            print(f"     (Timestep在中间，权重较高，训练困难样本)")
                    
                    print(f"{'='*60}\n")
                
                # 保存checkpoint
                if global_step % args.save_interval == 0:
                    if accelerator.is_main_process:
                        # 保存完整训练状态（用于恢复训练）
                        state_dir = os.path.join(args.output_path, f"checkpoint_{global_step}")
                        accelerator.save_state(state_dir)
                        
                        # 🔥 额外保存LoRA权重（用于推理）
                        unwrapped_model = accelerator.unwrap_model(model)
                        lora_state_dict = {
                            name: param.cpu() 
                            for name, param in unwrapped_model.pipe.dit.named_parameters() 
                            if 'lora' in name.lower()
                        }
                        # 🔥 保存可学习的temporal权重
                        checkpoint_dict = {
                            'model_state_dict': lora_state_dict,
                            'temporal_loss_weight': unwrapped_model.temporal_loss_weight.item()
                        }
                        lora_path = os.path.join(args.output_path, f"checkpoint_{global_step}_lora.pt")
                        torch.save(checkpoint_dict, lora_path)
                        
                        print(f"✓ Checkpoint saved:")
                        print(f"  训练状态: {state_dir}")
                        print(f"  LoRA权重: {lora_path}")
                        print(f"  Temporal权重: {unwrapped_model.temporal_loss_weight.item():.6f} 🎓")
    
    # 保存最终模型
    if accelerator.is_main_process:
        # 保存完整训练状态
        final_state_dir = os.path.join(args.output_path, "final_model")
        accelerator.save_state(final_state_dir)
        
        # 🔥 保存LoRA权重（用于推理）
        unwrapped_model = accelerator.unwrap_model(model)
        lora_state_dict = {
            name: param.cpu() 
            for name, param in unwrapped_model.pipe.dit.named_parameters() 
            if 'lora' in name.lower()
        }
        # 🔥 保存可学习的temporal权重
        checkpoint_dict = {
            'model_state_dict': lora_state_dict,
            'temporal_loss_weight': unwrapped_model.temporal_loss_weight.item()
        }
        lora_path = os.path.join(args.output_path, "final_model_lora.pt")
        torch.save(checkpoint_dict, lora_path)
        
        print(f"\n✓ 训练完成！")
        print(f"  训练状态: {final_state_dir}")
        print(f"  LoRA权重（用于推理）: {lora_path}")
        print(f"  LoRA参数数量: {len(lora_state_dict)}")
        print(f"  最终Temporal权重: {unwrapped_model.temporal_loss_weight.item():.6f} 🎓")


if __name__ == '__main__':
    main()
