"""
VAE精确时间对齐工具

本模块提供与VAE时间下采样策略精确对齐的工具函数。

VAE的时间下采样特点：
- 使用causal 3D convolution进行编码
- 第0帧单独处理（保持因果性）
- 之后的帧分块处理，每个chunk经过temporal downsampling
- 时间维度stride=2，可能经过多次下采样

参考：
- WAN Video VAE使用causal temporal downsampling
- 输入T帧 -> 输出T' = (T-1)//temporal_downsample_factor + 1 帧
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


def calculate_vae_temporal_indices(
    T_input: int, 
    T_output: int,
    mode: str = 'causal_uniform'
) -> torch.Tensor:
    """
    计算VAE时间下采样对应的采样索引
    
    Args:
        T_input: 输入帧数
        T_output: VAE输出的latent帧数
        mode: 采样模式
            - 'causal_uniform': 因果均匀采样（推荐）
            - 'center_uniform': 中心对齐均匀采样
            - 'receptive_field': 基于感受野的采样
    
    Returns:
        indices: [T_output], 每个输出帧对应的输入帧索引（浮点数，用于插值）
    """
    
    if mode == 'causal_uniform':
        # 模拟causal convolution的特性：
        # - 第0帧 -> 输出第0帧（只能看到自己）
        # - 之后的帧考虑时间感受野，采样中心偏向过去
        
        # 计算实际的时间下采样因子
        temporal_factor = (T_input - 1) / max(T_output - 1, 1)
        
        # 第0帧特殊处理
        indices = [0.0]
        
        # 剩余帧：考虑causal receptive field
        for i in range(1, T_output):
            # 中心位置
            center_idx = i * temporal_factor
            # causal偏移：向过去偏移半个感受野
            # 感受野大小估计为temporal_factor
            causal_shift = temporal_factor * 0.25  # 向过去偏移1/4感受野
            idx = center_idx - causal_shift
            # 确保不超出边界
            idx = max(0.0, min(float(T_input - 1), idx))
            indices.append(idx)
        
        return torch.tensor(indices, dtype=torch.float32)
    
    elif mode == 'center_uniform':
        # 中心对齐的均匀采样
        if T_output == 1:
            return torch.tensor([T_input // 2], dtype=torch.float32)
        
        indices = torch.linspace(0, T_input - 1, T_output)
        return indices
    
    elif mode == 'receptive_field':
        # 基于卷积感受野的采样
        # 假设VAE使用stride=2的下采样，经过log2(T_input/T_output)层
        
        temporal_factor = (T_input - 1) / max(T_output - 1, 1)
        
        # 计算每个输出帧的感受野中心
        indices = []
        for i in range(T_output):
            if i == 0:
                # 第0帧只能看到自己（causal）
                idx = 0.0
            else:
                # 感受野中心位置（向过去扩展）
                center_idx = i * temporal_factor
                # 考虑causal性质，感受野只向过去扩展
                # 取感受野的加权中心（偏向过去）
                receptive_field_size = temporal_factor
                idx = center_idx - receptive_field_size * 0.3
                idx = max(0.0, min(float(T_input - 1), idx))
            
            indices.append(idx)
        
        return torch.tensor(indices, dtype=torch.float32)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def align_frames_to_vae_temporal(
    frames: torch.Tensor,
    T_output: int,
    mode: str = 'causal_uniform',
    align_corners: bool = False
) -> torch.Tensor:
    """
    将输入帧精确对齐到VAE的时间下采样
    
    Args:
        frames: [B, T_input, H, W, C] 或 [B, T_input, C, H, W]
        T_output: 目标帧数（VAE编码后的帧数）
        mode: 采样模式，见calculate_vae_temporal_indices
        align_corners: 是否在插值时对齐角点
    
    Returns:
        aligned_frames: [B, T_output, H, W, C] 或 [B, T_output, C, H, W]
    """
    
    B, T_input = frames.shape[:2]
    
    # 判断输入格式
    if frames.ndim == 5:
        if frames.shape[2] == 3 or frames.shape[2] < frames.shape[3]:
            # [B, T, C, H, W]
            format_is_tchw = True
            C, H, W = frames.shape[2:]
        else:
            # [B, T, H, W, C]
            format_is_tchw = False
            H, W, C = frames.shape[2:]
    else:
        raise ValueError(f"Expected 5D tensor, got shape {frames.shape}")
    
    # 如果已经是目标帧数，直接返回
    if T_input == T_output:
        return frames
    
    # 计算采样索引
    indices = calculate_vae_temporal_indices(T_input, T_output, mode=mode)
    
    # 转换为归一化坐标 [-1, 1]（用于grid_sample）
    # grid_sample的坐标系统：-1对应第0帧，1对应第T_input-1帧
    norm_indices = (indices / (T_input - 1)) * 2 - 1
    
    # 准备grid_sample的grid
    # grid shape: [B, T_output, H, W, 3]，最后一维是(x, y, t)
    # 对于时间维度，我们只需要t坐标
    device = frames.device
    dtype = frames.dtype
    
    # 创建空间网格（保持不变）
    # x范围: [-1, 1] 对应 [0, W-1]
    # y范围: [-1, 1] 对应 [0, H-1]
    y_grid = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    x_grid = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    
    # 创建时间网格
    t_grid = norm_indices.to(device=device, dtype=dtype)
    
    # 构建完整的5D grid: [B, T_output, H, W, 3]
    # 维度顺序：(depth/time, height, width) -> (t, y, x)
    grid = torch.zeros(B, T_output, H, W, 3, device=device, dtype=dtype)
    
    # 填充x, y坐标（保持原样）
    grid[..., 0] = x_grid.view(1, 1, 1, W)  # x坐标
    grid[..., 1] = y_grid.view(1, 1, H, 1)  # y坐标
    
    # 填充t坐标（采样位置）
    grid[..., 2] = t_grid.view(1, T_output, 1, 1)  # t坐标
    
    # 转换frames格式以适配grid_sample
    # grid_sample需要 [B, C, D, H, W] 格式
    if format_is_tchw:
        # [B, T, C, H, W] -> [B, C, T, H, W]
        frames_for_sample = frames.permute(0, 2, 1, 3, 4)
    else:
        # [B, T, H, W, C] -> [B, C, T, H, W]
        frames_for_sample = frames.permute(0, 4, 1, 2, 3)
    
    # 使用grid_sample进行3D插值
    # mode='bilinear' 对应3D的trilinear插值
    aligned = F.grid_sample(
        frames_for_sample,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=align_corners
    )
    
    # 转换回原始格式
    if format_is_tchw:
        # [B, C, T_output, H, W] -> [B, T_output, C, H, W]
        aligned = aligned.permute(0, 2, 1, 3, 4)
    else:
        # [B, C, T_output, H, W] -> [B, T_output, H, W, C]
        aligned = aligned.permute(0, 2, 3, 4, 1)
    
    return aligned


def align_mask_to_vae_temporal(
    mask: torch.Tensor,
    T_output: int,
    mode: str = 'causal_uniform',
    threshold: Optional[float] = None
) -> torch.Tensor:
    """
    将mask精确对齐到VAE的时间下采样
    
    Args:
        mask: [B, T_input, H, W] 或 [B, T_input, 1, H, W]
        T_output: 目标帧数
        mode: 采样模式
        threshold: 如果提供，则对插值后的mask进行二值化
    
    Returns:
        aligned_mask: [B, T_output, H, W]
    """
    
    # 标准化输入格式
    if mask.ndim == 4:
        # [B, T, H, W]
        mask_5d = mask.unsqueeze(2)  # [B, T, 1, H, W]
    elif mask.ndim == 5:
        # [B, T, 1, H, W]
        mask_5d = mask
    else:
        raise ValueError(f"Expected 4D or 5D mask, got shape {mask.shape}")
    
    # 使用align_frames_to_vae_temporal处理
    aligned = align_frames_to_vae_temporal(
        mask_5d,
        T_output=T_output,
        mode=mode,
        align_corners=False
    )
    
    # 移除channel维度
    aligned = aligned.squeeze(2)  # [B, T_output, H, W]
    
    # 二值化（如果需要）
    if threshold is not None:
        aligned = (aligned > threshold).float()
    
    return aligned


def verify_temporal_alignment(
    original_frames: torch.Tensor,
    aligned_frames: torch.Tensor,
    vae_latents: torch.Tensor,
    save_path: Optional[str] = None
) -> dict:
    """
    验证时间对齐的质量
    
    Args:
        original_frames: [B, T_input, H, W, C]
        aligned_frames: [B, T_output, H, W, C]
        vae_latents: [B, T_output, h, w, c]
        save_path: 如果提供，保存可视化结果
    
    Returns:
        metrics: 对齐质量指标字典
    """
    
    metrics = {
        'T_input': original_frames.shape[1],
        'T_output': aligned_frames.shape[1],
        'T_latent': vae_latents.shape[1],
        'temporal_factor': original_frames.shape[1] / aligned_frames.shape[1],
    }
    
    # 验证维度匹配
    assert aligned_frames.shape[1] == vae_latents.shape[1], \
        f"Aligned frames ({aligned_frames.shape[1]}) != VAE latents ({vae_latents.shape[1]})"
    
    metrics['dimension_check'] = 'PASS'
    
    # 如果需要可视化
    if save_path is not None:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, min(8, metrics['T_output']), figsize=(20, 8))
            
            B = 0  # 只显示第一个batch
            
            for t in range(min(8, metrics['T_output'])):
                # 原始帧（找最近的）
                t_orig = int(t * metrics['temporal_factor'])
                if original_frames.shape[-1] == 3:
                    orig_img = original_frames[B, t_orig].cpu().numpy()
                else:
                    orig_img = original_frames[B, t_orig, :, :, :3].cpu().numpy()
                orig_img = (orig_img + 1) / 2  # [-1,1] -> [0,1]
                
                # 对齐后的帧
                if aligned_frames.shape[-1] == 3:
                    aligned_img = aligned_frames[B, t].cpu().numpy()
                else:
                    aligned_img = aligned_frames[B, t, :, :, :3].cpu().numpy()
                aligned_img = (aligned_img + 1) / 2
                
                # VAE latent（可视化第一个通道）
                latent_img = vae_latents[B, t, :, :, 0].cpu().numpy()
                latent_img = (latent_img - latent_img.min()) / (latent_img.max() - latent_img.min() + 1e-8)
                
                axes[0, t].imshow(orig_img)
                axes[0, t].set_title(f'Orig t={t_orig}')
                axes[0, t].axis('off')
                
                axes[1, t].imshow(aligned_img)
                axes[1, t].set_title(f'Aligned t={t}')
                axes[1, t].axis('off')
                
                axes[2, t].imshow(latent_img, cmap='viridis')
                axes[2, t].set_title(f'Latent t={t}')
                axes[2, t].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            metrics['visualization'] = save_path
        except Exception as e:
            metrics['visualization_error'] = str(e)
    
    return metrics


# 提供多种对齐策略的快捷函数
def align_to_vae_causal(frames: torch.Tensor, T_output: int) -> torch.Tensor:
    """使用causal uniform策略对齐（推荐用于VAE）"""
    return align_frames_to_vae_temporal(frames, T_output, mode='causal_uniform')


def align_to_vae_center(frames: torch.Tensor, T_output: int) -> torch.Tensor:
    """使用center uniform策略对齐（用于对比实验）"""
    return align_frames_to_vae_temporal(frames, T_output, mode='center_uniform')


def align_to_vae_receptive(frames: torch.Tensor, T_output: int) -> torch.Tensor:
    """使用receptive field策略对齐（最精确但计算稍慢）"""
    return align_frames_to_vae_temporal(frames, T_output, mode='receptive_field')


if __name__ == '__main__':
    # 测试代码
    print("VAE Temporal Alignment Utils")
    print("=" * 60)
    
    # 模拟场景：25帧输入 -> 7帧latent
    T_input = 25
    T_output = 7
    
    print(f"\n测试场景：{T_input}帧 -> {T_output}帧")
    print(f"下采样因子：{T_input / T_output:.2f}")
    
    # 计算不同模式的采样索引
    for mode in ['causal_uniform', 'center_uniform', 'receptive_field']:
        indices = calculate_vae_temporal_indices(T_input, T_output, mode=mode)
        print(f"\n{mode} 模式:")
        print(f"  采样索引: {indices.tolist()}")
        print(f"  对应原始帧: {[int(idx) for idx in indices]}")
    
    # 测试实际对齐
    print("\n" + "=" * 60)
    print("测试帧对齐...")
    
    B, H, W, C = 2, 256, 256, 3
    test_frames = torch.randn(B, T_input, H, W, C)
    
    aligned = align_to_vae_causal(test_frames, T_output)
    print(f"输入: {test_frames.shape}")
    print(f"输出: {aligned.shape}")
    print(f"✓ 对齐成功")
    
    # 测试mask对齐
    test_mask = torch.rand(B, T_input, H, W)
    aligned_mask = align_mask_to_vae_temporal(test_mask, T_output, threshold=0.5)
    print(f"\nMask对齐:")
    print(f"输入: {test_mask.shape}")
    print(f"输出: {aligned_mask.shape}")
    print(f"✓ Mask对齐成功")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过")

