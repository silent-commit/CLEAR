"""
字幕mask生成工具
用于focal loss中的权重计算
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
import cv2


def generate_subtitle_mask(
    clean_video: torch.Tensor, 
    subtitle_video: torch.Tensor,
    threshold: float = 0.1,
    kernel_size: int = 5,
    min_area: int = 100
) -> torch.Tensor:
    """
    通过像素差分生成字幕区域mask
    
    Args:
        clean_video: 无字幕视频, [T, H, W, C], 值范围[0, 1]
        subtitle_video: 有字幕视频, [T, H, W, C], 值范围[0, 1]
        threshold: 差分阈值
        kernel_size: 形态学操作的kernel大小
        min_area: 最小连通区域面积（过滤噪声）
        
    Returns:
        mask: [T, H, W], 值为0或1的float tensor
    """
    # 计算差分
    diff = torch.abs(subtitle_video - clean_video)
    
    # 对RGB三个通道求平均
    diff_mean = diff.mean(dim=-1)  # [T, H, W]
    
    # 二值化
    mask = (diff_mean > threshold).float()
    
    # 形态学操作（膨胀 + 闭运算）来平滑mask
    mask = morphology_operations(mask, kernel_size)
    
    # 过滤小区域
    mask = filter_small_regions(mask, min_area)
    
    return mask


def morphology_operations(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    对mask进行形态学操作
    
    Args:
        mask: [T, H, W], 值为0或1
        kernel_size: kernel大小
        
    Returns:
        处理后的mask
    """
    # 转换为numpy进行形态学操作（OpenCV）
    mask_np = mask.cpu().numpy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    processed_masks = []
    for frame_mask in mask_np:
        # 膨胀
        dilated = cv2.dilate(frame_mask, kernel, iterations=1)
        # 闭运算（填充小孔）
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        processed_masks.append(closed)
    
    processed_mask = torch.from_numpy(np.stack(processed_masks)).to(mask.device)
    
    return processed_mask


def filter_small_regions(mask: torch.Tensor, min_area: int) -> torch.Tensor:
    """
    过滤掉mask中的小区域
    
    Args:
        mask: [T, H, W]
        min_area: 最小面积
        
    Returns:
        过滤后的mask
    """
    mask_np = mask.cpu().numpy()
    filtered_masks = []
    
    for frame_mask in mask_np:
        # 连通区域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (frame_mask * 255).astype(np.uint8), connectivity=8
        )
        
        # 创建新的mask（只保留大区域）
        new_mask = np.zeros_like(frame_mask)
        for i in range(1, num_labels):  # 0是背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                new_mask[labels == i] = 1.0
        
        filtered_masks.append(new_mask)
    
    filtered_mask = torch.from_numpy(np.stack(filtered_masks)).to(mask.device)
    
    return filtered_mask


def detect_subtitle_region_by_ocr(frame: np.ndarray) -> Optional[torch.Tensor]:
    """
    使用OCR检测字幕区域（可选方法）
    需要安装paddleocr或easyocr
    
    Args:
        frame: numpy array, [H, W, C]
        
    Returns:
        mask: [H, W], 值为0或1
    """
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True)
        
        # OCR检测
        result = ocr.ocr(frame, cls=True)
        
        # 创建mask
        mask = np.zeros(frame.shape[:2], dtype=np.float32)
        
        if result and result[0]:
            for line in result[0]:
                bbox = line[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                bbox = np.array(bbox, dtype=np.int32)
                
                # 填充多边形
                cv2.fillPoly(mask, [bbox], 1.0)
        
        return torch.from_numpy(mask)
    
    except ImportError:
        print("Warning: PaddleOCR not installed. Using pixel difference instead.")
        return None


def create_bottom_region_mask(height: int, width: int, ratio: float = 0.2) -> torch.Tensor:
    """
    创建视频底部区域的mask（字幕通常在底部）
    可用于初始化或加权
    
    Args:
        height: 视频高度
        width: 视频宽度
        ratio: 底部区域占比（0-1）
        
    Returns:
        mask: [H, W]
    """
    mask = torch.zeros(height, width)
    bottom_start = int(height * (1 - ratio))
    mask[bottom_start:, :] = 1.0
    
    return mask


def soft_mask_with_gaussian(mask: torch.Tensor, sigma: float = 5.0) -> torch.Tensor:
    """
    对mask进行高斯模糊，产生soft mask
    
    Args:
        mask: [T, H, W], 硬mask (0或1)
        sigma: 高斯核标准差
        
    Returns:
        soft_mask: [T, H, W], 值在[0, 1]之间
    """
    # 添加channel维度用于高斯模糊
    mask = mask.unsqueeze(1)  # [T, 1, H, W]
    
    # 创建高斯kernel
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 使用PyTorch的高斯模糊
    from torchvision.transforms.functional import gaussian_blur
    soft_mask = gaussian_blur(mask, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    
    soft_mask = soft_mask.squeeze(1)  # [T, H, W]
    
    return soft_mask


def expand_mask(mask: torch.Tensor, expand_ratio: float = 1.2) -> torch.Tensor:
    """
    扩大mask区域（防止字幕边缘遗留）
    
    Args:
        mask: [T, H, W]
        expand_ratio: 扩大比例
        
    Returns:
        扩大后的mask
    """
    # 找到每帧的非零区域边界框
    expanded_masks = []
    
    for frame_mask in mask:
        mask_np = frame_mask.cpu().numpy()
        
        # 找到非零区域
        coords = np.argwhere(mask_np > 0)
        if len(coords) == 0:
            expanded_masks.append(frame_mask)
            continue
        
        # 计算边界框
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # 计算中心和扩大后的尺寸
        cy, cx = (y_min + y_max) / 2, (x_min + x_max) / 2
        h, w = y_max - y_min, x_max - x_min
        
        new_h, new_w = int(h * expand_ratio), int(w * expand_ratio)
        
        # 新的边界框
        new_y_min = max(0, int(cy - new_h / 2))
        new_y_max = min(mask_np.shape[0], int(cy + new_h / 2))
        new_x_min = max(0, int(cx - new_w / 2))
        new_x_max = min(mask_np.shape[1], int(cx + new_w / 2))
        
        # 创建扩大的mask
        new_mask = np.zeros_like(mask_np)
        new_mask[new_y_min:new_y_max, new_x_min:new_x_max] = 1.0
        
        expanded_masks.append(torch.from_numpy(new_mask).to(mask.device))
    
    return torch.stack(expanded_masks)


def temporal_smoothing(mask: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    """
    对mask进行时间维度的平滑（减少抖动）
    
    Args:
        mask: [T, H, W]
        window_size: 平滑窗口大小
        
    Returns:
        平滑后的mask
    """
    T, H, W = mask.shape
    
    # 添加维度用于卷积
    mask = mask.view(1, 1, T, H, W)  # [B, C, T, H, W]
    
    # 创建1D时间卷积kernel
    kernel = torch.ones(1, 1, window_size, 1, 1) / window_size
    kernel = kernel.to(mask.device)
    
    # Padding
    padding = (window_size - 1) // 2
    mask = F.pad(mask, (0, 0, 0, 0, padding, padding), mode='replicate')
    
    # 卷积
    smoothed_mask = F.conv3d(mask, kernel)
    
    smoothed_mask = smoothed_mask.squeeze(0).squeeze(0)  # [T, H, W]
    
    return smoothed_mask

