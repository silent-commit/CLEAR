"""
CLEAR Stage I: Self-Supervised Prior Learning

Extracts coarse occlusion guidance from paired videos using pixel differences
as weak supervision, eliminating annotation dependency through orthogonal
feature constraints and adversarial purification.

Training objectives:
1. Disentangled feature learning via dual encoders (E_sub, E_content)
2. Orthogonality constraint for feature independence
3. Binary mask prediction from subtitle features
4. Content reconstruction verification

Reference: Section 3.2 in the CLEAR paper
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.disentangled_modules import (
    DisentangledSubtitleAdapter,
    compute_mask_loss,
    compute_disentangle_loss,
    compute_reconstruction_loss,
    compute_mask_metrics
)
from models.dual_encoder import MultiscaleDisentangledAdapter
from utils.contrastive_loss import DifferenceContrastiveLossWithStats
from torch.cuda.amp import autocast, GradScaler

# 导入视频加载工具
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils import load_video, match_video_pairs
import cv2
import json


class FilteredException(Exception):
    """样本被过滤的异常"""
    pass


class VideoGroupedBatchSampler(torch.utils.data.Sampler):
    """
    视频分组的Batch采样器（支持Subset）
    
    确保每个batch内的所有样本都来自同一个视频，从而保证尺寸一致。
    
    工作原理：
    1. 如果是Subset，先获取实际的原始索引
    2. 将原始索引按视频分组
    3. 每个batch从同一个视频组中采样
    4. 支持分布式训练
    """
    
    def __init__(self, dataset, frames_per_video, batch_size, 
                 rank=0, world_size=1, shuffle=True, seed=0):
        """
        Args:
            dataset: 数据集或Subset
            frames_per_video: 每个视频的帧数
            batch_size: 每个batch的大小
            rank: 当前进程的rank
            world_size: 总进程数
            shuffle: 是否打乱视频顺序
            seed: 随机种子
        """
        self.dataset = dataset
        self.frames_per_video = frames_per_video
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # 获取实际的索引列表
        if hasattr(dataset, 'indices'):
            # 这是一个Subset
            self.indices = dataset.indices
        else:
            # 这是原始数据集
            self.indices = list(range(len(dataset)))
        
        # 按视频分组索引
        self._group_by_video()
        
    def _group_by_video(self):
        """将索引按视频分组"""
        # 将索引按照它们所属的视频分组
        video_groups = {}  # {video_id: [idx1, idx2, ...]}
        
        for idx in self.indices:
            # 计算这个索引属于哪个视频
            video_id = idx // self.frames_per_video
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(idx)
        
        # 过滤：只保留有足够帧的视频（至少有1帧）
        self.video_groups = {
            vid: sorted(indices) for vid, indices in video_groups.items() 
            if len(indices) > 0
        }
        
        self.video_ids = list(self.video_groups.keys())
        print(f"[Rank {self.rank}] VideoGroupedBatchSampler: "
              f"{len(self.video_ids)} videos, "
              f"{len(self.indices)} frames")
        
    def __iter__(self):
        # 获取视频ID列表
        video_ids = self.video_ids.copy()
        
        # 打乱视频顺序（如果需要）
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(video_ids), generator=g).tolist()
            video_ids = [video_ids[i] for i in perm]
        
        # 为每个视频生成batch
        all_batches = []
        for video_id in video_ids:
            video_frame_indices = self.video_groups[video_id]
            
            # 将这个视频的帧分成多个batch
            for i in range(0, len(video_frame_indices), self.batch_size):
                batch = video_frame_indices[i:i + self.batch_size]
                all_batches.append(batch)
        
        # 分布式：每个rank只处理一部分batch
        # 使用round-robin方式分配batch
        rank_batches = [all_batches[i] for i in range(len(all_batches)) 
                       if i % self.world_size == self.rank]
        
        # 返回这个rank的所有batch的索引
        for batch in rank_batches:
            yield batch
    
    def __len__(self):
        # 计算这个rank的batch数量
        total_batches = sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.video_groups.values()
        )
        return (total_batches + self.world_size - 1) // self.world_size
    
    def set_epoch(self, epoch):
        """设置epoch（用于分布式训练的随机性）"""
        self.epoch = epoch


class VideoBatchDataset(torch.utils.data.Dataset):
    """
    视频级别批次数据集（每个样本就是一个视频的多帧batch）
    
    核心思想：
    - 每个样本返回一个视频的N帧（构成一个batch）
    - 在数据集内部完成过滤（过滤掉无字幕帧）
    - 同一视频的所有帧尺寸一致（根据视频方向确定）
    - 避免跨视频的batch混合
    
    优势：
    1. 彻底解决batch尺寸不匹配问题
    2. 更高效（不需要预先检测所有视频）
    3. 过滤逻辑清晰（在视频级别）
    4. 训练更稳定（同一batch来自同一场景）
    """
    
    def __init__(self, clean_dirs, subtitle_dirs, sample_filter_file=None,
                 target_height=None, target_width=None, augmentation_config=None,
                 frames_per_video=10, max_video_pairs=None, random_seed=42,
                 diff_threshold=0.1, min_diff_threshold=0.003, 
                 save_samples=False, save_dir=None, additional_sources=None):
        """
        Args:
            frames_per_video: 每个视频采样多少帧（一个batch的大小）
            additional_sources: 额外数据源列表，每个数据源包含clean_dir和subtitle_dir
            其他参数同之前
        """
        self.clean_dirs = clean_dirs if isinstance(clean_dirs, list) else [clean_dirs]
        self.subtitle_dirs = subtitle_dirs if isinstance(subtitle_dirs, list) else [subtitle_dirs]
        self.target_height = target_height
        self.target_width = target_width
        self.frames_per_video = frames_per_video
        self.max_video_pairs = max_video_pairs
        self.random_seed = random_seed
        self.diff_threshold = diff_threshold
        self.min_diff_threshold = min_diff_threshold
        self.save_samples = save_samples
        self.save_dir = save_dir
        self.additional_sources = additional_sources or []
        
        # 统计信息
        self.filter_stats = {}
        self.total_filtered = 0
        self.total_kept = 0
        
        # 创建保存目录
        if self.save_samples and self.save_dir:
            self.filtered_dir = os.path.join(self.save_dir, 'filtered_frames')
            os.makedirs(self.filtered_dir, exist_ok=True)
            print(f"样本保存目录:")
            print(f"  过滤帧: {self.filtered_dir}")
        
        # 收集视频对
        self.video_pairs = []
        self._collect_video_pairs(sample_filter_file)
        
        # 数据增强
        self.augmentation_config = augmentation_config or {}
        self.setup_transforms()
    
    def _collect_video_pairs(self, sample_filter_file):
        """收集视频对（支持多数据源）"""
        # ========== 数据源1：非绿幕视频（使用过滤文件） ==========
        if sample_filter_file and os.path.exists(sample_filter_file):
            if sample_filter_file.endswith('.json'):
                with open(sample_filter_file, 'r') as f:
                    items = json.load(f)
                def _to_video_name(s: str) -> str:
                    s = s.rsplit('.', 1)[0]
                    parts = s.split('_')
                    if len(parts) > 1 and parts[-1].isdigit():
                        return '_'.join(parts[:-1])
                    return s
                allowed_samples = set(_to_video_name(s) for s in items)
            else:
                with open(sample_filter_file, 'r') as f:
                    allowed_samples = set(line.strip() for line in f if line.strip())
        else:
            allowed_samples = None
        
        # 使用match_video_pairs工具函数
        all_pairs = match_video_pairs(self.clean_dirs, self.subtitle_dirs)
        
        # 过滤样本
        if allowed_samples:
            def _allowed(name: str) -> bool:
                if name in allowed_samples:
                    return True
                for a in allowed_samples:
                    if a and (a in name or name in a):
                        return True
                return False
            self.video_pairs = [
                (pair['clean'], pair['subtitle'])
                for pair in all_pairs
                if _allowed(pair['name'])
            ]
        else:
            self.video_pairs = [(pair['clean'], pair['subtitle']) for pair in all_pairs]

        base_pairs_count = len(self.video_pairs)
        print(f"✓ 数据源1 (non_text.json过滤): {base_pairs_count} 视频对")
        
        # ========== 数据源2+：额外数据源（如szr4绿幕视频） ==========
        additional_pairs_count = 0
        for source in self.additional_sources:
            if not source.get('enabled', True):
                continue
                
            source_name = source.get('name', 'unknown')
            clean_dir = source.get('clean_dir')
            subtitle_dir = source.get('subtitle_dir')
            source_max_samples = source.get('max_samples', None)
            
            if not clean_dir or not subtitle_dir:
                print(f"⚠ 跳过数据源 {source_name}: 缺少clean_dir或subtitle_dir")
                continue
            
            # 加载该数据源的所有视频对
            source_pairs = match_video_pairs([clean_dir], [subtitle_dir])
            source_pairs_list = [(pair['clean'], pair['subtitle']) for pair in source_pairs]
            
            # 如果设置了max_samples，随机抽样
            if source_max_samples and len(source_pairs_list) > source_max_samples:
                rng = np.random.RandomState(self.random_seed)
                idxs = rng.choice(len(source_pairs_list), size=source_max_samples, replace=False)
                source_pairs_list = [source_pairs_list[i] for i in idxs]
                print(f"✓ 数据源 [{source_name}]: {len(source_pairs_list)} 视频对 (随机抽样，原始: {len(source_pairs)})")
            else:
                print(f"✓ 数据源 [{source_name}]: {len(source_pairs_list)} 视频对")
            
            # 添加到总列表
            self.video_pairs.extend(source_pairs_list)
            additional_pairs_count += len(source_pairs_list)
        
        # 限制最大数量（如果指定）
        if self.max_video_pairs and len(self.video_pairs) > self.max_video_pairs:
            rng = np.random.RandomState(self.random_seed)
            idxs = rng.choice(len(self.video_pairs), size=self.max_video_pairs, replace=False)
            self.video_pairs = [self.video_pairs[i] for i in idxs]
        
        print(f"=" * 60)
        print(f"✓ 总计: {len(self.video_pairs)} 视频对")
        print(f"  - non_text.json: {base_pairs_count} 对")
        print(f"  - 额外数据源: {additional_pairs_count} 对")
        print(f"  - 每个视频采样: {self.frames_per_video} 帧")
        print(f"  - 预估总帧数: ~{len(self.video_pairs) * self.frames_per_video} 帧")
        print(f"=" * 60)
    
    def setup_transforms(self):
        """设置数据增强"""
        aug_config = self.augmentation_config
        
        transforms_list = []
        
        # 轻微颜色抖动（可选）
        if aug_config.get('color_jitter', False):
            transforms_list.append(
                transforms.ColorJitter(
                    brightness=aug_config.get('brightness', 0.05),
                    contrast=aug_config.get('contrast', 0.05),
                    saturation=aug_config.get('saturation', 0.05),
                    hue=aug_config.get('hue', 0.02)
                )
            )
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.transform = transforms.Compose(transforms_list)
    
    def __len__(self):
        return len(self.video_pairs)
    
    def __getitem__(self, idx):
        """
        返回一个视频的多帧batch
        
        Returns:
            dict: {
                'subtitle': Tensor [N, 3, H, W] - N帧带字幕图像
                'clean': Tensor [N, 3, H, W] - N帧干净图像
                'mask': Tensor [N, 1, H, W] - N个字幕mask
            }
            其中N <= frames_per_video（因为会过滤掉无字幕帧）
        """
        clean_video_path, subtitle_video_path = self.video_pairs[idx]
        video_name = os.path.basename(clean_video_path).split('.')[0]
        
        try:
            # 加载视频
            clean_video = load_video(clean_video_path)
            subtitle_video = load_video(subtitle_video_path)
            
            num_frames = min(len(clean_video), len(subtitle_video))
            if num_frames == 0:
                raise ValueError(f"Empty video: {clean_video_path}")
            
            # 均匀采样帧索引
            if num_frames <= self.frames_per_video:
                frame_indices = list(range(num_frames))
            else:
                # 均匀采样
                step = num_frames / self.frames_per_video
                frame_indices = [int(i * step) for i in range(self.frames_per_video)]
            
            # ⭐ 根据第一帧确定视频方向和目标尺寸
            first_frame = (clean_video[frame_indices[0]].detach().cpu().numpy() * 255).astype(np.uint8)
            from PIL import Image
            first_img = Image.fromarray(first_frame)
            orig_w, orig_h = first_img.size
            
            is_portrait = orig_h > orig_w
            if is_portrait:
                tgt_h = self.target_height or 1920
                tgt_w = self.target_width or 1080
                if tgt_w > tgt_h:
                    tgt_h, tgt_w = tgt_w, tgt_h
            else:
                tgt_h = self.target_height or 720
                tgt_w = self.target_width or 1280
                if tgt_h > tgt_w:
                    tgt_h, tgt_w = tgt_w, tgt_h
            
            def _resize_crop(img, th, tw):
                w, h = img.size
                scale = max(th / h, tw / w)
                nw = int(round(w * scale))
                nh = int(round(h * scale))
                img = img.resize((nw, nh), Image.BILINEAR)
                l = (nw - tw) // 2
                t = (nh - th) // 2
                return img.crop((l, t, l + tw, t + th))
            
            # 初始化统计
            if video_name not in self.filter_stats:
                self.filter_stats[video_name] = {'total': 0, 'filtered': 0, 'kept': 0}
            
            # 处理所有采样的帧
            subtitle_list = []
            clean_list = []
            mask_list = []
            
            for frame_idx in frame_indices:
                self.filter_stats[video_name]['total'] += 1
                
                # 获取帧
                clean_frame = (clean_video[frame_idx].detach().cpu().numpy() * 255).astype(np.uint8)
                subtitle_frame = (subtitle_video[frame_idx].detach().cpu().numpy() * 255).astype(np.uint8)
                
                clean_img = Image.fromarray(clean_frame)
                subtitle_img = Image.fromarray(subtitle_frame)
                
                # Resize到统一尺寸
                clean_img = _resize_crop(clean_img, tgt_h, tgt_w)
                subtitle_img = _resize_crop(subtitle_img, tgt_h, tgt_w)
                
                # 应用相同的随机种子
                seed = np.random.randint(2147483647)
                
                torch.manual_seed(seed)
                np.random.seed(seed)
                clean_tensor = self.transform(clean_img)
                
                torch.manual_seed(seed)
                np.random.seed(seed)
                subtitle_tensor = self.transform(subtitle_img)
                
                # 计算差异和mask
                diff_tensor = torch.abs(subtitle_tensor - clean_tensor)
                diff_magnitude = diff_tensor.mean(dim=0, keepdim=True)
                
                # ⭐ 过滤检查
                high_diff_pixels = (diff_magnitude > 0.1).float()
                high_diff_ratio = high_diff_pixels.mean().item()
                
                if high_diff_ratio < self.min_diff_threshold:
                    # 过滤掉这一帧
                    self.filter_stats[video_name]['filtered'] += 1
                    self.total_filtered += 1
                    
                    # 保存被过滤的帧
                    if self.save_samples and self.save_dir:
                        self._save_filtered_frame(
                            subtitle_img, clean_img, diff_magnitude,
                            video_name, frame_idx, high_diff_ratio
                        )
                    continue
                
                # 保留这一帧
                self.filter_stats[video_name]['kept'] += 1
                self.total_kept += 1
                
                # 生成mask
                diff_mean = diff_magnitude.mean()
                diff_std = diff_magnitude.std()
                adaptive_threshold = max((diff_mean + self.diff_threshold * diff_std).item(), 0.01)
                mask = (diff_magnitude > adaptive_threshold).float()
                
                subtitle_list.append(subtitle_tensor)
                clean_list.append(clean_tensor)
                mask_list.append(mask)
            
            # 如果所有帧都被过滤，返回至少一个dummy样本避免训练中断
            if len(subtitle_list) == 0:
                print(f"⚠️ 视频 {video_name} 的所有帧都被过滤，返回dummy样本")
                subtitle_list = [torch.zeros(3, tgt_h, tgt_w)]
                clean_list = [torch.zeros(3, tgt_h, tgt_w)]
                mask_list = [torch.zeros(1, tgt_h, tgt_w)]
            
            # 堆叠成batch
            return {
                'subtitle': torch.stack(subtitle_list, dim=0),  # [N, 3, H, W]
                'clean': torch.stack(clean_list, dim=0),         # [N, 3, H, W]
                'mask': torch.stack(mask_list, dim=0),           # [N, 1, H, W]
            }
            
        except Exception as e:
            print(f"Error loading video pair {video_name}: {e}")
            # 返回dummy样本
            tgt_h, tgt_w = self.target_height or 720, self.target_width or 1280
            return {
                'subtitle': torch.zeros(1, 3, tgt_h, tgt_w),
                'clean': torch.zeros(1, 3, tgt_h, tgt_w),
                'mask': torch.zeros(1, 1, tgt_h, tgt_w),
            }
    
    def _save_filtered_frame(self, subtitle_img, clean_img, diff_magnitude, video_name, frame_idx, high_diff_ratio):
        """保存被过滤的帧"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(np.array(subtitle_img))
            axes[0].set_title('Subtitle Frame', fontsize=12)
            axes[0].axis('off')
            
            axes[1].imshow(np.array(clean_img))
            axes[1].set_title('Clean Frame', fontsize=12)
            axes[1].axis('off')
            
            diff_np = diff_magnitude.cpu().numpy().squeeze()
            im = axes[2].imshow(diff_np, cmap='hot')
            axes[2].set_title(f'Difference (ratio={high_diff_ratio:.3%})', fontsize=12, color='red')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            
            fig.suptitle(f'⚠️ FILTERED: {video_name}_frame{frame_idx}\nReason: high_diff_ratio ({high_diff_ratio:.3%}) < threshold ({self.min_diff_threshold:.3%})', 
                        fontsize=14, fontweight='bold', color='red')
            
            plt.tight_layout()
            save_path = os.path.join(self.filtered_dir, f'{video_name}_frame{frame_idx}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to save filtered frame: {e}")
    
    def print_filter_stats(self):
        """打印过滤统计"""
        print("\n" + "=" * 80)
        print("📊 数据过滤统计（视频级别）")
        print("=" * 80)
        
        total_processed = self.total_filtered + self.total_kept
        if total_processed > 0:
            filter_rate = self.total_filtered / total_processed * 100
            keep_rate = self.total_kept / total_processed * 100
            
            print(f"\n总体统计:")
            print(f"  总处理帧数: {total_processed}")
            print(f"  ✓ 保留: {self.total_kept} ({keep_rate:.1f}%)")
            print(f"  ✗ 过滤: {self.total_filtered} ({filter_rate:.1f}%)")
        
        print(f"\n每个视频的过滤情况:")
        print(f"{'视频名称':<30} {'总帧数':>8} {'保留':>8} {'过滤':>8} {'过滤率':>10}")
        print("-" * 80)
        
        for video_name in sorted(self.filter_stats.keys()):
            stats = self.filter_stats[video_name]
            total = stats['total']
            kept = stats['kept']
            filtered = stats['filtered']
            
            if total > 0:
                filter_rate = filtered / total * 100
                print(f"{video_name:<30} {total:>8} {kept:>8} {filtered:>8} {filter_rate:>9.1f}%")
        
        print("=" * 80)


class DifferenceContrastiveDataset(torch.utils.data.Dataset):
    """
    差异对比学习数据集（原始逐帧版本，保留用于兼容性）
    
    核心思想：
    - 输入：(带字幕帧, 干净帧) 对
    - 计算差异：diff = |subtitle - clean|
    - 生成伪标签：字幕位置mask（通过阈值）
    - 对比学习：字幕区域特征 vs 干净区域特征
    
    返回：
    - subtitle: 带字幕帧
    - clean: 干净帧
    - diff: 差异图（可选，可在模型内计算）
    - mask: 字幕位置mask（伪标签）
    """
    
    def __init__(self, clean_dirs, subtitle_dirs, sample_filter_file=None,
                 target_height=None, target_width=None, augmentation_config=None,
                 frames_per_video=10, max_video_pairs=None, random_seed=42,
                 diff_threshold=0.1, min_diff_threshold=0.003, 
                 save_samples=False, save_dir=None, additional_sources=None):
        """
        Args:
            clean_dirs: 干净视频目录列表
            subtitle_dirs: 带字幕视频目录列表
            sample_filter_file: 样本过滤文件
            target_height: 目标高度
            target_width: 目标宽度
            augmentation_config: 数据增强配置
            frames_per_video: 每个视频采样多少帧
            max_video_pairs: 最大视频对数量
            random_seed: 随机种子
            diff_threshold: 差异阈值（用于生成字幕mask）
            min_diff_threshold: 最小差异阈值（过滤无字幕帧）
            save_samples: 是否保存样本用于检查
            save_dir: 样本保存目录
            additional_sources: 额外数据源列表
        """
        self.clean_dirs = clean_dirs if isinstance(clean_dirs, list) else [clean_dirs]
        self.subtitle_dirs = subtitle_dirs if isinstance(subtitle_dirs, list) else [subtitle_dirs]
        self.target_height = target_height
        self.target_width = target_width
        self.frames_per_video = frames_per_video
        self.max_video_pairs = max_video_pairs
        self.random_seed = random_seed
        self.diff_threshold = diff_threshold
        self.min_diff_threshold = min_diff_threshold
        self.save_samples = save_samples
        self.save_dir = save_dir
        self.additional_sources = additional_sources or []
        
        # 统计信息
        self.filter_stats = {}  # {video_name: {'total': 10, 'filtered': 3, 'kept': 7}}
        self.total_filtered = 0
        self.total_kept = 0
        self.saved_kept_count = 0  # 已保存的kept样本数量
        
        # 保存样本的随机采样（避免保存太多）
        # 目标：每个epoch保存100-200个kept样本
        # 如果总样本25000，保留率90%，则kept=22500
        # 抽样率10%可以保存约2250个 → 改为5%保存约1125个 → 改为2%保存约450个
        self.save_sample_prob = 0.1  # 10%的kept样本会被保存（前期）
        self.max_kept_samples = 200  # 最多保存200个kept样本
        
        # 创建保存目录（只保存filtered样本）
        if self.save_samples and self.save_dir:
            self.filtered_dir = os.path.join(self.save_dir, 'filtered_frames')
            # ⭐ 不再保存kept样本，节省空间
            # self.kept_dir = os.path.join(self.save_dir, 'kept_frames')
            os.makedirs(self.filtered_dir, exist_ok=True)
            # os.makedirs(self.kept_dir, exist_ok=True)
            print(f"样本保存目录:")
            print(f"  过滤帧: {self.filtered_dir} (全部保存)")
            print(f"  保留帧: 不保存 (节省空间)")
        
        # 收集所有视频对
        self.video_pairs = []
        self._collect_video_pairs(sample_filter_file)
        
        # 数据增强
        self.augmentation_config = augmentation_config or {}
        self.setup_transforms()
        
    def _collect_video_pairs(self, sample_filter_file):
        """收集视频对并预先确定每个视频的目标尺寸（支持多数据源）"""
        # ========== 数据源1：非绿幕视频（使用过滤文件） ==========
        if sample_filter_file and os.path.exists(sample_filter_file):
            if sample_filter_file.endswith('.json'):
                with open(sample_filter_file, 'r') as f:
                    items = json.load(f)
                def _to_video_name(s: str) -> str:
                    s = s.rsplit('.', 1)[0]
                    parts = s.split('_')
                    if len(parts) > 1 and parts[-1].isdigit():
                        return '_'.join(parts[:-1])
                    return s
                allowed_samples = set(_to_video_name(s) for s in items)
            else:
                with open(sample_filter_file, 'r') as f:
                    allowed_samples = set(line.strip() for line in f if line.strip())
        else:
            allowed_samples = None
        
        # 使用match_video_pairs工具函数
        all_pairs = match_video_pairs(self.clean_dirs, self.subtitle_dirs)
        
        # 过滤样本
        if allowed_samples:
            def _allowed(name: str) -> bool:
                if name in allowed_samples:
                    return True
                for a in allowed_samples:
                    if a and (a in name or name in a):
                        return True
                return False
            self.video_pairs = [
                (pair['clean'], pair['subtitle'])
                for pair in all_pairs
                if _allowed(pair['name'])
            ]
        else:
            self.video_pairs = [(pair['clean'], pair['subtitle']) for pair in all_pairs]

        base_pairs_count = len(self.video_pairs)
        print(f"✓ 数据源1 (non_text.json过滤): {base_pairs_count} 视频对")
        
        # ========== 数据源2+：额外数据源（如szr4绿幕视频） ==========
        additional_pairs_count = 0
        for source in self.additional_sources:
            if not source.get('enabled', True):
                continue
                
            source_name = source.get('name', 'unknown')
            clean_dir = source.get('clean_dir')
            subtitle_dir = source.get('subtitle_dir')
            source_max_samples = source.get('max_samples', None)
            
            if not clean_dir or not subtitle_dir:
                print(f"⚠ 跳过数据源 {source_name}: 缺少clean_dir或subtitle_dir")
                continue
            
            # 加载该数据源的所有视频对
            source_pairs = match_video_pairs([clean_dir], [subtitle_dir])
            source_pairs_list = [(pair['clean'], pair['subtitle']) for pair in source_pairs]
            
            # 如果设置了max_samples，随机抽样
            if source_max_samples and len(source_pairs_list) > source_max_samples:
                rng = np.random.RandomState(self.random_seed)
                idxs = rng.choice(len(source_pairs_list), size=source_max_samples, replace=False)
                source_pairs_list = [source_pairs_list[i] for i in idxs]
                print(f"✓ 数据源 [{source_name}]: {len(source_pairs_list)} 视频对 (随机抽样，原始: {len(source_pairs)})")
            else:
                print(f"✓ 数据源 [{source_name}]: {len(source_pairs_list)} 视频对")
            
            # 添加到总列表
            self.video_pairs.extend(source_pairs_list)
            additional_pairs_count += len(source_pairs_list)
        
        # 限制最大数量（如果指定）
        if self.max_video_pairs and len(self.video_pairs) > self.max_video_pairs:
            rng = np.random.RandomState(self.random_seed)
            idxs = rng.choice(len(self.video_pairs), size=self.max_video_pairs, replace=False)
            self.video_pairs = [self.video_pairs[i] for i in idxs]
        
        print(f"=" * 60)
        print(f"✓ 总计: {len(self.video_pairs)} 视频对")
        print(f"  - non_text.json: {base_pairs_count} 对")
        print(f"  - 额外数据源: {additional_pairs_count} 对")
        print(f"=" * 60)
        
        # ⭐ 关键修复：预先为每个视频确定目标尺寸（视频级别，而不是帧级别）
        # 这样同一视频的所有帧都有相同的目标尺寸
        self.video_target_sizes = {}  # {video_idx: (target_h, target_w)}
        
        print("正在检测视频方向并确定目标尺寸...")
        for video_idx, (clean_path, _) in enumerate(self.video_pairs):
            try:
                # 读取视频第一帧来确定方向
                video = load_video(clean_path)
                if len(video) > 0:
                    first_frame = (video[0].detach().cpu().numpy() * 255).astype(np.uint8)
                    from PIL import Image
                    first_img = Image.fromarray(first_frame)
                    orig_w, orig_h = first_img.size
                    
                    # 判断方向并设置目标尺寸
                    is_portrait = orig_h > orig_w
                    
                    if is_portrait:
                        # 竖屏
                        tgt_h = self.target_height or 1920
                        tgt_w = self.target_width or 1080
                        if tgt_w > tgt_h:
                            tgt_h, tgt_w = tgt_w, tgt_h
                    else:
                        # 横屏
                        tgt_h = self.target_height or 720
                        tgt_w = self.target_width or 1280
                        if tgt_h > tgt_w:
                            tgt_h, tgt_w = tgt_w, tgt_h
                    
                    self.video_target_sizes[video_idx] = (tgt_h, tgt_w)
                else:
                    # 视频为空，使用默认横屏尺寸
                    self.video_target_sizes[video_idx] = (self.target_height or 720, self.target_width or 1280)
            except Exception as e:
                # 加载失败，使用默认横屏尺寸
                print(f"Warning: Failed to load video {clean_path}: {e}")
                self.video_target_sizes[video_idx] = (self.target_height or 720, self.target_width or 1280)
        
        # 统计横竖屏分布
        portrait_count = sum(1 for h, w in self.video_target_sizes.values() if h > w)
        landscape_count = len(self.video_target_sizes) - portrait_count
        print(f"  横屏视频: {landscape_count}, 竖屏视频: {portrait_count}")
        
        self.total_samples = len(self.video_pairs) * self.frames_per_video
        print(f"Total frame pair samples: {self.total_samples}")
    
    def setup_transforms(self):
        """设置数据增强（不破坏空间对应）"""
        aug_config = self.augmentation_config
        
        transforms_list = []
        
        # ⚠️ 不在这里做resize！
        # 因为我们在__getitem__中已经用_resize_crop做了自适应resize
        # 这里添加transforms.Resize会重复resize并可能破坏横竖屏
        
        # 轻微颜色抖动（可选）
        if aug_config.get('color_jitter', False):
            transforms_list.append(
                transforms.ColorJitter(
                    brightness=aug_config.get('brightness', 0.05),
                    contrast=aug_config.get('contrast', 0.05),
                    saturation=aug_config.get('saturation', 0.05),
                    hue=aug_config.get('hue', 0.02)
                )
            )
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.transform = transforms.Compose(transforms_list)
    
    def __len__(self):
        return self.total_samples
    
    def _save_filtered_frame(self, subtitle_img, clean_img, diff_magnitude, video_name, frame_idx, high_diff_ratio):
        """保存被过滤的帧"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # 不显示图形
            import matplotlib.pyplot as plt
            
            # 创建可视化
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Subtitle
            axes[0].imshow(np.array(subtitle_img))
            axes[0].set_title('Subtitle Frame', fontsize=12)
            axes[0].axis('off')
            
            # Clean
            axes[1].imshow(np.array(clean_img))
            axes[1].set_title('Clean Frame', fontsize=12)
            axes[1].axis('off')
            
            # Difference heatmap
            diff_np = diff_magnitude.cpu().numpy().squeeze()
            im = axes[2].imshow(diff_np, cmap='hot')
            axes[2].set_title(f'Difference (ratio={high_diff_ratio:.3%})', fontsize=12, color='red')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            
            fig.suptitle(f'⚠️ FILTERED: {video_name}_frame{frame_idx}\nReason: high_diff_ratio ({high_diff_ratio:.3%}) < threshold ({self.min_diff_threshold:.3%})', 
                        fontsize=14, fontweight='bold', color='red')
            
            plt.tight_layout()
            save_path = os.path.join(self.filtered_dir, f'{video_name}_frame{frame_idx}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to save filtered frame: {e}")
    
    def _save_kept_frame(self, subtitle_img, clean_img, diff_magnitude, mask, video_name, frame_idx, high_diff_ratio):
        """保存保留的帧（随机采样）"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # 创建可视化
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Subtitle
            axes[0].imshow(np.array(subtitle_img))
            axes[0].set_title('Subtitle Frame', fontsize=12)
            axes[0].axis('off')
            
            # Clean
            axes[1].imshow(np.array(clean_img))
            axes[1].set_title('Clean Frame', fontsize=12)
            axes[1].axis('off')
            
            # Difference heatmap
            diff_np = diff_magnitude.cpu().numpy().squeeze()
            im = axes[2].imshow(diff_np, cmap='hot')
            axes[2].set_title(f'Difference (ratio={high_diff_ratio:.3%})', fontsize=12)
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            
            # Mask
            mask_np = mask.cpu().numpy().squeeze()
            im2 = axes[3].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            axes[3].set_title(f'Mask (ratio={mask_np.mean():.3%})', fontsize=12)
            axes[3].axis('off')
            plt.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)
            
            fig.suptitle(f'✓ KEPT: {video_name}_frame{frame_idx}\nhigh_diff_ratio ({high_diff_ratio:.3%}) >= threshold ({self.min_diff_threshold:.3%})', 
                        fontsize=14, fontweight='bold', color='green')
            
            plt.tight_layout()
            save_path = os.path.join(self.kept_dir, f'{video_name}_frame{frame_idx}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to save kept frame: {e}")
    
    def print_filter_stats(self):
        """打印过滤统计信息"""
        print("\n" + "=" * 80)
        print("📊 数据过滤统计")
        print("=" * 80)
        
        # 总体统计
        total_processed = self.total_filtered + self.total_kept
        if total_processed > 0:
            filter_rate = self.total_filtered / total_processed * 100
            keep_rate = self.total_kept / total_processed * 100
            
            print(f"\n总体统计:")
            print(f"  总处理帧数: {total_processed}")
            print(f"  ✓ 保留: {self.total_kept} ({keep_rate:.1f}%)")
            print(f"  ✗ 过滤: {self.total_filtered} ({filter_rate:.1f}%)")
        
        # 按视频统计
        print(f"\n每个视频的过滤情况:")
        print(f"{'视频名称':<30} {'总帧数':>8} {'保留':>8} {'过滤':>8} {'过滤率':>10}")
        print("-" * 80)
        
        for video_name in sorted(self.filter_stats.keys()):
            stats = self.filter_stats[video_name]
            total = stats['total']
            kept = stats['kept']
            filtered = stats['filtered']
            
            if total > 0:
                filter_rate = filtered / total * 100
                print(f"{video_name:<30} {total:>8} {kept:>8} {filtered:>8} {filter_rate:>9.1f}%")
        
        print("=" * 80)
        
        if self.save_samples and self.save_dir:
            print(f"\n保存的样本图片:")
            print(f"  过滤帧: {self.filtered_dir} (全部保存)")
            filtered_count = len([f for f in os.listdir(self.filtered_dir) if f.endswith('.png')]) if os.path.exists(self.filtered_dir) else 0
            print(f"    已保存: {filtered_count} 张")
            
            # ⭐ kept样本不再保存（节省空间）
            print(f"  保留帧: 不保存 (节省磁盘空间和I/O时间)")
            print("=" * 80)
    
    def __getitem__(self, idx):
        """
        从视频对中采样帧并计算差异
        
        Returns:
            dict: {
                'subtitle': 带字幕帧 [3, H, W]
                'clean': 干净帧 [3, H, W]
                'diff': 差异图 [3, H, W] (可选)
                'mask': 字幕位置mask [1, H, W] (伪标签)
            }
        """
        # 防止无限递归：如果连续尝试多次都被过滤，返回一个dummy样本
        max_retries = 50
        
        for retry in range(max_retries):
            try:
                return self._get_single_item(idx + retry)
            except FilteredException:
                # 这个样本被过滤，尝试下一个
                continue
        
        # 如果重试max_retries次都被过滤，返回一个全零的dummy样本
        # 这样可以避免训练中断
        print(f"⚠️  Warning: Sample {idx} and {max_retries} subsequent samples all filtered. "
              f"Returning dummy sample to avoid crash.")
        
        # 返回dummy样本
        h, w = self.target_height or 720, self.target_width or 1280
        return {
            'subtitle': torch.zeros(3, h, w),
            'clean': torch.zeros(3, h, w),
            'mask': torch.zeros(1, h, w),
        }
    
    def _get_single_item(self, idx):
        """
        获取单个样本（可能抛出FilteredException）
        """
        idx = idx % len(self)  # 确保索引在范围内
        
        video_idx = idx // self.frames_per_video
        frame_offset = idx % self.frames_per_video
        
        clean_video_path, subtitle_video_path = self.video_pairs[video_idx]
        
        try:
            # 加载视频
            clean_video = load_video(clean_video_path)
            subtitle_video = load_video(subtitle_video_path)
            
            num_frames = min(len(clean_video), len(subtitle_video))
            if num_frames == 0:
                raise ValueError(f"Empty video: {clean_video_path}")
            
            # 均匀采样帧
            if num_frames <= self.frames_per_video:
                frame_idx = frame_offset % num_frames
            else:
                segment_size = num_frames / self.frames_per_video
                frame_idx = int(frame_offset * segment_size)
            
            # 获取帧
            clean_frame = (clean_video[frame_idx].detach().cpu().numpy() * 255).astype(np.uint8)
            subtitle_frame = (subtitle_video[frame_idx].detach().cpu().numpy() * 255).astype(np.uint8)
            
            from PIL import Image
            clean_img = Image.fromarray(clean_frame)
            subtitle_img = Image.fromarray(subtitle_frame)
            
            # ⭐ 关键修复：使用预先确定的视频级别目标尺寸
            # 这确保同一视频的所有帧都有相同的尺寸，可以组成batch
            tgt_h, tgt_w = self.video_target_sizes.get(video_idx, (self.target_height or 720, self.target_width or 1280))
            
            def _resize_crop(img, th, tw):
                w, h = img.size
                scale = max(th / h, tw / w)
                nw = int(round(w * scale))
                nh = int(round(h * scale))
                img = img.resize((nw, nh), Image.BILINEAR)
                l = (nw - tw) // 2
                t = (nh - th) // 2
                return img.crop((l, t, l + tw, t + th))
            
            clean_img = _resize_crop(clean_img, tgt_h, tgt_w)
            subtitle_img = _resize_crop(subtitle_img, tgt_h, tgt_w)
            
            # 应用相同的随机种子
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            clean_tensor = self.transform(clean_img)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            subtitle_tensor = self.transform(subtitle_img)
            
            # 计算差异图
            diff_tensor = torch.abs(subtitle_tensor - clean_tensor)
            
            # 生成字幕位置mask（伪标签）- 自适应阈值
            diff_magnitude = diff_tensor.mean(dim=0, keepdim=True)  # [1, H, W]
            
            # ⭐ 关键过滤：检查是否有足够的差异（是否有字幕）
            
            # 方法：检查差异的绝对强度
            # diff_magnitude范围通常是0-2（因为输入归一化到[-1,1]）
            # 我们检查是否有足够多的像素超过阈值
            
            # 计算高差异像素的比例
            # 使用固定阈值（如0.1）来判断哪些像素有明显差异
            high_diff_pixels = (diff_magnitude > 0.1).float()
            high_diff_ratio = high_diff_pixels.mean().item()
            
            # 获取视频名称（用于统计）
            video_name = os.path.basename(clean_video_path).split('.')[0]
            if video_name not in self.filter_stats:
                self.filter_stats[video_name] = {'total': 0, 'filtered': 0, 'kept': 0}
            self.filter_stats[video_name]['total'] += 1
            
            # 先生成mask（在过滤检查之前）
            # 自适应阈值：使用差异分布的统计特性
            # 方案：mean + k * std，其中k由配置的diff_threshold控制
            # diff_threshold现在表示"标准差的倍数"
            diff_mean = diff_magnitude.mean()
            diff_std = diff_magnitude.std()
            
            # 自适应阈值 = mean + k*std
            # diff_threshold=0.1表示mean+0.1*std
            # diff_threshold=1.0表示mean+1.0*std（更严格）
            adaptive_threshold = diff_mean + self.diff_threshold * diff_std
            
            # 限制最小阈值，避免噪声
            min_threshold = 0.01
            adaptive_threshold = max(adaptive_threshold.item(), min_threshold)
            
            mask = (diff_magnitude > adaptive_threshold).float()
            
            # 如果高差异像素占比太低，说明没有明显字幕
            # min_diff_threshold现在表示"高差异像素占比的最小值"
            # 例如：0.003表示至少0.3%的像素有明显差异才保留
            if high_diff_ratio < self.min_diff_threshold:
                self.filter_stats[video_name]['filtered'] += 1
                self.total_filtered += 1
                
                # 保存被过滤的帧（用于检查）
                if self.save_samples and self.save_dir:
                    self._save_filtered_frame(
                        subtitle_img, clean_img, diff_magnitude, 
                        video_name, frame_idx, high_diff_ratio
                    )
                
                if idx % 100 == 0:
                    print(f"[Sample {idx}] ⚠️ FILTERED: {video_name} frame{frame_idx}, high_diff_ratio={high_diff_ratio:.3%} < {self.min_diff_threshold:.3%}")
                
                # 抛出异常而不是递归调用，避免无限递归
                raise FilteredException(f"Sample {idx} filtered: high_diff_ratio={high_diff_ratio:.3%}")
            
            # 记录保留的样本
            self.filter_stats[video_name]['kept'] += 1
            self.total_kept += 1
            
            # ⭐ 不再保存kept样本（节省空间和时间）
            # 只保存filtered样本用于检查为什么被过滤
            # if self.save_samples and self.save_dir:
            #     # 限制最大保存数量，避免保存太多
            #     if self.saved_kept_count < self.max_kept_samples:
            #         if np.random.random() < self.save_sample_prob:
            #             self._save_kept_frame(
            #                 subtitle_img, clean_img, diff_magnitude, mask,
            #                 video_name, frame_idx, high_diff_ratio
            #             )
            #             self.saved_kept_count += 1
            #             
            #             # 每保存10个打印一次进度
            #             if self.saved_kept_count % 10 == 0:
            #                 print(f"  已保存 {self.saved_kept_count}/{self.max_kept_samples} 个kept样本")
            
            # 调试信息（可选，每100个样本打印一次）
            if idx % 100 == 0:
                ratio = mask.mean().item()
                print(f"[Sample {idx}] diff_mean={diff_mean:.4f}, diff_std={diff_std:.4f}, "
                      f"threshold={adaptive_threshold:.4f}, mask_ratio={ratio:.3%}, "
                      f"high_diff_ratio={high_diff_ratio:.3%}")
            
            return {
                'subtitle': subtitle_tensor,  # 输入：带字幕的帧
                'clean': clean_tensor,        # 用于生成mask（不输入网络）
                'mask': mask,                 # 字幕位置mask（引导对比学习）
            }
            
        except FilteredException:
            # 重新抛出过滤异常
            raise
        except Exception as e:
            print(f"Error loading video pair: {clean_video_path}")
            print(f"Error: {e}")
            # 加载错误时抛出异常，由外层重试
            raise FilteredException(f"Error loading video: {e}")


def setup_distributed():
    """设置分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        # ⭐ 增加NCCL超时时间：从默认10分钟增加到60分钟
        # 因为每个step需要约4分钟，ALLREDUCE可能需要更长时间
        import datetime
        timeout = datetime.timedelta(minutes=60)
        dist.init_process_group(
            backend='nccl', 
            init_method='env://', 
            timeout=timeout
        )
        torch.cuda.set_device(local_rank)
        
        if rank == 0:
            print(f"✓ 分布式初始化完成，NCCL超时时间: {timeout}")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch_disentangled(model, dataloader, optimizer, scaler, epoch, config, 
                                 writer=None, rank=0, checkpoint_dir=None, adaptive_loss_weights=None, scheduler=None):
    """
    训练一个epoch（DisentangleAdapter版本）
    
    使用解耦学习的三种损失：
    1. Mask Loss: Focal + Dice + IoU
    2. Disentangle Loss: 正交约束
    3. Reconstruction Loss: L1
    
    Args:
        adaptive_loss_weights: 独立的自适应权重模块（不是模型的一部分，避免DDP问题）
    """
    model.train()
    
    total_loss = 0.0
    total_mask_loss = 0.0
    total_disentangle_loss = 0.0
    total_recon_loss = 0.0
    total_iou = 0.0
    num_batches = 0
    
    # 损失权重（如果使用自适应权重则会被覆盖）
    use_adaptive_weights = config['adapter'].get('use_adaptive_loss_weights', False)
    mask_loss_weight = config['adapter'].get('mask_loss_weight', 1.0)
    disentangle_loss_weight = config['adapter'].get('disentangle_loss_weight', 0.5)
    reconstruction_loss_weight = config['adapter'].get('reconstruction_loss_weight', 0.3)
    
    # ⭐ 自适应权重作为独立模块传入（不是模型的一部分）
    has_adaptive_weights = adaptive_loss_weights is not None
    
    # 获取world_size用于分布式同步
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # 梯度累积步数
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    
    # 计算checkpoint保存点
    total_steps = len(dataloader)
    checkpoint_steps = {
        int(total_steps * 0.25): '25pct',
        int(total_steps * 0.50): '50pct', 
        int(total_steps * 0.75): '75pct',
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0))
    
    # ⭐ 初始化梯度（梯度累积需要）
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        # 获取数据
        subtitle = batch['subtitle'].cuda()  # 带字幕的帧
        clean = batch.get('clean')           # 干净帧（用于重建监督）
        if clean is not None:
            clean = clean.cuda()
        mask = batch['mask'].cuda()          # 字幕位置mask
        
        # 前向传播
        with autocast(enabled=(config['training']['mixed_precision'] == 'bf16'), dtype=torch.bfloat16):
            # 模型前向：subtitle → (content_feat, subtitle_feat, mask_logits, recon_clean)
            content_feat, subtitle_feat, mask_logits, recon_clean = model(subtitle, return_all=True)
            
            # 1. Mask Loss (Focal + Dice + IoU)
            mask_loss, mask_stats = compute_mask_loss(mask_logits, mask)
            
            # 2. Disentangle Loss (正交约束)
            disentangle_loss = compute_disentangle_loss(content_feat, subtitle_feat)
            
            # 3. Reconstruction Loss (L1)
            if recon_clean is not None and clean is not None:
                recon_loss = compute_reconstruction_loss(recon_clean, clean)
            else:
                recon_loss = torch.tensor(0.0, device=subtitle.device)
        
        # ⭐ 总损失计算移到 autocast 外面（确保 adaptive_loss_weights 的梯度精度）
        # adaptive_loss_weights 的 log_vars 是 float32 Parameter，
        # PyTorch 会自动将 bf16 loss 提升到 float32 进行计算，无需显式转换
        if has_adaptive_weights:
            # 使用独立的自适应权重模块（log_vars 在 float32 精度下更新）
            loss, current_weights, precisions = adaptive_loss_weights(
                [mask_loss, disentangle_loss, recon_loss]
            )
        else:
            # 使用固定权重
            loss = (
                mask_loss_weight * mask_loss +
                disentangle_loss_weight * disentangle_loss +
                reconstruction_loss_weight * recon_loss
            )
            current_weights = [mask_loss_weight, disentangle_loss_weight, reconstruction_loss_weight]
            precisions = None
        
        # 保存未缩放的loss用于显示
        loss_for_display = loss.item()
        
        # 梯度累积：缩放损失
        loss = loss / gradient_accumulation_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # ⭐ 梯度累积：只在累积步数达到时才更新参数
        if (step + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪（包括模型和自适应权重）
            if config['training'].get('max_grad_norm', 0) > 0:
                scaler.unscale_(optimizer)
                # 裁剪所有参数（包括adaptive_loss_weights）
                params_to_clip = list(model.parameters())
                if has_adaptive_weights and adaptive_loss_weights is not None:
                    params_to_clip.extend(list(adaptive_loss_weights.parameters()))
                
                torch.nn.utils.clip_grad_norm_(
                    params_to_clip, 
                    config['training']['max_grad_norm']
                )
            
            # ⭐ 调试信息：保存更新前的参数和梯度（确认模型在更新）
            # 前5次优化更新 = 前 5*gradient_accumulation_steps 个训练步
            debug_this_step = (rank == 0 and step < 5 * gradient_accumulation_steps)
            first_param = None
            first_param_name = None
            grad_norms = []
            
            if debug_this_step:
                # 保存更新前的参数（第一个卷积层）
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        first_param = param.data.clone()
                        first_param_name = name
                        break
                
                # 收集梯度统计
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
            
            # 执行优化器步骤
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # ⭐ 学习率调度（每个优化器步骤后调用，实现warmup）
            if scheduler is not None:
                scheduler.step()
            
            # ⭐ 调试信息：打印更新后的信息
            if debug_this_step:
                print(f"\n[调试 优化步骤 {(step+1)//gradient_accumulation_steps}] (训练step {step+1}):")
                
                # 打印更新后的学习率
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   更新后学习率: {current_lr:.2e}")
                
                # 打印梯度统计
                if grad_norms:
                    print(f"   梯度范数: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, mean={sum(grad_norms)/len(grad_norms):.6f}")
                
                # 检查参数是否更新
                if first_param is not None:
                    for name, param in model.named_parameters():
                        if name == first_param_name:
                            param_diff = (param.data - first_param).abs().mean().item()
                            print(f"   参数变化 ({name}): {param_diff:.6e}")
                            if param_diff > 1e-8:
                                print(f"   ✅ 参数已更新")
                            else:
                                print(f"   ⚠️  参数变化很小，可能学习率太低")
                            break
            
            # ⭐ 分布式训练：同步adaptive_loss_weights的参数（只在更新后同步）
            if has_adaptive_weights and world_size > 1:
                # 在所有GPU上广播log_vars参数（从rank 0）
                for param in adaptive_loss_weights.parameters():
                    dist.broadcast(param.data, src=0)
        
        # 统计（使用未缩放的loss）
        total_loss += loss_for_display
        total_mask_loss += mask_loss.item()
        total_disentangle_loss += disentangle_loss.item()
        total_recon_loss += recon_loss.item()
        total_iou += mask_stats['iou']
        num_batches += 1
        
        # 计算mask预测指标
        with torch.no_grad():
            metrics = compute_mask_metrics(mask_logits, mask)
        
        # 更新进度条（包含权重信息，使用未缩放的loss）
        postfix_dict = {
            'loss': f"{loss_for_display:.4f}",
            'mask': f"{mask_loss.item():.4f}",
            'dis': f"{disentangle_loss.item():.3f}",
            'rec': f"{recon_loss.item():.3f}",
            'iou': f"{metrics['iou']:.3f}",
        }
        
        # 添加权重信息
        if has_adaptive_weights:
            postfix_dict['w_mask'] = f"{current_weights[0]:.2f}"
            postfix_dict['w_dis'] = f"{current_weights[1]:.2f}"
            postfix_dict['w_rec'] = f"{current_weights[2]:.2f}"
        
        pbar.set_postfix(postfix_dict)
        
        # ⭐ 每10步打印一次log_vars的实际值（调试）
        if has_adaptive_weights and rank == 0 and step % 10 == 0 and step > 0:
            log_vars = adaptive_loss_weights.get_log_vars()
            print(f"\n[Step {step}] log_vars: {log_vars}")
            print(f"  Weights (precision): mask={current_weights[0]:.4f}, "
                  f"disentangle={current_weights[1]:.4f}, recon={current_weights[2]:.4f}")
            
            # 检查梯度
            if hasattr(adaptive_loss_weights, 'log_vars'):
                if adaptive_loss_weights.log_vars.grad is not None:
                    grad_norm = adaptive_loss_weights.log_vars.grad.norm().item()
                    print(f"  log_vars grad norm: {grad_norm:.6f}")
                else:
                    print(f"  ⚠️  log_vars grad is None!")
        
        # TensorBoard日志
        if writer is not None and rank == 0:
            global_step = epoch * len(dataloader) + step
            
            # 损失（使用未缩放的loss）
            writer.add_scalar('train/total_loss', loss_for_display, global_step)
            writer.add_scalar('train/mask_loss', mask_loss.item(), global_step)
            writer.add_scalar('train/mask_focal', mask_stats['focal'], global_step)
            writer.add_scalar('train/mask_dice', mask_stats['dice'], global_step)
            writer.add_scalar('train/mask_iou_loss', mask_stats['iou_loss'], global_step)
            writer.add_scalar('train/disentangle_loss', disentangle_loss.item(), global_step)
            writer.add_scalar('train/recon_loss', recon_loss.item(), global_step)
            
            # 指标
            writer.add_scalar('train/mask_iou', metrics['iou'], global_step)
            writer.add_scalar('train/mask_acc', metrics['acc'], global_step)
            writer.add_scalar('train/pred_pos_ratio', metrics['pred_pos'], global_step)
            writer.add_scalar('train/gt_pos_ratio', metrics['gt_pos'], global_step)
            
            # 自适应权重（如果启用）
            if has_adaptive_weights:
                writer.add_scalar('train/weight_mask', current_weights[0], global_step)
                writer.add_scalar('train/weight_disentangle', current_weights[1], global_step)
                writer.add_scalar('train/weight_recon', current_weights[2], global_step)
                if precisions is not None:
                    writer.add_scalar('train/precision_mask', precisions[0], global_step)
                    writer.add_scalar('train/precision_disentangle', precisions[1], global_step)
                    writer.add_scalar('train/precision_recon', precisions[2], global_step)
        
        # 定期保存checkpoint
        if checkpoint_dir is not None and step in checkpoint_steps:
            if rank == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, 
                    f"epoch_{epoch}_{checkpoint_steps[step]}.pt"
                )
                _model = model.module if hasattr(model, 'module') else model
                checkpoint_dict = {
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': _model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                # ⭐ 保存自适应权重
                if adaptive_loss_weights is not None:
                    checkpoint_dict['adaptive_loss_weights'] = adaptive_loss_weights.state_dict()
                # ⭐ 保存scheduler和scaler（用于resume）
                if scheduler is not None:
                    checkpoint_dict['scheduler'] = scheduler.state_dict()
                if scaler is not None:
                    checkpoint_dict['scaler'] = scaler.state_dict()
                
                torch.save(checkpoint_dict, checkpoint_path)
                print(f"\n✓ Checkpoint saved: {checkpoint_path}")
    
    # Epoch统计
    avg_loss = total_loss / num_batches
    avg_mask_loss = total_mask_loss / num_batches
    avg_disentangle_loss = total_disentangle_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_iou = total_iou / num_batches
    
    if rank == 0:
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Mask Loss: {avg_mask_loss:.4f}")
        print(f"  Avg Disentangle Loss: {avg_disentangle_loss:.4f}")
        print(f"  Avg Recon Loss: {avg_recon_loss:.4f}")
        print(f"  Avg IoU: {avg_iou:.4f}")
    
    return {
        'loss': avg_loss,
        'mask_loss': avg_mask_loss,
        'disentangle_loss': avg_disentangle_loss,
        'recon_loss': avg_recon_loss,
        'iou': avg_iou,
    }


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch, config, 
                    writer=None, rank=0, checkpoint_dir=None):
    """训练一个epoch（对比学习版本）"""
    model.train()
    
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    # 计算checkpoint保存点
    total_steps = len(dataloader)
    checkpoint_steps = {
        int(total_steps * 0.25): '25pct',
        int(total_steps * 0.50): '50pct', 
        int(total_steps * 0.75): '75pct',
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0))
    
    for step, batch in enumerate(pbar):
        # 获取数据
        subtitle = batch['subtitle'].cuda()  # 带字幕的帧（训练和推理都用这个）
        mask = batch['mask'].cuda()          # 字幕位置mask（从差分生成，作为监督信号）
        # 注意：clean只在数据集中用于生成mask，不输入网络
        
        # 前向传播
        with autocast(enabled=(config['training']['mixed_precision'] == 'bf16'), dtype=torch.bfloat16):
            # ✅ 正确逻辑：训练和推理都只输入subtitle（3通道）
            # 差分图只用于生成mask（监督信号），不输入网络
            # encoder需要学会从subtitle中直接识别字幕位置
            
            # 获取subtitle的原始尺寸（用于mask预测）
            target_h, target_w = subtitle.shape[2], subtitle.shape[3]
            
            # 获取模型输出（对比学习特征 + mask预测）
            h, pred_mask = model(subtitle, clean=None, return_mask=True, target_size=(target_h, target_w))
            
            # 1. 对比学习损失（mask引导特征学习）
            mask_detached = mask.detach()
            contrastive_loss, stats = criterion(h, mask_detached)
            
            # 2. 分割损失（mask预测）
            # 使用BCE loss + Dice loss的组合
            if pred_mask is not None:
                # ⭐ 计算正样本权重（处理类别极度不平衡）
                # 正样本占比很小（0.25%-3%），需要增加正样本权重
                pos_ratio = mask.mean().item()  # 正样本占比
                if pos_ratio > 0:
                    # 动态计算pos_weight：使负样本和正样本的总权重相等
                    # pos_weight = 负样本数 / 正样本数 = (1 - pos_ratio) / pos_ratio
                    # 
                    # ⚠️ 关键修正：针对极小目标（0.2%-0.8%）防止过度预测
                    # 
                    # 实验发现：之前权重太高导致严重过度预测
                    # 例：pos_ratio=0.8%, 使用权重11 → pred_pos=26%（33倍！）
                    # 
                    # 解决方案：使用更激进的衰减 + 添加过度预测惩罚
                    raw_pos_weight = (1.0 - pos_ratio) / pos_ratio
                    
                    # 使用0.35次方衰减（比平方根0.5更激进）
                    pos_weight = raw_pos_weight ** 0.35
                    pos_weight = min(pos_weight, 8.0)  # 降低上限到8
                    
                    # 权重效果（字幕占比 → 最终权重）：
                    # 0.2% (raw=499) → 4.7 → 4.7
                    # 0.5% (raw=199) → 3.8 → 3.8
                    # 0.8% (raw=124) → 3.3 → 3.3
                    # 1.0% (raw=99)  → 2.9 → 2.9
                    # 目标：让pred_pos接近pos_ratio（允许2-3倍内）
                else:
                    pos_weight = 1.0
                
                pos_weight_tensor = torch.tensor([pos_weight], dtype=pred_mask.dtype, device=pred_mask.device)
                
                # BCE loss with pos_weight (使用 with_logits 版本，autocast 安全)
                bce_loss = F.binary_cross_entropy_with_logits(
                    pred_mask, mask, 
                    pos_weight=pos_weight_tensor,
                    reduction='mean'
                )
                
                # Dice loss（处理类别不平衡）
                # 需要先对 logits 应用 sigmoid
                smooth = 1e-5
                pred_prob = torch.sigmoid(pred_mask)  # 将 logits 转为概率
                pred_flat = pred_prob.view(-1)
                target_flat = mask.view(-1)
                intersection = (pred_flat * target_flat).sum()
                dice_loss = 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
                
                # ⭐ 添加过度预测惩罚（关键！防止预测过多）
                # 计算预测和真实的占比
                pred_pos_ratio = pred_flat.mean()
                gt_pos_ratio = target_flat.mean()
                
                # 如果预测过多，添加惩罚
                # 允许预测是真实的2.5倍内（考虑模糊边界）
                # 例：gt=0.8% → 允许pred<=2.0%，超过则惩罚
                over_pred_penalty = F.relu(pred_pos_ratio - gt_pos_ratio * 2.5)
                
                # 组合分割损失：BCE + Dice + 过度预测惩罚
                # 过度预测惩罚权重：2.0（强力抑制过度预测）
                segmentation_loss = bce_loss + 1.0 * dice_loss + 2.0 * over_pred_penalty
                
                # 总损失：对比学习 + 分割
                seg_weight = config['adapter'].get('segmentation_loss_weight', 0.5)
                loss = contrastive_loss + seg_weight * segmentation_loss
                
                # 更新统计信息
                stats['seg_loss'] = segmentation_loss.item()
                stats['bce_loss'] = bce_loss.item()
                stats['dice_loss'] = dice_loss.item()
                stats['over_pred_penalty'] = over_pred_penalty.item()
                stats['pred_pos_ratio'] = pred_pos_ratio.item()
                stats['gt_pos_ratio'] = gt_pos_ratio.item()
                stats['pos_weight'] = pos_weight  # 记录使用的权重
                
                # 计算mask预测的IoU（用于监控）
                with torch.no_grad():
                    pred_binary = (pred_prob > 0.5).float()  # 使用 sigmoid 后的概率
                    
                    # ⭐ 修正：按样本计算IoU再平均（向量化实现，高效）
                    # 原来：整个batch的总IoU → 被大样本主导
                    # 现在：每个样本的IoU平均 → 每个样本权重相等，对小目标更公平
                    
                    # 将spatial维度flatten：(B, 1, H, W) → (B, H*W)
                    pred_flat = pred_binary.flatten(start_dim=1)  # (B, H*W)
                    mask_flat = mask.flatten(start_dim=1)         # (B, H*W)
                    
                    # 按样本计算交集和并集：(B,)
                    intersection = (pred_flat * mask_flat).sum(dim=1)  # (B,)
                    union = pred_flat.sum(dim=1) + mask_flat.sum(dim=1) - intersection  # (B,)
                    
                    # 按样本计算IoU：(B,)
                    # 注意处理除零：union=0时（两个都是空mask）
                    iou_per_sample = (intersection + smooth) / (union + smooth)  # (B,)
                    
                    # 取batch平均
                    iou = iou_per_sample.mean()
                    stats['mask_iou'] = iou.item()
                    
                    # ⭐ 监控预测的正样本比例（防止模型退化为全预测0）
                    pred_pos_ratio = pred_binary.mean().item()
                    gt_pos_ratio = mask.mean().item()
                    stats['pred_pos_ratio'] = pred_pos_ratio
                    stats['gt_pos_ratio'] = gt_pos_ratio
                    stats['pos_weight'] = pos_weight  # 记录当前使用的正样本权重
            else:
                # 如果没有mask预测，只用对比学习损失
                loss = contrastive_loss
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        # 梯度裁剪
        if config['training'].get('max_grad_norm', 0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
        
        scaler.step(optimizer)
        scaler.update()
        
        # 统计
        total_loss += stats['loss']
        total_accuracy += stats['accuracy']
        num_batches += 1
        
        # 更新进度条（包含过滤统计）
        if hasattr(dataloader.dataset, 'dataset'):
            # 如果是Subset，获取原始dataset
            original_dataset = dataloader.dataset.dataset
        else:
            original_dataset = dataloader.dataset
        
        filter_info = {}
        if hasattr(original_dataset, 'total_filtered') and hasattr(original_dataset, 'total_kept'):
            total = original_dataset.total_filtered + original_dataset.total_kept
            if total > 0:
                filter_rate = original_dataset.total_filtered / total * 100
                filter_info['filtered'] = f"{original_dataset.total_filtered}/{total}({filter_rate:.1f}%)"
        
        postfix_dict = {
            'loss': f"{stats['loss']:.4f}",
            'acc': f"{stats['accuracy']:.4f}",
            'pos_ratio': f"{stats['positive_ratio']:.3f}",
        }
        
        # 添加mask预测统计
        if 'mask_iou' in stats:
            postfix_dict['iou'] = f"{stats['mask_iou']:.3f}"
        if 'seg_loss' in stats:
            postfix_dict['seg'] = f"{stats['seg_loss']:.3f}"
        # 添加预测正样本比例（用于检测模型是否退化）
        if 'pred_pos_ratio' in stats:
            postfix_dict['pred_pos'] = f"{stats['pred_pos_ratio']:.4f}"
        # 添加过度预测惩罚（重要监控指标）
        if 'over_pred_penalty' in stats:
            postfix_dict['over_penalty'] = f"{stats['over_pred_penalty']:.3f}"
        
        # 添加过滤统计
        postfix_dict.update(filter_info)
        
        pbar.set_postfix(postfix_dict)
        
        # 记录日志
        if writer and step % config['logging']['log_interval'] == 0:
            global_step = epoch * len(dataloader) + step
            writer.add_scalar('train/loss', stats['loss'], global_step)
            writer.add_scalar('train/accuracy', stats['accuracy'], global_step)
            writer.add_scalar('train/positive_ratio', stats['positive_ratio'], global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            
            # 记录mask预测指标
            if 'mask_iou' in stats:
                writer.add_scalar('train/mask_iou', stats['mask_iou'], global_step)
            if 'seg_loss' in stats:
                writer.add_scalar('train/seg_loss', stats['seg_loss'], global_step)
                writer.add_scalar('train/bce_loss', stats['bce_loss'], global_step)
                writer.add_scalar('train/dice_loss', stats['dice_loss'], global_step)
            # 记录预测正样本比例和权重（用于监控模型是否退化）
            if 'pred_pos_ratio' in stats:
                writer.add_scalar('train/pred_pos_ratio', stats['pred_pos_ratio'], global_step)
                writer.add_scalar('train/gt_pos_ratio', stats['gt_pos_ratio'], global_step)
                writer.add_scalar('train/pos_weight', stats['pos_weight'], global_step)
            # 记录过度预测惩罚（关键监控指标）
            if 'over_pred_penalty' in stats:
                writer.add_scalar('train/over_pred_penalty', stats['over_pred_penalty'], global_step)
            
            # 记录过滤统计
            if hasattr(dataloader.dataset, 'dataset'):
                original_dataset = dataloader.dataset.dataset
            else:
                original_dataset = dataloader.dataset
            
            if hasattr(original_dataset, 'total_filtered') and hasattr(original_dataset, 'total_kept'):
                total = original_dataset.total_filtered + original_dataset.total_kept
                if total > 0:
                    filter_rate = original_dataset.total_filtered / total
                    writer.add_scalar('train/filter_rate', filter_rate, global_step)
                    writer.add_scalar('train/kept_samples', original_dataset.total_kept, global_step)
                    writer.add_scalar('train/filtered_samples', original_dataset.total_filtered, global_step)
        
        # ⭐ 检测模型异常（每10步检查一次）
        if rank == 0 and step > 0 and step % 10 == 0 and 'pred_pos_ratio' in stats:
            pred_pos = stats['pred_pos_ratio']
            gt_pos = stats['gt_pos_ratio']
            
            # 情况1: 预测不足（倾向于全预测0）
            if pred_pos < gt_pos * 0.1 and gt_pos > 0:
                print(f"\n⚠️ [Step {step}] 模型退化-预测不足: pred_pos={pred_pos:.4f} << gt_pos={gt_pos:.4f}")
                print(f"   建议: 1) 增加pos_weight上限; 2) 增加seg_weight; 3) 增加dice_weight")
            
            # 情况2: 过度预测（到处乱预测）
            elif pred_pos > gt_pos * 3.0 and gt_pos > 0:
                print(f"\n⚠️ [Step {step}] 模型异常-过度预测: pred_pos={pred_pos:.4f} >> gt_pos={gt_pos:.4f}")
                print(f"   建议: 1) 降低pos_weight上限; 2) 降低seg_weight; 3) 降低dice_weight")
                print(f"   当前pos_weight={stats.get('pos_weight', 'N/A'):.1f}")
        
        # 每100步打印一次详细过滤统计
        if rank == 0 and step > 0 and step % 100 == 0:
            if hasattr(dataloader.dataset, 'dataset'):
                original_dataset = dataloader.dataset.dataset
            else:
                original_dataset = dataloader.dataset
            
            if hasattr(original_dataset, 'total_filtered') and hasattr(original_dataset, 'total_kept'):
                total = original_dataset.total_filtered + original_dataset.total_kept
                if total > 0:
                    filter_rate = original_dataset.total_filtered / total * 100
                    keep_rate = original_dataset.total_kept / total * 100
                    print(f"\n[Step {step}] 过滤统计: 总处理={total}, 保留={original_dataset.total_kept}({keep_rate:.1f}%), 过滤={original_dataset.total_filtered}({filter_rate:.1f}%)")
                    
                    # 显示过滤率最高的前5个视频
                    if hasattr(original_dataset, 'filter_stats'):
                        video_stats = []
                        for video_name, stats_dict in original_dataset.filter_stats.items():
                            if stats_dict['total'] > 0:
                                vf_rate = stats_dict['filtered'] / stats_dict['total'] * 100
                                video_stats.append((video_name, vf_rate, stats_dict))
                        
                        if video_stats:
                            video_stats.sort(key=lambda x: x[1], reverse=True)
                            print(f"  过滤率最高的5个视频:")
                            for i, (vname, vf_rate, vstats) in enumerate(video_stats[:5]):
                                print(f"    {i+1}. {vname}: {vstats['filtered']}/{vstats['total']} ({vf_rate:.1f}%)")
        
        # 保存进度checkpoint
        if rank == 0 and checkpoint_dir and step in checkpoint_steps:
            progress_name = checkpoint_steps[step]
            save_path = os.path.join(checkpoint_dir, f'checkpoint_{progress_name}_epoch{epoch}.pt')
            
            # 准备checkpoint
            checkpoint_dict = {
                'epoch': epoch,
                'step': step,
                'progress': progress_name,
                'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': stats['loss'],
                'accuracy': stats['accuracy'],
                'config': config
            }
            
            # 保存mask_head权重（如果有）
            _model = model.module if hasattr(model, 'module') else model
            if hasattr(_model, 'mask_head'):
                checkpoint_dict['mask_head'] = _model.mask_head.state_dict()
                if 'mask_iou' in stats:
                    checkpoint_dict['mask_iou'] = stats['mask_iou']
            
            torch.save(checkpoint_dict, save_path)
            print(f"\n💾 Saved checkpoint at {progress_name}: {save_path}")
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def validate(model, dataloader, criterion, epoch, config, writer=None, rank=0):
    """
    验证函数（匹配解耦学习训练）
    
    注意：验证集可能没有clean帧，所以reconstruction loss设为0
    """
    model.eval()
    
    total_loss = 0.0
    total_mask_loss = 0.0
    total_disentangle_loss = 0.0
    total_mask_iou = 0.0
    num_batches = 0
    
    # 获取损失权重（用于显示，验证时使用固定权重）
    mask_loss_weight = config['adapter'].get('mask_loss_weight', 1.0)
    disentangle_loss_weight = config['adapter'].get('disentangle_loss_weight', 0.5)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", disable=(rank != 0)):
            subtitle = batch['subtitle'].cuda()
            mask = batch['mask'].cuda()
            
            with autocast(enabled=(config['training']['mixed_precision'] == 'bf16'), dtype=torch.bfloat16):
                # 模型前向（与训练循环一致）
                content_feat, subtitle_feat, mask_logits, recon_clean = model(subtitle, return_all=True)
                
                # 1. Mask Loss
                mask_loss, mask_stats = compute_mask_loss(mask_logits, mask)
                
                # 2. Disentangle Loss
                disentangle_loss = compute_disentangle_loss(content_feat, subtitle_feat)
                
                # 3. Reconstruction Loss（验证时设为0，因为没有clean帧）
                recon_loss = torch.tensor(0.0, device=subtitle.device)
            
            # 总损失（使用固定权重）
            loss = (
                mask_loss_weight * mask_loss +
                disentangle_loss_weight * disentangle_loss +
                0.0 * recon_loss  # 验证时recon_loss不参与
            )
            
            total_loss += loss.item()
            total_mask_loss += mask_loss.item()
            total_disentangle_loss += disentangle_loss.item()
            total_mask_iou += mask_stats.get('iou', 0.0)
            num_batches += 1
    
    # 计算平均值
    avg_loss = total_loss / num_batches
    avg_mask_loss = total_mask_loss / num_batches
    avg_disentangle_loss = total_disentangle_loss / num_batches
    avg_mask_iou = total_mask_iou / num_batches
    
    # 记录到TensorBoard
    if writer and rank == 0:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/mask_loss', avg_mask_loss, epoch)
        writer.add_scalar('val/disentangle_loss', avg_disentangle_loss, epoch)
        writer.add_scalar('val/mask_iou', avg_mask_iou, epoch)
    
    # 返回总损失和mask IoU（作为accuracy的替代）
    return avg_loss, avg_mask_iou


def main():
    parser = argparse.ArgumentParser(description='SEDiT-WAN Adapter Stage 1 Training (Difference Contrastive)')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='从checkpoint恢复训练的路径')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置分布式
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print("=" * 60)
        print("SEDiT-WAN Adapter Stage 1 Training (Difference Contrastive)")
        print("差异对比学习 | Difference-Aware Contrastive Learning")
        print("=" * 60)
        print(f"Config: {args.config}")
        print(f"World size: {world_size}")
        print("=" * 60)
        print("\n🔧 核心思想:")
        print("  ✅ 学习差异特征（字幕位置）")
        print("  ✅ 区域对比学习（字幕区域 vs 干净区域）")
        print("  ✅ 保留对比学习框架（创新点）")
        print("  ✅ 自动生成字幕mask作为监督")
        print("=" * 60)
    
    # 创建模型（根据配置选择）
    adapter_method = config['adapter'].get('method', 'contrastive')  # 'contrastive' or 'disentangled'
    
    if adapter_method == 'disentangled':
        # 使用DisentangleAdapter（解耦学习）
        # 根据配置选择使用多尺度版本还是原始版本
        use_multiscale = config['adapter'].get('use_multiscale', False)
        fusion_layer = config['adapter'].get('fusion_layer', 'layer2')
        
        if rank == 0:
            print("\n" + "=" * 60)
            if use_multiscale:
                print("使用 MultiscaleDisentangleAdapter (多尺度解耦学习)")
                print(f"  ✅ FPN特征融合，使用 {fusion_layer}")
                if fusion_layer == 'layer1':
                    print("  ✅ 4x下采样，最多细节，适合超小字幕")
                elif fusion_layer == 'layer2':
                    print("  ✅ 8x下采样，平衡细节和语义")
                elif fusion_layer == 'layer1+2':
                    print("  ✅ 融合layer1+2，最强细节保留")
                print("  ✅ 更好地保留小字幕细节")
            else:
                print("使用 DisentangleAdapter (标准解耦学习)")
                print("  ⚠️  32x下采样，小字幕可能丢失")
            print("=" * 60)
        
        # 创建模型
        model_kwargs = {
            'backbone': config['adapter']['backbone'],
            'encoder_output_dim': config['adapter']['encoder_output_dim'],
            'content_dim': config['adapter'].get('content_dim', 256),
            'subtitle_dim': config['adapter'].get('subtitle_dim', 256),
            'use_reconstruction': config['adapter'].get('use_reconstruction', True),
            'pretrained': config['adapter']['use_pretrained_backbone'],
            'backbone_weight_path': config['adapter'].get('backbone_weight_path', None),
        }
        
        if use_multiscale:
            model_kwargs['use_multiscale'] = True
            model_kwargs['fusion_layer'] = fusion_layer  # 传递fusion_layer参数
            
            # ⭐ 不在模型中创建自适应权重（避免DDP问题）
            # 自适应权重将作为独立模块在训练循环中管理
            model_kwargs['use_adaptive_loss_weights'] = False
            
            model = MultiscaleDisentangledAdapter(**model_kwargs)
        else:
            model = DisentangledSubtitleAdapter(**model_kwargs)
        
        if rank == 0:
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            # 计算各模块参数量（兼容多尺度和原始版本）
            if use_multiscale:
                # 多尺度版本：分层backbone
                encoder_params = sum(p.numel() for p in model.stem.parameters())
                encoder_params += sum(p.numel() for p in model.layer1.parameters())
                encoder_params += sum(p.numel() for p in model.layer2.parameters())
                encoder_params += sum(p.numel() for p in model.layer3.parameters())
                encoder_params += sum(p.numel() for p in model.layer4.parameters())
                fusion_params = sum(p.numel() for p in model.fusion.parameters())
                fusion_params += sum(p.numel() for p in model.fusion_projection.parameters())
            else:
                # 原始版本
                encoder_params = sum(p.numel() for p in model.backbone.parameters())
                encoder_params += sum(p.numel() for p in model.feature_projection.parameters())
                fusion_params = 0
            
            disentangle_params = sum(p.numel() for p in model.disentangle_head.parameters())
            mask_params = sum(p.numel() for p in model.mask_head.parameters())
            decoder_params = sum(p.numel() for p in model.decoder.parameters()) if model.use_reconstruction else 0
            
            print(f"\nModel parameters:")
            print(f"  Encoder (Backbone): {encoder_params:,}")
            if use_multiscale:
                print(f"  Fusion Module: {fusion_params:,}  ← 多尺度特征融合")
            print(f"  Disentangle Head: {disentangle_params:,}")
            print(f"  Mask Head: {mask_params:,}")
            if model.use_reconstruction:
                print(f"  Reconstruction Decoder: {decoder_params:,}")
            print(f"  Total: {total_params:,}")
    else:
        # Fallback: use disentangled adapter without multiscale
        if rank == 0:
            print("\n" + "=" * 60)
            print("Using DisentangledSubtitleAdapter (without multiscale)")
            print("=" * 60)
        
        model = DisentangledSubtitleAdapter(
            backbone=config['adapter']['backbone'],
            encoder_output_dim=config['adapter']['encoder_output_dim'],
            content_dim=config['adapter'].get('content_dim', 256),
            subtitle_dim=config['adapter'].get('subtitle_dim', 256),
            use_reconstruction=config['adapter'].get('use_reconstruction', True),
            pretrained=config['adapter']['use_pretrained_backbone'],
        )
        
        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\nModel parameters: {total_params:,}")
    
    model = model.cuda()
    
    # ⭐ 创建独立的自适应权重模块（不作为模型的一部分，避免DDP问题）
    adaptive_loss_weights = None
    if config['adapter'].get('use_adaptive_loss_weights', False):
        from models.disentangled_modules import AdaptiveLossWeights
        
        # 初始权重
        init_weights = [
            config['adapter'].get('mask_loss_weight', 5.0),
            config['adapter'].get('disentangle_loss_weight', 0.1),
            config['adapter'].get('reconstruction_loss_weight', 0.1),
        ]
        
        # 转换为log_vars
        import math
        init_log_vars = [-math.log(2 * w) if w > 0 else 0.0 for w in init_weights]
        
        adaptive_loss_weights = AdaptiveLossWeights(
            num_tasks=3,
            init_log_vars=init_log_vars
        ).cuda()
        
        if rank == 0:
            print("\n✓ 自适应损失权重已启用（独立模块）")
            print(f"  初始权重: mask={init_weights[0]:.2f}, "
                  f"disentangle={init_weights[1]:.2f}, "
                  f"recon={init_weights[2]:.2f}")
            print(f"  初始log_vars: {init_log_vars}")
            print(f"  log_vars requires_grad: {adaptive_loss_weights.log_vars.requires_grad}")
            print(f"  log_vars device: {adaptive_loss_weights.log_vars.device}")
    
    # 分布式包装
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True  # ✅ 多尺度模型需要，允许部分参数未使用
        )
        
        # ⭐ 如果有自适应权重，也需要包装（但它是独立的，不会有DDP问题）
        # 注意：不需要包装，因为它很小，直接在所有GPU上复制即可
        # 但需要在optimizer中注册
    
    # ⭐ 创建视频级别批次数据集（每个样本就是一个视频的多帧batch）
    save_samples = config['data'].get('save_filter_samples', False)
    save_dir = config['checkpoint']['save_dir'] if save_samples else None
    
    train_dataset = VideoBatchDataset(
        clean_dirs=config['data']['clean_dirs'],
        subtitle_dirs=config['data']['subtitle_dirs'],
        sample_filter_file=config['data'].get('sample_filter_file'),
        target_height=config['data'].get('target_height'),
        target_width=config['data'].get('target_width'),
        augmentation_config=config['data'].get('augmentation', {}),
        frames_per_video=config['data'].get('frames_per_video', 10),
        max_video_pairs=config['data'].get('max_samples'),
        random_seed=config['data']['random_seed'],
        diff_threshold=config['adapter'].get('diff_threshold', 0.1),
        min_diff_threshold=config['adapter'].get('min_diff_threshold', 0.003),
        save_samples=save_samples,
        save_dir=save_dir,
        additional_sources=config['data'].get('additional_sources', [])
    )
    
    # ⭐ 视频级别数据集：直接按视频划分
    val_ratio = config['data'].get('val_ratio', 0.1)
    num_videos = len(train_dataset)  # 每个样本就是一个视频
    
    # 按视频划分
    num_val_videos = int(num_videos * val_ratio)
    num_train_videos = num_videos - num_val_videos
    
    # 生成索引
    train_indices = list(range(0, num_train_videos))
    val_indices = list(range(num_train_videos, num_videos))
    
    # 创建Subset
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    if rank == 0:
        print(f"\n📊 数据集划分（视频级别）:")
        print(f"  总视频数: {num_videos}")
        print(f"  训练视频: {num_train_videos} ({num_train_videos/num_videos*100:.1f}%)")
        print(f"  验证视频: {num_val_videos} ({num_val_videos/num_videos*100:.1f}%)")
        print(f"  每个视频最多采样: {config['data'].get('frames_per_video', 10)} 帧")
    
    # ⭐ 自定义collate_fn：将多个视频batch合并（如果需要）
    # 由于每个样本已经是一个视频的batch，我们可以选择：
    # 1. batch_size=1：每次处理一个视频（简单）
    # 2. batch_size>1：合并多个视频的batch（需要特殊处理）
    #
    # 推荐使用batch_size=1，因为：
    # - 每个视频的帧数可能不同（过滤后）
    # - 避免跨视频混合
    # - 梯度累积可以模拟更大的batch
    
    def video_batch_collate_fn(batch_list):
        """
        Collate function for VideoBatchDataset
        
        batch_list中的每个元素已经是一个视频的多帧batch：
        {
            'subtitle': [N1, 3, H, W],
            'clean': [N1, 3, H, W],
            'mask': [N1, 1, H, W]
        }
        
        如果batch_size=1，直接返回第一个元素即可。
        如果batch_size>1，需要将多个视频的batch拼接（注意N可能不同）。
        """
        if len(batch_list) == 1:
            # batch_size=1，直接返回
            return batch_list[0]
        else:
            # batch_size>1，拼接多个视频的batch
            # 注意：每个视频的帧数可能不同
            subtitle_list = []
            clean_list = []
            mask_list = []
            
            for batch in batch_list:
                subtitle_list.append(batch['subtitle'])
                clean_list.append(batch['clean'])
                mask_list.append(batch['mask'])
            
            # 拼接所有帧
            return {
                'subtitle': torch.cat(subtitle_list, dim=0),
                'clean': torch.cat(clean_list, dim=0),
                'mask': torch.cat(mask_list, dim=0),
            }
    
    # 创建分布式sampler
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config['data']['random_seed']
        )
        val_sampler = DistributedSampler(
            val_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=config['data']['random_seed']
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # 创建DataLoader
    # ⭐ batch_size=1：每次处理一个视频
    # ⭐ 实际的batch_size通过梯度累积来控制
    train_loader = DataLoader(
        train_subset,
        batch_size=1,  # 每次处理一个视频
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        prefetch_factor=config['training'].get('prefetch_factor', 2),
        collate_fn=video_batch_collate_fn,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=video_batch_collate_fn,
    )
    
    if rank == 0:
        print(f"\n📊 数据加载配置:")
        print(f"  总视频数: {len(train_dataset)}")
        print(f"  训练视频: {num_train_videos}")
        print(f"  验证视频: {num_val_videos}")
        print(f"  每个视频采样帧数: ~{config['data'].get('frames_per_video', 10)} 帧（过滤后可能更少）")
        print(f"\n🎯 批次策略（视频级别）:")
        print(f"  DataLoader batch_size: 1 视频/batch")
        print(f"  每个视频内部: ~{config['data'].get('frames_per_video', 10)} 帧")
        print(f"  梯度累积步数: {config['training'].get('gradient_accumulation_steps', 1)}")
        print(f"  有效batch size: ~{config['data'].get('frames_per_video', 10)} 帧 × {config['training'].get('gradient_accumulation_steps', 1)} 累积 × {world_size} GPUs")
        print(f"  优势: ✅ 同一batch来自同一视频，尺寸一致，无跨视频混合")
        
        # 打印过滤统计（如果启用了保存样本）
        if save_samples:
            print("\n等待数据集初始化完成...")
            print("注意：过滤统计将在第一个epoch后显示")
    
    # 创建损失函数（差异对比学习版本）
    criterion = DifferenceContrastiveLossWithStats(
        temperature=config['adapter']['temperature'],
        loss_type=config['adapter']['loss_type'],
        negative_sampling_ratio=config['adapter'].get('negative_sampling_ratio', 0.5)
    )
    
    # 创建优化器
    # ⭐ 如果有自适应权重，需要将其参数也加入优化器
    if adaptive_loss_weights is not None:
        # adaptive_loss_weights的学习率设置为基础学习率的10倍
        # 因为log_vars的梯度通常很小，需要更大的学习率
        adaptive_lr = config['training']['learning_rate'] * 10.0
        if rank == 0:
            print(f"  Adaptive weights learning rate: {adaptive_lr:.2e} (10x base LR)")
        
        optimizer_params = [
            {'params': model.parameters()},
            {'params': adaptive_loss_weights.parameters(), 'lr': adaptive_lr, 'weight_decay': 0.0}  # 权重不用weight decay
        ]
    else:
        optimizer_params = model.parameters()
    
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['training']['betas'],
        eps=config['training']['eps']
    )
    
    # 学习率调度器
    num_training_steps = len(train_loader) * config['training']['max_epochs']
    warmup_steps = config['training']['warmup_steps']
    lr_scheduler_type = config['training'].get('lr_scheduler', 'cosine')
    
    def lr_lambda(current_step):
        # ⭐ Warmup阶段（如果启用）
        if warmup_steps > 0 and current_step < warmup_steps:
            # 学习率线性增长，从很小的值开始
            return max(1e-6, float(current_step) / float(max(1, warmup_steps)))
        
        # ⭐ 正常训练阶段
        if lr_scheduler_type == 'constant':
            # 固定学习率（1个epoch推荐）
            return 1.0
        elif lr_scheduler_type == 'cosine':
            # Cosine衰减（多epoch推荐）
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return max(0.0, cosine_decay) * (1 - config['training']['min_lr_ratio']) + config['training']['min_lr_ratio']
        else:
            # 默认：固定学习率
            return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 梯度缩放器
    scaler = GradScaler(enabled=(config['training']['mixed_precision'] == 'bf16'))
    
    # TensorBoard
    writer = None
    if rank == 0 and config['logging']['use_tensorboard']:
        log_dir = os.path.join(
            config['logging']['log_dir'],
            config['logging']['experiment_name']
        )
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    
    # Checkpoint目录
    checkpoint_dir = config['checkpoint']['save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ========== Resume训练（如果指定） ==========
    start_epoch = 0
    best_val_loss = float('inf')
    
    # 检查是否有自适应权重（用于resume时加载）
    has_adaptive_weights = adaptive_loss_weights is not None
    
    if args.resume:
        if rank == 0:
            print("=" * 60)
            print(f"🔄 从checkpoint恢复训练: {args.resume}")
            print("=" * 60)
        
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            
            # 加载模型权重（兼容两种key名称）
            model_state = None
            if 'model' in checkpoint:
                model_state = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            
            if model_state is not None:
                # 处理DDP的module前缀
                if hasattr(model, 'module'):
                    model.module.load_state_dict(model_state, strict=False)
                else:
                    model.load_state_dict(model_state, strict=False)
                if rank == 0:
                    print("✓ 已加载模型权重")
            
            # 加载optimizer（兼容两种key名称）
            optimizer_state = None
            if 'optimizer' in checkpoint:
                optimizer_state = checkpoint['optimizer']
            elif 'optimizer_state_dict' in checkpoint:
                optimizer_state = checkpoint['optimizer_state_dict']
            
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
                if rank == 0:
                    print("✓ 已加载优化器状态")
            
            # 加载scheduler
            if 'scheduler' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
                if rank == 0:
                    print("✓ 已加载学习率调度器")
            
            # 加载scaler
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
                if rank == 0:
                    print("✓ 已加载梯度缩放器")
            
            # 加载自适应权重（如果有）
            if has_adaptive_weights and adaptive_loss_weights is not None and 'adaptive_loss_weights' in checkpoint:
                adaptive_loss_weights.load_state_dict(checkpoint['adaptive_loss_weights'])
                if rank == 0:
                    print("✓ 已加载自适应损失权重")
            
            # 加载epoch和best_val_loss
            start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个epoch开始
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            if rank == 0:
                print(f"✓ 从epoch {start_epoch}继续训练")
                print(f"✓ 最佳验证损失: {best_val_loss:.4f}")
                if 'mask_iou' in checkpoint or 'val_mask_iou' in checkpoint:
                    iou = checkpoint.get('mask_iou', checkpoint.get('val_mask_iou', 0.0))
                    print(f"✓ Mask IoU: {iou:.4f}")
                print("=" * 60)
        else:
            if rank == 0:
                print(f"⚠️  警告: checkpoint文件不存在: {args.resume}")
                print("    将从头开始训练")
                print("=" * 60)
    
    # 训练循环
    
    # 根据adapter_method选择训练函数
    adapter_method = config['adapter'].get('method', 'contrastive')
    
    for epoch in range(start_epoch, config['training']['max_epochs']):
        # 更新sampler的epoch（用于分布式训练的随机性）
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # 训练
        if adapter_method == 'disentangled':
            # 使用DisentangleAdapter训练函数
            train_stats = train_one_epoch_disentangled(
                model, train_loader, optimizer, scaler,
                epoch, config, writer, rank, checkpoint_dir,
                adaptive_loss_weights=adaptive_loss_weights,  # ⭐ 传入独立的自适应权重模块
                scheduler=scheduler  # ⭐ 传入scheduler，每步更新学习率
            )
            train_loss = train_stats['loss']
            train_acc = train_stats.get('iou', 0.0)  # 使用IoU作为accuracy
        else:
            # 使用对比学习训练函数
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler,
                epoch, config, writer, rank, checkpoint_dir
            )
            # 对比学习版本：scheduler在epoch结束后调用
            scheduler.step()
        
        if rank == 0:
            if adapter_method == 'disentangled':
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_iou={train_acc:.4f}")
            else:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
            
            # Epoch结束后打印完整过滤统计
            if save_samples:
                # 获取原始dataset（可能被Subset包装）
                if hasattr(train_subset, 'dataset'):
                    original_dataset = train_subset.dataset
                else:
                    original_dataset = train_dataset
                
                if hasattr(original_dataset, 'print_filter_stats'):
                    print("\n" + "="*80)
                    print(f"Epoch {epoch} 完整过滤统计:")
                    print("="*80)
                    original_dataset.print_filter_stats()
        
        # 验证
        if config['validation']['enabled'] and (epoch + 1) % max(1, config['validation']['val_interval'] // len(train_loader)) == 0:
            val_loss, val_mask_iou = validate(
                model, val_loader, criterion, epoch, config, writer, rank
            )
            
            if rank == 0:
                print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_mask_iou={val_mask_iou:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(checkpoint_dir, 'best_model.pt')
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_mask_iou': val_mask_iou,
                        'config': config
                    }
                    # ⭐ 保存自适应权重
                    if adaptive_loss_weights is not None:
                        checkpoint_dict['adaptive_loss_weights'] = adaptive_loss_weights.state_dict()
                    # ⭐ 保存scheduler和scaler（用于resume）
                    if scheduler is not None:
                        checkpoint_dict['scheduler'] = scheduler.state_dict()
                    if scaler is not None:
                        checkpoint_dict['scaler'] = scaler.state_dict()
                    
                    torch.save(checkpoint_dict, save_path)
                    print(f"✓ Saved best model: {save_path}")
        
        # 定期保存
        if rank == 0 and (epoch + 1) % max(1, config['checkpoint']['save_interval'] // len(train_loader)) == 0:
            save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            checkpoint_dict = {
                'epoch': epoch,
                'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config
            }
            # ⭐ 保存自适应权重
            if adaptive_loss_weights is not None:
                checkpoint_dict['adaptive_loss_weights'] = adaptive_loss_weights.state_dict()
            # ⭐ 保存scheduler和scaler（用于resume）
            if scheduler is not None:
                checkpoint_dict['scheduler'] = scheduler.state_dict()
            if scaler is not None:
                checkpoint_dict['scaler'] = scaler.state_dict()
            
            torch.save(checkpoint_dict, save_path)
            print(f"Saved checkpoint: {save_path}")
    
    # 保存最终模型
    if rank == 0:
        final_path = os.path.join(checkpoint_dir, 'final_encoder.pt')
        
        _model = model.module if hasattr(model, 'module') else model
        final_checkpoint = {
            'model': _model.state_dict(),
            'config': config
        }
        
        # 保存mask_head（如果有）
        if hasattr(_model, 'mask_head'):
            final_checkpoint['mask_head'] = _model.mask_head.state_dict()
            print(f"✓ Mask prediction head included in checkpoint")
        
        # ⭐ 保存最终的自适应权重（用于分析）
        if adaptive_loss_weights is not None:
            final_checkpoint['adaptive_loss_weights'] = adaptive_loss_weights.state_dict()
            final_weights = adaptive_loss_weights.get_weights()
            print(f"✓ Final adaptive weights: mask={final_weights[0]:.3f}, "
                  f"disentangle={final_weights[1]:.3f}, recon={final_weights[2]:.3f}")
        
        torch.save(final_checkpoint, final_path)
        print(f"\n✓ Saved final encoder: {final_path}")
        print("\n" + "=" * 60)
        print("🎉 Stage 1训练完成！")
        print("=" * 60)
    
    if writer:
        writer.close()
    
    cleanup_distributed()


if __name__ == '__main__':
    main()

