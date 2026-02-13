"""
DisentangleAdapter - 解耦式字幕特征学习

基于解耦表示学习（Disentangled Representation Learning）
核心思想：将内容特征和字幕特征分离，比对比学习更适合文本去除任务

主要组件：
1. DisentangleHead: 解耦模块，分离内容和字幕特征
2. LightweightDecoder: 轻量级解码器，重建clean frame
3. MaskHead: Mask预测头
4. DisentangledSubtitleAdapter: 完整模型

损失函数：
1. Mask Loss: Focal + Dice + IoU (处理类别不平衡)
2. Disentangle Loss: 正交约束 (确保特征独立)
3. Reconstruction Loss: L1 (确保内容特征质量)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import sigmoid_focal_loss


class AdaptiveLossWeights(nn.Module):
    """
    自适应损失权重学习模块
    
    基于论文: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    (Kendall et al., CVPR 2018)
    
    核心思想：
    - 为每个任务学习一个log variance参数: log(σ²)
    - 权重自动计算为: 1 / (2σ²)
    - 损失加上正则项: log(σ) 防止σ→∞
    
    优势：
    - 自动平衡多任务
    - 有理论基础（贝叶斯推断）
    - 防止某个任务主导训练
    """
    
    def __init__(self, num_tasks=3, init_log_vars=None):
        """
        Args:
            num_tasks: 任务数量（mask, disentangle, reconstruction）
            init_log_vars: 初始化log(σ²)值，如果为None则初始化为0（即σ²=1）
        """
        super().__init__()
        
        # 可学习的log variance参数
        if init_log_vars is None:
            # 默认初始化：所有任务权重相等
            init_log_vars = torch.zeros(num_tasks)
        else:
            init_log_vars = torch.tensor(init_log_vars, dtype=torch.float32)
        
        # 注册为可学习参数
        self.log_vars = nn.Parameter(init_log_vars)
        self.num_tasks = num_tasks
    
    def forward(self, losses):
        """
        计算加权后的总损失
        
        Args:
            losses: list of loss tensors [mask_loss, disentangle_loss, recon_loss]
        
        Returns:
            weighted_loss: 加权总损失
            weights: 当前的权重值（用于监控）
            precisions: 当前的精度值 1/σ²（用于监控）
        """
        assert len(losses) == self.num_tasks, f"Expected {self.num_tasks} losses, got {len(losses)}"
        
        # 计算权重和正则项
        weighted_loss = 0.0
        weights = []
        precisions = []
        
        for i, loss in enumerate(losses):
            # log_var = log(σ²)
            log_var = self.log_vars[i]
            
            # precision = 1 / σ² = exp(-log_var)
            precision = torch.exp(-log_var)
            
            # 加权损失: loss / (2σ²) + log(σ) / 2
            # = loss * precision / 2 + log_var / 2
            weighted_loss += 0.5 * precision * loss + 0.5 * log_var
            
            weights.append(precision.item())
            precisions.append(precision.item())
        
        return weighted_loss, weights, precisions
    
    def get_weights(self):
        """获取当前权重（用于日志记录）"""
        with torch.no_grad():
            precisions = torch.exp(-self.log_vars)
            return precisions.cpu().numpy()
    
    def get_log_vars(self):
        """获取log variance值（用于监控）"""
        return self.log_vars.detach().cpu().numpy()


class DisentangleHead(nn.Module):
    """解耦模块：分离内容特征和字幕特征"""
    
    def __init__(self, input_dim=512, content_dim=256, subtitle_dim=256):
        super().__init__()
        
        # 内容分支
        self.content_branch = nn.Sequential(
            nn.Conv2d(input_dim, content_dim*2, 1),
            nn.GroupNorm(32, content_dim*2),
            nn.ReLU(inplace=False),
            nn.Conv2d(content_dim*2, content_dim, 1),
        )
        
        # 字幕分支
        self.subtitle_branch = nn.Sequential(
            nn.Conv2d(input_dim, subtitle_dim*2, 1),
            nn.GroupNorm(32, subtitle_dim*2),
            nn.ReLU(inplace=False),
            nn.Conv2d(subtitle_dim*2, subtitle_dim, 1),
        )
    
    def forward(self, mixed_feat):
        """
        Args:
            mixed_feat: (B, 512, H, W)
        Returns:
            content_feat: (B, 256, H, W)
            subtitle_feat: (B, 256, H, W)
        """
        content_feat = self.content_branch(mixed_feat)
        subtitle_feat = self.subtitle_branch(mixed_feat)
        return content_feat, subtitle_feat


class LightweightDecoder(nn.Module):
    """轻量级解码器：用于重建clean frame"""
    
    def __init__(self, input_dim=256, output_channels=3):
        super().__init__()
        
        # 上采样路径
        self.decoder = nn.Sequential(
            # 上采样到2x
            nn.ConvTranspose2d(input_dim, 128, 4, 2, 1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=False),
            
            # 上采样到4x
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=False),
            
            # 上采样到8x
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=False),
            
            # 上采样到16x
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=False),
            
            # 输出
            nn.Conv2d(16, output_channels, 3, 1, 1),
            nn.Tanh(),  # 输出范围[-1, 1]
        )
    
    def forward(self, content_feat):
        """
        Args:
            content_feat: (B, 256, H/16, W/16)
        Returns:
            reconstructed: (B, 3, H, W)
        """
        return self.decoder(content_feat)


class MaskHead(nn.Module):
    """Mask预测头"""
    
    def __init__(self, input_dim=256, upsample_factor=16):
        super().__init__()
        
        # 逐步上采样
        layers = []
        current_dim = input_dim
        
        # 计算需要几次上采样 (2^n = upsample_factor)
        num_ups = int(torch.log2(torch.tensor(float(upsample_factor))))
        
        for i in range(num_ups):
            out_dim = max(current_dim // 2, 16)  # 最少16个通道
            layers.extend([
                nn.ConvTranspose2d(current_dim, out_dim, 4, 2, 1),
                nn.GroupNorm(min(8, out_dim // 2), out_dim),
                nn.ReLU(inplace=False),
            ])
            current_dim = out_dim
        
        # 最后输出1通道mask
        layers.append(nn.Conv2d(current_dim, 1, 3, 1, 1))
        
        self.mask_predictor = nn.Sequential(*layers)
    
    def forward(self, subtitle_feat, target_size=None):
        """
        Args:
            subtitle_feat: (B, 256, H', W')
            target_size: (H, W) 目标输出尺寸（可选）
        Returns:
            mask_logits: (B, 1, H, W)
        """
        mask = self.mask_predictor(subtitle_feat)
        
        # 如果指定了目标尺寸，resize到目标尺寸
        if target_size is not None:
            mask = F.interpolate(
                mask, size=target_size, 
                mode='bilinear', align_corners=False
            )
        
        return mask


class DisentangledSubtitleAdapter(nn.Module):
    """
    解耦式字幕特征学习
    
    核心思想：
    1. 输入subtitle frame
    2. 提取混合特征
    3. 解耦为内容特征和字幕特征
    4. 多任务学习：mask预测 + 解耦约束 + 重建
    """
    
    def __init__(
        self,
        backbone='resnet50',
        encoder_output_dim=512,
        content_dim=256,
        subtitle_dim=256,
        use_reconstruction=True,
        pretrained=True,
        backbone_weight_path=None,  # 自定义权重路径
    ):
        super().__init__()
        
        self.use_reconstruction = use_reconstruction
        
        # Encoder (Backbone)
        if backbone == 'resnet50':
            # 如果提供了自定义权重路径，从该路径加载
            if backbone_weight_path and os.path.exists(backbone_weight_path):
                if torch.distributed.is_initialized():
                    # 只在rank 0打印，避免重复输出
                    if torch.distributed.get_rank() == 0:
                        print(f"Loading ResNet50 weights from: {backbone_weight_path}")
                else:
                    print(f"Loading ResNet50 weights from: {backbone_weight_path}")
                
                base_model = models.resnet50(pretrained=False)  # 先创建模型
                # 加载自定义权重
                state_dict = torch.load(backbone_weight_path, map_location='cpu')
                
                # 处理权重键名（可能需要移除 'module.' 前缀）
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                # 加载权重（strict=False允许fc层等不匹配）
                missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)
                
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        print(f"✓ Successfully loaded ResNet50 weights from custom path")
                        if missing_keys:
                            print(f"  Missing keys (ignored): {len(missing_keys)}")
                        if unexpected_keys:
                            print(f"  Unexpected keys (ignored): {len(unexpected_keys)}")
                else:
                    print(f"✓ Successfully loaded ResNet50 weights from custom path")
                    if missing_keys:
                        print(f"  Missing keys (ignored): {len(missing_keys)}")
                    if unexpected_keys:
                        print(f"  Unexpected keys (ignored): {len(unexpected_keys)}")
            elif pretrained:
                # 使用默认的预训练权重（从torch hub下载）
                base_model = models.resnet50(pretrained=True)
            else:
                # 不使用预训练权重
                base_model = models.resnet50(pretrained=False)
            backbone_dim = 2048
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        elif backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 移除最后的FC和全局池化
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        
        # 投影到统一维度
        self.feature_projection = nn.Sequential(
            nn.Conv2d(backbone_dim, encoder_output_dim, 1, bias=False),
            nn.GroupNorm(32, encoder_output_dim),
            nn.ReLU(inplace=False),
        )
        
        # 解耦模块（核心）
        self.disentangle_head = DisentangleHead(
            input_dim=encoder_output_dim,
            content_dim=content_dim,
            subtitle_dim=subtitle_dim,
        )
        
        # Mask预测头
        self.mask_head = MaskHead(
            input_dim=subtitle_dim,
            upsample_factor=16,  # ResNet下采样16倍
        )
        
        # 重建解码器（可选）
        if use_reconstruction:
            self.decoder = LightweightDecoder(input_dim=content_dim)
    
    def encode(self, subtitle_frame):
        """
        提取并解耦特征
        
        Args:
            subtitle_frame: (B, 3, H, W)
        Returns:
            content_feat: (B, 256, H/16, W/16)
            subtitle_feat: (B, 256, H/16, W/16)
        """
        # Backbone特征提取
        feat = self.backbone(subtitle_frame)  # (B, backbone_dim, H/16, W/16)
        
        # 投影
        mixed_feat = self.feature_projection(feat)  # (B, 512, H/16, W/16)
        
        # 解耦
        content_feat, subtitle_feat = self.disentangle_head(mixed_feat)
        
        return content_feat, subtitle_feat
    
    def forward(self, subtitle_frame, return_all=True):
        """
        完整前向传播
        
        Args:
            subtitle_frame: (B, 3, H, W)
            return_all: 是否返回所有输出（训练时）
        
        Returns:
            如果return_all=True:
                (content_feat, subtitle_feat, mask_logits, recon_clean)
            否则:
                subtitle_feat (用于推理)
        """
        # 记录输入尺寸
        target_size = (subtitle_frame.shape[2], subtitle_frame.shape[3])
        
        # 解耦特征
        content_feat, subtitle_feat = self.encode(subtitle_frame)
        
        if not return_all:
            return subtitle_feat
        
        # Mask预测（resize到原始尺寸）
        mask_logits = self.mask_head(subtitle_feat, target_size=target_size)
        
        # 重建clean（可选，resize到原始尺寸）
        recon_clean = None
        if self.use_reconstruction:
            recon_clean = self.decoder(content_feat)
            # Resize到原始尺寸
            if recon_clean.shape[2:] != target_size:
                recon_clean = F.interpolate(
                    recon_clean, size=target_size,
                    mode='bilinear', align_corners=False
                )
        
        return content_feat, subtitle_feat, mask_logits, recon_clean


# ============ 损失函数 ============

def compute_mask_loss(pred_logits, gt_mask):
    """
    Mask预测损失：Focal + Dice + IoU + 高权重正样本
    
    Args:
        pred_logits: (B, 1, H, W) - 预测的mask logits
        gt_mask: (B, 1, H, W) - GT mask (0或1)
    
    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的字典
    """
    # ⭐ 策略1: 平衡的Focal Loss (针对1%字幕占比)
    # alpha=0.85意味着正样本权重约5.7倍于负样本（0.85/0.15）
    focal = sigmoid_focal_loss(
        pred_logits, gt_mask,
        alpha=0.85,  # ✅ 降低到0.85，避免过度预测
        gamma=2.0,   # ✅ 保持2.0（标准的难例挖掘）
        reduction='mean'
    )
    
    # Sigmoid for Dice and IoU
    pred_prob = torch.sigmoid(pred_logits)
    
    # ⭐ 策略2: 正样本加权的Dice Loss（修正版）
    smooth = 1e-5
    pred_flat = pred_prob.view(-1)
    target_flat = gt_mask.view(-1)
    
    # 给正样本区域更高的权重（5倍，避免过度预测）
    pos_mask = (target_flat > 0.5).float()
    pos_weight = 5.0  # ✅ 降低到5，平衡召回和精度
    sample_weight = torch.where(pos_mask > 0.5, pos_weight, 1.0)
    
    # ✅ 修正：使用权重作为sample importance，而不是直接乘到pred/target上
    # 这样可以避免分子分母缩放不一致的问题
    weighted_intersection = ((pred_flat * target_flat) * sample_weight).sum()
    weighted_pred_sum = (pred_flat * sample_weight).sum()
    weighted_target_sum = (target_flat * sample_weight).sum()
    
    dice = (2. * weighted_intersection + smooth) / (
        weighted_pred_sum + weighted_target_sum + smooth
    )
    dice_loss = 1 - dice
    
    # ⭐ 策略3: 正样本加权的IoU Loss（修正版）
    weighted_union = weighted_pred_sum + weighted_target_sum - weighted_intersection
    iou = (weighted_intersection + smooth) / (weighted_union + smooth)
    iou_loss = 1 - iou
    
    # ⭐ 策略4: 添加正样本召回损失（确保不会全预测0）
    # 如果GT有正样本，但预测全是负样本，这个损失会很高
    gt_pos_pixels = gt_mask.sum()
    if gt_pos_pixels > 0:
        pred_pos_pixels = (pred_prob > 0.5).float().sum()
        recall_penalty = torch.abs(pred_pos_pixels - gt_pos_pixels) / (gt_pos_pixels + 1e-5)
        recall_penalty = torch.clamp(recall_penalty, 0, 2.0)  # 限制在[0, 2]
    else:
        recall_penalty = torch.tensor(0.0, device=pred_logits.device)
    
    # ⭐ 策略5: 平衡的损失权重组合
    # Focal Loss权重最高（2.0），对类别不平衡敏感
    # Dice Loss次之（1.5），保证重叠
    # IoU Loss中等（1.0），优化评估指标
    # Recall Penalty最低（0.1），轻微辅助防止全预测0（降低避免过度预测）
    total = 2.0 * focal + 1.5 * dice_loss + 1.0 * iou_loss + 0.1 * recall_penalty
    
    return total, {
        'focal': focal.item(),
        'dice': dice_loss.item(),
        'iou_loss': iou_loss.item(),
        'recall_penalty': recall_penalty.item(),
        'iou': iou.item(),  # 真实IoU值（用于监控）
        'pred_pos_ratio': (pred_prob > 0.5).float().mean().item(),  # 预测正样本比例
        'gt_pos_ratio': gt_mask.mean().item(),  # GT正样本比例
    }


def compute_disentangle_loss(content_feat, subtitle_feat):
    """
    解耦损失：确保内容特征和字幕特征正交（独立）
    
    Args:
        content_feat: (B, C, H, W) - 内容特征
        subtitle_feat: (B, C, H, W) - 字幕特征
    
    Returns:
        loss: 解耦损失（相似度的绝对值，应该接近0）
    """
    B, C, H, W = content_feat.shape
    
    # Flatten spatial dimensions
    content_flat = content_feat.view(B, C, -1)  # (B, C, H*W)
    subtitle_flat = subtitle_feat.view(B, C, -1)
    
    # 归一化
    content_norm = F.normalize(content_flat, p=2, dim=1)
    subtitle_norm = F.normalize(subtitle_flat, p=2, dim=1)
    
    # 余弦相似度（应该接近0）
    # (B, C, H*W) * (B, C, H*W) -> (B, H*W)
    similarity = (content_norm * subtitle_norm).sum(dim=1).abs()
    
    # 平均相似度
    loss = similarity.mean()
    
    return loss


def compute_reconstruction_loss(pred_clean, gt_clean):
    """
    重建损失：L1
    
    Args:
        pred_clean: (B, 3, H, W) - 重建的clean frame
        gt_clean: (B, 3, H, W) - GT clean frame
    
    Returns:
        loss: L1损失
    """
    return F.l1_loss(pred_clean, gt_clean)


# ============ 训练辅助函数 ============

def compute_mask_metrics(pred_logits, gt_mask):
    """
    计算mask预测的评估指标
    
    Args:
        pred_logits: (B, 1, H, W)
        gt_mask: (B, 1, H, W)
    
    Returns:
        metrics: dict
    """
    pred_prob = torch.sigmoid(pred_logits)
    pred_binary = (pred_prob > 0.5).float()
    
    # IoU
    intersection = (pred_binary * gt_mask).sum()
    union = pred_binary.sum() + gt_mask.sum() - intersection
    iou = (intersection + 1e-5) / (union + 1e-5)
    
    # 准确率
    correct = (pred_binary == gt_mask).float().sum()
    total = gt_mask.numel()
    acc = correct / total
    
    # 正样本比例
    pred_pos = pred_binary.mean()
    gt_pos = gt_mask.mean()
    
    return {
        'iou': iou.item(),
        'acc': acc.item(),
        'pred_pos': pred_pos.item(),
        'gt_pos': gt_pos.item(),
    }

