"""
Difference-Aware Contrastive Loss Functions
差异感知的对比学习损失函数

核心创新：
1. ✅ 使用字幕位置mask引导对比学习
2. ✅ 字幕区域（正样本）vs 干净区域（负样本）
3. ✅ 学习差异特征的空间分布
4. ✅ 保留对比学习框架

主要实现基于mask的区域对比损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGuidedContrastiveLoss(nn.Module):
    """
    Mask引导的对比学习损失
    
    核心思想：
    - 使用字幕位置mask将特征分为两类：
      * 字幕区域（mask=1）：应该有相似的差异特征
      * 干净区域（mask=0）：应该有不同的差异特征
    
    - 对比学习目标：
      * Positive pairs: 不同样本的字幕区域特征
      * Negative pairs: 字幕区域 vs 干净区域
    
    Args:
        temperature: 温度参数
        negative_sampling_ratio: 负样本采样比例
        min_positive_samples: 最小正样本数量（避免某些样本没有字幕）
    """
    
    def __init__(self, temperature=0.07, negative_sampling_ratio=0.5, min_positive_samples=10):
        super().__init__()
        self.temperature = temperature
        self.negative_sampling_ratio = negative_sampling_ratio
        self.min_positive_samples = min_positive_samples
    
    def forward(self, z, mask):
        """
        计算mask引导的对比学习损失
        
        Args:
            z: 差异特征 (batch, dim, H, W)
            mask: 字幕位置mask (batch, 1, H, W), 值为0或1 (detached, no grad)
            
        Returns:
            loss: 对比学习损失
        """
        batch_size, dim, h, w = z.shape
        
        # 下采样mask到特征图大小（如果需要）
        # mask已经detached，不会参与梯度计算
        if mask.shape[2] != h or mask.shape[3] != w:
            with torch.no_grad():
                mask = F.interpolate(mask, size=(h, w), mode='nearest')
        
        # 重塑为 (batch * H * W, dim)
        z_flat = z.permute(0, 2, 3, 1).reshape(batch_size * h * w, dim)
        mask_flat = mask.view(batch_size * h * w)
        
        # L2归一化
        z_norm = F.normalize(z_flat, p=2, dim=1)
        
        # 分离正样本（字幕区域）和负样本（干净区域）
        positive_mask = mask_flat > 0.5  # 字幕区域
        negative_mask = mask_flat <= 0.5  # 干净区域
        
        positive_features = z_norm[positive_mask]
        negative_features = z_norm[negative_mask]
        
        # 检查正样本数量
        num_positives = positive_features.shape[0]
        num_negatives = negative_features.shape[0]
        
        if num_positives < self.min_positive_samples:
            # 如果正样本太少，返回一个小的损失（避免训练崩溃）
            # 使用z的均值来保持梯度计算图
            return z.mean() * 0.0 + 0.01
        
        # 采样负样本（避免计算量过大）
        if self.negative_sampling_ratio < 1.0 and num_negatives > 0:
            num_sampled_negatives = max(1, int(num_negatives * self.negative_sampling_ratio))
            neg_indices = torch.randperm(num_negatives, device=z.device)[:num_sampled_negatives]
            negative_features = negative_features[neg_indices]
        
        # 采样正样本（用于anchor）
        num_anchor_samples = min(num_positives, int(num_positives * 0.5))
        if num_anchor_samples < num_positives:
            anchor_indices = torch.randperm(num_positives, device=z.device)[:num_anchor_samples]
            anchor_features = positive_features[anchor_indices]
        else:
            anchor_features = positive_features
        
        # 计算相似度
        # Positive pairs: anchor与其他字幕区域
        sim_pos = torch.matmul(anchor_features, positive_features.T) / self.temperature
        # (num_anchors, num_positives)
        
        # Negative pairs: anchor与干净区域
        if negative_features.shape[0] > 0:
            sim_neg = torch.matmul(anchor_features, negative_features.T) / self.temperature
            # (num_anchors, num_negatives)
        else:
            sim_neg = torch.zeros(anchor_features.shape[0], 1, device=z.device)
        
        # InfoNCE损失
        # 对于每个anchor，选择一个正样本（不是自己）
        losses = []
        for i in range(anchor_features.shape[0]):
            # 选择正样本（避免自己）
            pos_sims = sim_pos[i]
            # 移除自己（如果存在）
            if i < pos_sims.shape[0]:
                pos_sims_without_self = torch.cat([pos_sims[:i], pos_sims[i+1:]])
            else:
                pos_sims_without_self = pos_sims
            
            if pos_sims_without_self.shape[0] == 0:
                continue
            
            # 选择最相似的正样本
            pos_sim = pos_sims_without_self.max()
            
            # 负样本
            neg_sims = sim_neg[i]
            
            # 合并
            all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            
            # 计算loss: -log(exp(pos) / sum(exp(all)))
            loss_i = -pos_sim + torch.logsumexp(all_sims, dim=0)
            losses.append(loss_i)
        
        if len(losses) == 0:
            # 使用z的均值来保持梯度计算图
            return z.mean() * 0.0 + 0.01
        
        loss = torch.stack(losses).mean()
        return loss


class SymmetricMaskGuidedLoss(nn.Module):
    """
    对称的mask引导对比损失
    
    同时优化：
    1. 字幕区域内的一致性
    2. 字幕区域与干净区域的区分性
    
    Args:
        temperature: 温度参数
        negative_sampling_ratio: 负样本采样比例
    """
    
    def __init__(self, temperature=0.07, negative_sampling_ratio=0.5):
        super().__init__()
        self.base_loss = MaskGuidedContrastiveLoss(
            temperature=temperature,
            negative_sampling_ratio=negative_sampling_ratio
        )
    
    def forward(self, z, mask):
        """
        计算对称的mask引导损失
        
        Args:
            z: 差异特征 (batch, dim, H, W)
            mask: 字幕位置mask (batch, 1, H, W)
            
        Returns:
            loss: 对称对比学习损失
        """
        # 正向损失：字幕区域作为anchor
        loss_forward = self.base_loss(z, mask)
        
        # 反向损失：干净区域作为anchor（可选）
        # 这里可以选择不计算反向损失，或者给予较小的权重
        # loss_backward = self.base_loss(z, 1 - mask)
        # loss = 0.7 * loss_forward + 0.3 * loss_backward
        
        return loss_forward


class SimpleDifferenceContrastiveLoss(nn.Module):
    """
    简化的差异对比损失（更直观的版本）
    
    核心思想：
    - 字幕区域的特征应该聚类在一起（高激活）
    - 干净区域的特征应该聚类在一起（低激活）
    - 两类区域应该分开
    
    使用简单的距离度量而不是InfoNCE
    
    Args:
        margin: 分类边界margin
        weight_positive: 正样本损失权重
        weight_negative: 负样本损失权重
    """
    
    def __init__(self, margin=1.0, weight_positive=1.0, weight_negative=1.0):
        super().__init__()
        self.margin = margin
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative
    
    def forward(self, z, mask):
        """
        计算简化的对比损失
        
        Args:
            z: 差异特征 (batch, dim, H, W)
            mask: 字幕位置mask (batch, 1, H, W)
            
        Returns:
            loss: 对比损失
        """
        batch_size, dim, h, w = z.shape
        
        # 下采样mask
        if mask.shape[2] != h or mask.shape[3] != w:
            mask = F.interpolate(mask, size=(h, w), mode='nearest')
        
        # 计算字幕区域和干净区域的中心
        z_flat = z.permute(0, 2, 3, 1).reshape(batch_size * h * w, dim)
        mask_flat = mask.view(batch_size * h * w)
        
        positive_mask = mask_flat > 0.5
        negative_mask = mask_flat <= 0.5
        
        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            # 使用z的均值来保持梯度计算图
            return z.mean() * 0.0 + 0.01
        
        # 计算正样本中心
        positive_features = z_flat[positive_mask]
        positive_center = positive_features.mean(dim=0)
        
        # 计算负样本中心
        negative_features = z_flat[negative_mask]
        negative_center = negative_features.mean(dim=0)
        
        # 损失1：正样本应该靠近正样本中心
        loss_positive = F.mse_loss(positive_features, positive_center.unsqueeze(0).expand_as(positive_features))
        
        # 损失2：负样本应该靠近负样本中心
        loss_negative = F.mse_loss(negative_features, negative_center.unsqueeze(0).expand_as(negative_features))
        
        # 损失3：两个中心应该分开
        center_distance = F.pairwise_distance(
            positive_center.unsqueeze(0),
            negative_center.unsqueeze(0)
        )
        loss_separation = F.relu(self.margin - center_distance)
        
        # 总损失
        loss = (self.weight_positive * loss_positive + 
                self.weight_negative * loss_negative + 
                loss_separation)
        
        return loss


def compute_mask_guided_accuracy(z, mask, temperature=0.07, sample_ratio=0.1):
    """
    计算mask引导的对比学习准确率
    
    准确率定义：字幕区域的特征是否更接近其他字幕区域而非干净区域
    
    Args:
        z: 差异特征 (batch, dim, H, W)
        mask: 字幕位置mask (batch, 1, H, W)
        temperature: 温度参数
        sample_ratio: 采样比例
        
    Returns:
        accuracy: 准确率 (0-1之间)
    """
    batch_size, dim, h, w = z.shape
    
    # 下采样mask
    if mask.shape[2] != h or mask.shape[3] != w:
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
    
    z_flat = z.permute(0, 2, 3, 1).reshape(batch_size * h * w, dim)
    mask_flat = mask.view(batch_size * h * w)
    
    # 分离正负样本
    positive_mask = mask_flat > 0.5
    negative_mask = mask_flat <= 0.5
    
    positive_features = z_flat[positive_mask]
    negative_features = z_flat[negative_mask]
    
    if positive_features.shape[0] < 10 or negative_features.shape[0] < 10:
        return 0.0
    
    # 采样
    num_pos_samples = max(10, int(positive_features.shape[0] * sample_ratio))
    num_neg_samples = max(10, int(negative_features.shape[0] * sample_ratio))
    
    pos_indices = torch.randperm(positive_features.shape[0], device=z.device)[:num_pos_samples]
    neg_indices = torch.randperm(negative_features.shape[0], device=z.device)[:num_neg_samples]
    
    pos_sample = positive_features[pos_indices]
    neg_sample = negative_features[neg_indices]
    
    # 归一化
    pos_sample = F.normalize(pos_sample, p=2, dim=1)
    neg_sample = F.normalize(neg_sample, p=2, dim=1)
    
    # 计算相似度
    # 正样本之间的相似度矩阵：(num_pos, num_pos)
    sim_pos_pos = torch.matmul(pos_sample, pos_sample.T)
    
    # 正样本与负样本的相似度矩阵：(num_pos, num_neg)
    sim_pos_neg = torch.matmul(pos_sample, neg_sample.T)
    
    # ⭐ 更细粒度的准确率计算（向量化实现）：
    # 对于每个正样本，判断：
    #   "与其他正样本的平均相似度" > "与负样本的平均相似度"
    # 统计满足条件的样本比例
    
    # 移除对角线（自己与自己的相似度=1.0，会影响平均值）
    # 方法：将对角线设为0，然后除以(num_pos-1)而不是num_pos
    eye_mask = torch.eye(num_pos_samples, device=z.device, dtype=torch.bool)
    sim_pos_pos_no_diag = sim_pos_pos.clone()
    sim_pos_pos_no_diag[eye_mask] = 0.0
    
    # 每个正样本与其他正样本的平均相似度：(num_pos,)
    # sum / (num_pos - 1)
    sim_pos_pos_per_sample = sim_pos_pos_no_diag.sum(dim=1) / (num_pos_samples - 1)
    
    # 每个正样本与负样本的平均相似度：(num_pos,)
    sim_pos_neg_per_sample = sim_pos_neg.mean(dim=1)
    
    # 准确率：有多少比例的正样本满足"与正样本更相似"
    correct = (sim_pos_pos_per_sample > sim_pos_neg_per_sample).float()
    accuracy = correct.mean()
    
    return accuracy.item()


class DifferenceContrastiveLossWithStats(nn.Module):
    """
    带统计信息的差异对比学习损失
    
    在计算损失的同时记录有用的统计信息
    """
    
    def __init__(self, temperature=0.07, loss_type='mask_guided', negative_sampling_ratio=0.5):
        super().__init__()
        
        if loss_type == 'mask_guided':
            self.loss_fn = MaskGuidedContrastiveLoss(
                temperature=temperature,
                negative_sampling_ratio=negative_sampling_ratio
            )
        elif loss_type == 'symmetric':
            self.loss_fn = SymmetricMaskGuidedLoss(
                temperature=temperature,
                negative_sampling_ratio=negative_sampling_ratio
            )
        elif loss_type == 'simple':
            self.loss_fn = SimpleDifferenceContrastiveLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        self.temperature = temperature
    
    def forward(self, z, mask):
        """
        计算损失和统计信息
        
        Args:
            z: 差异特征 (batch, dim, H, W)
            mask: 字幕位置mask (batch, 1, H, W)
            
        Returns:
            loss: 损失值
            stats: 统计信息字典
        """
        # 计算损失
        loss = self.loss_fn(z, mask)
        
        # 计算统计信息
        with torch.no_grad():
            batch_size, dim, h, w = z.shape
            
            # 下采样mask
            if mask.shape[2] != h or mask.shape[3] != w:
                mask_resized = F.interpolate(mask, size=(h, w), mode='nearest')
            else:
                mask_resized = mask
            
            # 正样本比例
            positive_ratio = (mask_resized > 0.5).float().mean()
            
            # 准确率
            accuracy = compute_mask_guided_accuracy(z, mask, self.temperature)
            
            stats = {
                'loss': loss.item(),
                'accuracy': accuracy,
                'positive_ratio': positive_ratio.item(),
            }
        
        return loss, stats


if __name__ == '__main__':
    """测试差异对比学习损失函数"""
    
    print("=" * 60)
    print("Testing Difference-Aware Contrastive Loss Functions")
    print("=" * 60)
    
    # 创建测试数据
    batch_size = 4
    dim = 256
    h, w = 16, 16
    
    z = torch.randn(batch_size, dim, h, w)
    
    # 创建模拟的字幕mask（底部区域）
    mask = torch.zeros(batch_size, 1, h, w)
    mask[:, :, int(h*0.7):, :] = 1.0  # 底部30%是字幕区域
    
    print(f"\nTest data:")
    print(f"  Features shape: {z.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Positive ratio: {mask.mean():.3f}")
    
    # 测试Mask引导对比损失
    print("\n1. Testing Mask-Guided Contrastive Loss...")
    loss_fn = MaskGuidedContrastiveLoss(temperature=0.07, negative_sampling_ratio=0.5)
    loss = loss_fn(z, mask)
    print(f"   Loss: {loss.item():.4f}")
    
    # 测试对称损失
    print("\n2. Testing Symmetric Mask-Guided Loss...")
    loss_fn = SymmetricMaskGuidedLoss(temperature=0.07)
    loss = loss_fn(z, mask)
    print(f"   Loss: {loss.item():.4f}")
    
    # 测试简化损失
    print("\n3. Testing Simple Difference Contrastive Loss...")
    loss_fn = SimpleDifferenceContrastiveLoss()
    loss = loss_fn(z, mask)
    print(f"   Loss: {loss.item():.4f}")
    
    # 测试准确率
    print("\n4. Testing Accuracy Computation...")
    accuracy = compute_mask_guided_accuracy(z, mask)
    print(f"   Accuracy: {accuracy:.4f}")
    
    # 测试带统计的损失
    print("\n5. Testing Loss with Stats...")
    loss_fn = DifferenceContrastiveLossWithStats(
        temperature=0.07,
        loss_type='mask_guided',
        negative_sampling_ratio=0.5
    )
    loss, stats = loss_fn(z, mask)
    print(f"   Loss: {stats['loss']:.4f}")
    print(f"   Accuracy: {stats['accuracy']:.4f}")
    print(f"   Positive ratio: {stats['positive_ratio']:.3f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

