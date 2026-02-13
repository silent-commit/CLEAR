"""
CLEAR Stage I: Dual Encoder with Multi-Scale Feature Fusion

Implements the disentangled feature learning framework (Section 3.2):
- Dual ResNet-50 encoders: E_sub (subtitle) and E_content (content)
- Multi-scale FPN fusion for small subtitle detection
- Orthogonality constraint for feature independence
- Binary mask prediction from subtitle features F^sub

Architecture:
    Input → ResNet-50 → [layer1, layer2, layer3, layer4] → FPN Fusion → 
    → DisentangleHead → (F^sub, F^content) → MaskHead → M^prior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .disentangled_modules import (
    DisentangleHead, 
    LightweightDecoder, 
    MaskHead,
    AdaptiveLossWeights,
    compute_mask_loss,
    compute_disentangle_loss,
    compute_reconstruction_loss,
)


class MultiscaleFusionModule(nn.Module):
    """多尺度特征融合模块（FPN-like）"""
    
    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: 各层的输入通道数 [C1, C2, C3, C4]
                             例如ResNet50: [256, 512, 1024, 2048]
            out_channels: 统一的输出通道数
        """
        super().__init__()
        
        # 1x1卷积将各层通道数统一到out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False)
            for in_ch in in_channels_list
        ])
        
        # 3x3卷积平滑融合后的特征
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=False),
            )
            for _ in in_channels_list
        ])
    
    def forward(self, features):
        """
        Args:
            features: list of (B, Ci, Hi, Wi) - 从浅到深
                     [feat_layer1, feat_layer2, feat_layer3, feat_layer4]
        Returns:
            fused_features: list of (B, out_channels, Hi, Wi)
        """
        # Step 1: 1x1卷积统一通道数
        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, features)
        ]
        
        # Step 2: 自上而下融合（从深层到浅层）
        fused = []
        # 最深层直接使用
        fused.append(self.smooth_convs[-1](laterals[-1]))
        
        # 逐层向上融合
        for i in range(len(laterals) - 2, -1, -1):
            # 上采样深层特征
            upsampled = F.interpolate(
                fused[0], 
                size=laterals[i].shape[2:],
                mode='bilinear', 
                align_corners=False
            )
            # 与当前层相加
            merged = laterals[i] + upsampled
            # 平滑
            smoothed = self.smooth_convs[i](merged)
            fused.insert(0, smoothed)
        
        return fused


class MultiscaleDisentangledAdapter(nn.Module):
    """
    多尺度解耦Adapter
    
    改进：
    1. 提取ResNet的多层特征
    2. FPN-like融合
    3. 在融合特征上做mask预测（利用浅层高分辨率信息）
    """
    
    def __init__(
        self,
        backbone='resnet50',
        encoder_output_dim=512,
        content_dim=256,
        subtitle_dim=256,
        use_reconstruction=True,
        pretrained=True,
        backbone_weight_path=None,
        use_multiscale=True,  # 是否使用多尺度融合
        fusion_layer='layer2',  # 使用哪一层融合特征: 'layer1'(4x), 'layer2'(8x), 'layer1+2'(融合)
        use_adaptive_loss_weights=True,  # 是否使用自适应损失权重
        init_loss_weights=None,  # 初始损失权重 [mask, disentangle, recon]
    ):
        super().__init__()
        
        self.use_reconstruction = use_reconstruction
        self.use_multiscale = use_multiscale
        self.fusion_layer = fusion_layer
        self.use_adaptive_loss_weights = use_adaptive_loss_weights
        
        # ===== Backbone (分层提取) =====
        if backbone == 'resnet50':
            # 加载预训练权重
            if backbone_weight_path:
                base_model = models.resnet50(pretrained=False)
                state_dict = torch.load(backbone_weight_path, map_location='cpu')
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                base_model.load_state_dict(state_dict, strict=False)
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    print(f"✓ Loaded ResNet50 from: {backbone_weight_path}")
            elif pretrained:
                base_model = models.resnet50(pretrained=True)
            else:
                base_model = models.resnet50(pretrained=False)
            
            # ResNet50的各层通道数
            layer_channels = [256, 512, 1024, 2048]
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 分层提取特征
        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
        )  # 4x下采样
        
        self.layer1 = base_model.layer1  # 4x下采样  (256通道)
        self.layer2 = base_model.layer2  # 8x下采样  (512通道)
        self.layer3 = base_model.layer3  # 16x下采样 (1024通道)
        self.layer4 = base_model.layer4  # 32x下采样 (2048通道)
        
        # ===== 多尺度融合 =====
        if use_multiscale:
            self.fusion = MultiscaleFusionModule(
                in_channels_list=layer_channels,
                out_channels=256,  # 统一到256通道
            )
            # 投影层：统一通道数到encoder_output_dim
            self.fusion_projection = nn.Sequential(
                nn.Conv2d(256, encoder_output_dim, 1, bias=False),
                nn.GroupNorm(32, encoder_output_dim),
                nn.ReLU(inplace=False),
            )
            
            # 根据fusion_layer决定有效下采样倍数
            if fusion_layer == 'layer1':
                effective_downsample = 4   # layer1: 4x下采样
            elif fusion_layer == 'layer2':
                effective_downsample = 8   # layer2: 8x下采样
            elif fusion_layer == 'layer1+2':
                effective_downsample = 8   # 融合后在layer2尺度，8x下采样
            else:
                raise ValueError(f"Unknown fusion_layer: {fusion_layer}")
                
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"✓ Multiscale fusion enabled: using {fusion_layer}")
                print(f"  Effective downsample: {effective_downsample}x")
        else:
            # 原始版本：只用最深层
            self.feature_projection = nn.Sequential(
                nn.Conv2d(backbone_dim, encoder_output_dim, 1, bias=False),
                nn.GroupNorm(32, encoder_output_dim),
                nn.ReLU(inplace=False),
            )
            effective_downsample = 32
        
        # ===== 解耦模块 =====
        self.disentangle_head = DisentangleHead(
            input_dim=encoder_output_dim,
            content_dim=content_dim,
            subtitle_dim=subtitle_dim,
        )
        
        # ===== Mask预测头（改进：直接从融合特征预测）=====
        # 关键改进：mask预测直接从融合后的浅层特征，不经过解耦模块
        # 这样可以保留更多空间细节，对小字幕更友好
        if use_multiscale:
            # 多尺度版本：直接从融合特征预测
            self.mask_head = MaskHead(
                input_dim=256,  # 融合特征的通道数
                upsample_factor=effective_downsample,
            )
        else:
            # 原始版本：从subtitle特征预测
            self.mask_head = MaskHead(
                input_dim=subtitle_dim,
                upsample_factor=effective_downsample,
            )
        
        # ===== 重建解码器 =====
        if use_reconstruction:
            self.decoder = LightweightDecoder(
                input_dim=content_dim,
                output_channels=3,
            )
        
        # ===== 自适应损失权重（可选）=====
        if use_adaptive_loss_weights:
            # 初始化权重：如果提供则使用，否则使用配置值转换为log_vars
            if init_loss_weights is not None:
                # 将权重 [w1, w2, w3] 转换为 log_vars
                # w = 1/(2σ²) => σ² = 1/(2w) => log(σ²) = -log(2w)
                import math
                init_log_vars = [-math.log(2 * w) if w > 0 else 0.0 for w in init_loss_weights]
            else:
                # 默认初始化为0（即σ²=1，权重=0.5）
                init_log_vars = None
            
            self.adaptive_loss_weights = AdaptiveLossWeights(
                num_tasks=3,  # mask, disentangle, reconstruction
                init_log_vars=init_log_vars
            )
            
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print("✓ 自适应损失权重已启用")
                if init_loss_weights:
                    print(f"  初始权重: mask={init_loss_weights[0]:.2f}, "
                          f"disentangle={init_loss_weights[1]:.2f}, "
                          f"recon={init_loss_weights[2]:.2f}")
        else:
            self.adaptive_loss_weights = None
    
    def extract_features(self, x):
        """提取多层特征"""
        # Stem
        x = self.stem(x)  # 4x
        
        # 各层特征
        feat1 = self.layer1(x)   # 4x,  256通道
        feat2 = self.layer2(feat1)  # 8x,  512通道
        feat3 = self.layer3(feat2)  # 16x, 1024通道
        feat4 = self.layer4(feat3)  # 32x, 2048通道
        
        return [feat1, feat2, feat3, feat4]
    
    def encode(self, subtitle_frame):
        """
        编码：从图像到解耦特征
        
        Returns:
            content_feat: 内容特征（用于重建）
            subtitle_feat: 字幕特征（用于语义学习）
            mask_feat: 用于mask预测的特征（融合后的浅层特征，包含更多空间细节）
        """
        # 提取多层特征
        features = self.extract_features(subtitle_frame)
        
        if self.use_multiscale:
            # 多尺度融合
            fused_features = self.fusion(features)
            
            # ⭐ 关键：mask预测使用的特征（融合后但未投影，保留空间细节）
            # 根据配置选择使用哪一层或哪几层
            if self.fusion_layer == 'layer1':
                # 使用layer1 (4x下采样) - 最多细节
                mask_feat = fused_features[0]  # (B, 256, H/4, W/4)
                mixed_feat = self.fusion_projection(mask_feat)
            elif self.fusion_layer == 'layer2':
                # 使用layer2 (8x下采样) - 平衡细节和语义
                mask_feat = fused_features[1]  # (B, 256, H/8, W/8)
                mixed_feat = self.fusion_projection(mask_feat)
            elif self.fusion_layer == 'layer1+2':
                # ✅ 融合layer1和layer2 - 最强细节保留！
                # mask预测使用融合后的特征
                feat1 = fused_features[0]  # (B, 256, H/4, W/4)
                feat2 = fused_features[1]  # (B, 256, H/8, W/8)
                
                # 下采样feat1到feat2的尺寸
                feat1_down = F.interpolate(feat1, size=feat2.shape[2:], mode='bilinear', align_corners=False)
                
                # 加权融合：layer2权重更高（更稳定）
                mask_feat = 0.4 * feat1_down + 0.6 * feat2  # (B, 256, H/8, W/8)
                
                # 投影用于解耦
                mixed_feat = self.fusion_projection(mask_feat)
            else:
                raise ValueError(f"Unknown fusion_layer: {self.fusion_layer}")
        else:
            # 只用最深层
            mask_feat = None  # 原始版本不使用分离的mask特征
            mixed_feat = self.feature_projection(features[-1])
        
        # 解耦（用于语义级别的内容/字幕分离）
        content_feat, subtitle_feat = self.disentangle_head(mixed_feat)
        
        return content_feat, subtitle_feat, mask_feat
    
    def forward(self, subtitle_frame, return_all=True):
        """
        完整前向传播
        
        架构说明：
        1. Mask预测：从融合特征（包含浅层+深层）直接预测 → 保留空间细节
        2. 解耦学习：从mixed_feat分离内容/字幕 → 学习语义表示
        3. 重建：从content_feat重建clean → 确保内容特征质量
        """
        target_size = (subtitle_frame.shape[2], subtitle_frame.shape[3])
        
        # 编码：获取解耦特征和mask预测特征
        content_feat, subtitle_feat, mask_feat = self.encode(subtitle_frame)
        
        if not return_all:
            return subtitle_feat
        
        # ⭐ Mask预测：使用融合特征（而不是subtitle_feat）
        # 原因：融合特征包含更多空间细节，对小字幕更友好
        if self.use_multiscale and mask_feat is not None:
            # 多尺度版本：从融合特征预测
            mask_logits = self.mask_head(mask_feat, target_size=target_size)
        else:
            # 原始版本：从subtitle特征预测（兼容性）
            mask_logits = self.mask_head(subtitle_feat, target_size=target_size)
        
        # 重建
        recon_clean = None
        if self.use_reconstruction:
            recon_clean = self.decoder(content_feat)
            if recon_clean.shape[2:] != target_size:
                recon_clean = F.interpolate(
                    recon_clean, size=target_size,
                    mode='bilinear', align_corners=False
                )
        
        return content_feat, subtitle_feat, mask_logits, recon_clean

