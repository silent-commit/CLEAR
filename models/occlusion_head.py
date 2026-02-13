"""
Context-Dependent Occlusion Head for CLEAR Stage II

As described in the paper:
  M^pred = σ(H(h_enc; φ))
  H(h_enc) = Conv_1x1^1(SiLU(Conv_3x3^64(h_enc)))

The occlusion head is attached to the diffusion transformer's middle encoder layer 
and predicts adaptive weights from DiT encoder features.

Key design:
- Lightweight: only ~2.1M parameters
- Context-dependent: accesses both low-level texture and high-level semantics
- Not a mask predictor, but a weighting function for adaptive generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OcclusionHead(nn.Module):
    """
    Context-Dependent Occlusion Head
    
    Architecture:
        h_enc (D-dim) → Conv3x3(D→64) → SiLU → Conv1x1(64→1) → Sigmoid → M^pred
    
    The predicted M^pred is used for:
    1. Spatial emphasis weighting in the diffusion loss
    2. Context distillation from Stage I priors
    3. Context consistency regularization
    """
    
    def __init__(self, in_dim=1024, hidden_dim=64):
        """
        Args:
            in_dim: Input dimension from DiT encoder features (D=1024 for 1.3B model)
            hidden_dim: Hidden dimension (default 64)
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1, bias=True)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)
        
        # Initialize with small weights for stable training start
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='linear')
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, -2.0)  # Sigmoid(-2) ≈ 0.12, start sparse
    
    def forward(self, h_enc):
        """
        Args:
            h_enc: [B, D, H, W] encoder features from DiT middle layer
                   D is the embedding dimension of the DiT model
        
        Returns:
            M_pred: [B, 1, H, W] predicted occlusion weights in [0, 1]
        """
        x = self.conv1(h_enc)
        x = self.act(x)
        logits = self.conv2(x)
        M_pred = torch.sigmoid(logits)
        return M_pred


def compute_adaptive_weights(M_pred, epsilon_gen, alpha_k, gamma=0.8, delta=1e-6):
    """
    Compute adaptive focal weights as described in the paper:
    
    w_{i,j,t} = (1 + α(k) · M^pred) × (ε^gen + δ)^γ
    
    Args:
        M_pred: [B, 1, T, H, W] predicted occlusion weights
        epsilon_gen: [B, 1, T, H, W] per-pixel reconstruction error
        alpha_k: scalar, dynamic emphasis strength at step k
        gamma: focal weighting exponent (default 0.8)
        delta: small constant for numerical stability
    
    Returns:
        weights: [B, 1, T, H, W] adaptive focal weights
    """
    spatial_emphasis = 1.0 + alpha_k * M_pred
    difficulty_weight = (epsilon_gen.detach() + delta).pow(gamma)
    weights = spatial_emphasis * difficulty_weight
    return weights


def dynamic_alpha_schedule(step, alpha_min=5.0, alpha_max=15.0, period=40):
    """
    Triangular scheduling for dynamic alpha as described in the paper:
    
    α(k) = α_min + (α_max - α_min) · |sin(2πk / T_period)|
    
    Args:
        step: current training step k
        alpha_min: minimum alpha value
        alpha_max: maximum alpha value  
        period: oscillation period T_period
    
    Returns:
        alpha: current alpha value
    """
    import math
    alpha = alpha_min + (alpha_max - alpha_min) * abs(math.sin(2 * math.pi * step / period))
    return alpha


def compute_context_distillation_loss(M_pred, M_prior):
    """
    Context Distillation Loss (Eq. 16 in paper):
    
    L_distill = (1/THW) Σ SmoothL1(M^pred, M^prior)
    
    Provides soft guidance from Stage I while tolerating small deviations.
    
    Args:
        M_pred: [B, T, H, W] or [B, 1, T, H, W] predicted weights
        M_prior: [B, T, H, W] or [B, 1, T, H, W] Stage I prior masks
    
    Returns:
        loss: scalar distillation loss
    """
    return F.smooth_l1_loss(M_pred, M_prior, reduction='mean')


def compute_context_consistency_loss(M_pred, M_prior, sparsity_weight=1.0, kl_weight=0.5):
    """
    Context Consistency Loss (Eq. 20-21 in paper):
    
    L_sparse = L1_sparsity + 0.5 · D_KL(M^pred || M^prior)
    
    - L1 sparsity maintains contextual selectivity
    - KL divergence prevents complete deviation from prior structure
    
    Args:
        M_pred: [B, T, H, W] predicted weights in [0, 1]
        M_prior: [B, T, H, W] prior masks in [0, 1]
        sparsity_weight: weight for L1 sparsity term
        kl_weight: weight for KL divergence term
    
    Returns:
        loss: scalar consistency loss
    """
    # L1 sparsity: encourage sparse predictions
    l1_sparsity = M_pred.mean()
    
    # KL divergence: D_KL(M_pred || M_prior)
    eps = 1e-7
    M_pred_clamped = M_pred.clamp(eps, 1 - eps)
    M_prior_clamped = M_prior.clamp(eps, 1 - eps)
    
    kl_div = M_pred_clamped * torch.log(M_pred_clamped / M_prior_clamped) + \
             (1 - M_pred_clamped) * torch.log((1 - M_pred_clamped) / (1 - M_prior_clamped))
    kl_loss = kl_div.mean()
    
    loss = sparsity_weight * l1_sparsity + kl_weight * kl_loss
    return loss

