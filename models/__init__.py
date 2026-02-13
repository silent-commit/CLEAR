"""
CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference 
for Adaptive Video Subtitle Removal

Model components:
- DualEncoder: Disentangled dual encoder for Stage I prior learning
- OcclusionHead: Context-dependent occlusion head for Stage II
"""

from .dual_encoder import MultiscaleDisentangledAdapter as DualEncoder
from .disentangled_modules import (
    DisentangleHead,
    LightweightDecoder,
    MaskHead,
    AdaptiveLossWeights,
    compute_mask_loss,
    compute_disentangle_loss,
    compute_reconstruction_loss,
)
