"""
CLEAR Utility Functions

Provides video loading, mask generation, and alignment utilities.
"""

from .video_utils import (
    load_video, save_video, get_video_info,
    match_video_pairs, sample_frames_uniformly,
    determine_num_frames, determine_chunk_size,
    normalize_video, denormalize_video, resize_video, adaptive_resize_video
)

from .mask_utils import (
    generate_subtitle_mask, morphology_operations,
    filter_small_regions, create_bottom_region_mask,
    soft_mask_with_gaussian, expand_mask, temporal_smoothing
)

__all__ = [
    'load_video', 'save_video', 'get_video_info',
    'match_video_pairs', 'sample_frames_uniformly',
    'determine_num_frames', 'determine_chunk_size',
    'normalize_video', 'denormalize_video', 'resize_video', 'adaptive_resize_video',
    'generate_subtitle_mask', 'morphology_operations',
    'filter_small_regions', 'create_bottom_region_mask',
    'soft_mask_with_gaussian', 'expand_mask', 'temporal_smoothing'
]
