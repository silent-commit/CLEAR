"""
视频处理工具函数
包括视频加载、帧采样、VAE编码等
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import decord
from decord import VideoReader, cpu, gpu


decord.bridge.set_bridge('torch')


def load_video(video_path: str, num_frames: Optional[int] = None) -> torch.Tensor:
    """
    加载视频文件
    
    Args:
        video_path: 视频路径
        num_frames: 需要采样的帧数，None表示加载所有帧
        
    Returns:
        视频tensor, shape: [T, H, W, C], 值范围[0, 1]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    if num_frames is None:
        frame_indices = list(range(total_frames))
    else:
        # 均匀采样
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    frames = vr.get_batch(frame_indices)  # [T, H, W, C]
    frames = frames.float() / 255.0  # 归一化到[0, 1]
    
    return frames


def save_video(video_tensor: torch.Tensor, output_path: str, fps: int = 30, audio_path: Optional[str] = None):
    """
    保存视频tensor到文件
    
    Args:
        video_tensor: [T, H, W, C], 值范围[0, 1]
        output_path: 输出路径
        fps: 帧率
        audio_path: 可选的音频文件路径，用于合并音频
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # 转换为numpy并调整值范围
    # 先转换为float32（避免BFloat16不被numpy支持的问题）
    video_tensor = video_tensor.float()
    video_np = (video_tensor.cpu().numpy() * 255).astype(np.uint8)
    T, H, W, C = video_np.shape
    
    # 临时视频路径（无音频）
    temp_path = output_path.replace('.mp4', '_temp.mp4')
    temp_raw_path = output_path.replace('.mp4', '_temp_raw.yuv')
    
    # 方法1：优先尝试使用ffmpeg直接编码（高质量H.264）
    try:
        import subprocess
        
        # 保存原始帧数据到临时文件
        # 使用ffmpeg从原始YUV数据编码为H.264 MP4
        # 注意：需要将RGB转换为YUV格式
        
        # 创建ffmpeg进程，通过管道输入原始RGB数据
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-f', 'rawvideo',  # 输入格式：原始视频
            '-vcodec', 'rawvideo',
            '-s', f'{W}x{H}',  # 视频尺寸
            '-pix_fmt', 'rgb24',  # 像素格式
            '-r', str(fps),  # 帧率
            '-i', '-',  # 从stdin读取
            '-c:v', 'libx264',  # 使用libx264编码
            '-preset', 'medium',  # 编码预设（fast/medium/slow）
            '-crf', '18',  # 质量因子（18是高质量，23是默认）
            '-pix_fmt', 'yuv420p',  # 输出像素格式
            temp_path
        ]
        
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 写入所有帧
        for frame in video_np:
            process.stdin.write(frame.tobytes())
        
        process.stdin.close()
        process.wait()
        
        if process.returncode == 0:
            print(f"✓ Using ffmpeg with libx264 codec (high quality, small size)")
            
            # 如果提供了音频，使用ffmpeg合并
            if audio_path and os.path.exists(audio_path):
                final_cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_path,
                    '-i', audio_path,
                    '-c:v', 'copy',  # 不重新编码视频
                    '-c:a', 'aac',   # 音频编码为AAC
                    '-strict', 'experimental',
                    output_path
                ]
                subprocess.run(final_cmd, check=True, capture_output=True)
                os.remove(temp_path)
            else:
                os.rename(temp_path, output_path)
            
            return  # 成功，直接返回
            
        else:
            stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
            print(f"⚠ ffmpeg encoding failed: {stderr_output}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise RuntimeError("ffmpeg encoding failed")
            
    except Exception as e:
        print(f"⚠ ffmpeg method failed: {e}")
        print(f"  Falling back to OpenCV VideoWriter...")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # 方法2：回退到OpenCV VideoWriter（如果ffmpeg失败）
    # 写入视频 - 尝试多个编码器，按优先级回退
    # 优先级: H264 (x264) > XVID > MJPEG > mp4v
    codecs = [
        ('X264', 'x264'),  # H264 (最常用)
        ('avc1', 'H264'),  # H264 alternative
        ('XVID', 'xvid'),  # XVID
        ('MJPG', 'mjpeg'), # Motion JPEG (兼容性好)
        ('mp4v', 'mpeg4'), # MPEG4 (最后选择)
    ]
    
    writer = None
    for fourcc_str, codec_name in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(temp_path, fourcc, fps, (W, H))
            if writer.isOpened():
                print(f"✓ Using OpenCV VideoWriter with codec: {codec_name} ({fourcc_str})")
                print(f"  ⚠ Warning: This codec may produce large files. Consider installing ffmpeg with libx264.")
                break
            else:
                writer.release()
                writer = None
        except Exception as e:
            if writer is not None:
                writer.release()
            writer = None
            continue
    
    if writer is None or not writer.isOpened():
        raise RuntimeError(f"Failed to initialize VideoWriter with any codec. Tried: {[c[1] for c in codecs]}")
    
    for frame in video_np:
        # OpenCV使用BGR格式
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()
    
    # 如果提供了音频，使用ffmpeg合并
    if audio_path and os.path.exists(audio_path):
        cmd = f"ffmpeg -y -i {temp_path} -i {audio_path} -c:v copy -c:a aac -strict experimental {output_path}"
        os.system(cmd)
        os.remove(temp_path)
    else:
        os.rename(temp_path, output_path)


def get_video_info(video_path: str) -> dict:
    """
    获取视频信息
    
    Returns:
        dict包含: total_frames, fps, width, height, duration
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    
    return {
        'total_frames': len(vr),
        'fps': vr.get_avg_fps(),
        'width': vr[0].shape[1],
        'height': vr[0].shape[0],
        'duration': len(vr) / vr.get_avg_fps()
    }


def match_video_pairs(clean_dirs: List[str], subtitle_dirs: List[str]) -> List[dict]:
    """
    匹配无字幕和有字幕的视频对
    
    Args:
        clean_dirs: 无字幕视频目录列表
        subtitle_dirs: 有字幕视频目录列表
        
    Returns:
        视频对列表，每个元素是 {'clean': path, 'subtitle': path, 'name': name}
    """
    video_pairs = []
    
    for clean_dir, subtitle_dir in zip(clean_dirs, subtitle_dirs):
        clean_videos = sorted(Path(clean_dir).glob('*.mp4'))
        subtitle_videos = sorted(Path(subtitle_dir).glob('*.mp4'))
        
        # 构建字典：clean视频使用原始文件名
        clean_dict = {v.stem: str(v) for v in clean_videos}
        
        # subtitle视频需要去掉 "_with_subtitle" 后缀来匹配
        subtitle_dict = {}
        for v in subtitle_videos:
            # 去掉可能的后缀以匹配clean视频
            base_name = v.stem.replace('_with_subtitle', '')
            subtitle_dict[base_name] = str(v)
        
        # 找出公共的视频名称
        common_names = set(clean_dict.keys()) & set(subtitle_dict.keys())
        
        for name in sorted(common_names):
            video_pairs.append({
                'clean': clean_dict[name],
                'subtitle': subtitle_dict[name],
                'name': name
            })
    
    return video_pairs


def sample_frames_uniformly(video: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    从视频中均匀采样指定数量的帧
    
    Args:
        video: [T, H, W, C]
        num_frames: 采样帧数
        
    Returns:
        采样后的视频 [num_frames, H, W, C]
    """
    T = video.shape[0]
    if T <= num_frames:
        return video
    
    indices = np.linspace(0, T - 1, num_frames, dtype=int)
    return video[indices]


def determine_num_frames(height: int) -> int:
    """
    根据视频分辨率确定训练时使用的帧数
    
    Args:
        height: 视频高度
        
    Returns:
        建议的帧数
    """
    if height <= 720:
        return 121
    elif height <= 1080:
        return 65
    elif height <= 1536:
        return 41
    else:
        return 25  # 更高分辨率使用更少帧


def determine_chunk_size(height: int) -> int:
    """
    根据视频分辨率确定推理时的chunk大小
    
    Args:
        height: 视频高度
        
    Returns:
        chunk大小（帧数）
    
    注意：对于mask_guided训练的模型，必须使用训练时的帧数（16帧）
    """
    # 🔥 关键修复：使用与训练一致的帧数
    return 16  # 与mask_guided训练保持一致（之前是25，导致推理失败）
    
    # 原始的分辨率自适应逻辑（暂时禁用）
    #if height <= 720:
    #    return 121
    #elif height <= 1080:
    #    return 65
    #elif height <= 1536:
    #    return 41
    #else:
    #    return 25


def extract_audio(video_path: str, audio_output_path: str):
    """
    从视频中提取音频
    
    Args:
        video_path: 输入视频路径
        audio_output_path: 音频输出路径
    """
    cmd = f"ffmpeg -i {video_path} -vn -acodec copy -y {audio_output_path}"
    os.system(cmd)


def normalize_video(video: torch.Tensor) -> torch.Tensor:
    """
    归一化视频到[-1, 1]范围（用于VAE输入）
    
    Args:
        video: [T, H, W, C], 值范围[0, 1]
        
    Returns:
        归一化后的视频 [-1, 1]
    """
    return video * 2.0 - 1.0


def denormalize_video(video: torch.Tensor) -> torch.Tensor:
    """
    反归一化视频从[-1, 1]到[0, 1]
    
    Args:
        video: [T, H, W, C], 值范围[-1, 1]
        
    Returns:
        反归一化后的视频 [0, 1]
    """
    return (video + 1.0) / 2.0


def resize_video(video: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    """
    调整视频分辨率到固定尺寸（保持长宽比 + center crop）
    
    ⚠️  重要：确保所有视频统一到相同尺寸，以便在batch中stack
    
    Args:
        video: [T, H, W, C]
        target_height: 目标高度（固定输出高度）
        target_width: 目标宽度（固定输出宽度）
        
    Returns:
        调整后的视频 [T, target_height, target_width, C]
    """
    import torch.nn.functional as F
    
    # 获取原始尺寸
    T, H, W, C = video.shape
    
    # 计算原始和目标的长宽比
    src_aspect = W / H
    tgt_aspect = target_width / target_height
    
    # [T, H, W, C] -> [T, C, H, W]
    video = video.permute(0, 3, 1, 2)
    
    # Step 1: 保持长宽比resize，使短边≥目标尺寸
    if src_aspect > tgt_aspect:
        # 源视频更宽，按高度缩放（宽度会超出）
        scale_h = target_height
        scale_w = int(target_height * src_aspect)
    else:
        # 源视频更高，按宽度缩放（高度会超出）
        scale_w = target_width
        scale_h = int(target_width / src_aspect)
    
    # 确保尺寸是偶数
    scale_h = scale_h if scale_h % 2 == 0 else scale_h + 1
    scale_w = scale_w if scale_w % 2 == 0 else scale_w + 1
    
    # Resize（保持长宽比）
    video = F.interpolate(video, size=(scale_h, scale_w), 
                          mode='bilinear', align_corners=False)
    
    # Step 2: Center crop到目标尺寸
    if scale_h > target_height:
        # 高度超出，裁剪
        start_h = (scale_h - target_height) // 2
        video = video[:, :, start_h:start_h+target_height, :]
    
    if scale_w > target_width:
        # 宽度超出，裁剪
        start_w = (scale_w - target_width) // 2
        video = video[:, :, :, start_w:start_w+target_width]
    
    # Step 3: 如果尺寸还不够，padding（理论上不应该发生，但保险起见）
    if video.shape[2] < target_height or video.shape[3] < target_width:
        pad_h = max(0, target_height - video.shape[2])
        pad_w = max(0, target_width - video.shape[3])
        video = F.pad(video, (0, pad_w, 0, pad_h), mode='constant', value=0)
    
    # [T, C, H, W] -> [T, H, W, C]
    video = video.permute(0, 2, 3, 1)
    
    # 最终验证
    assert video.shape[1] == target_height and video.shape[2] == target_width, \
        f"Resize failed: got {video.shape}, expected [T, {target_height}, {target_width}, C]"
    
    return video


def adaptive_resize_video(video: torch.Tensor, base_height: int, base_width: int) -> torch.Tensor:
    """
    根据输入视频的方向自适应resize
    - 竖屏视频(H>W) → 竖屏尺寸（高>宽）
    - 横屏视频(W>H) → 横屏尺寸（宽>高）
    
    Args:
        video: [T, H, W, C]
        base_height: 基础高度
        base_width: 基础宽度
        
    Returns:
        调整后的视频 [T, target_H, target_W, C]
    """
    T, H, W, C = video.shape
    
    # 判断视频方向
    is_portrait = H > W  # 竖屏：高>宽
    
    # 确保目标尺寸匹配视频方向
    if is_portrait:
        # 竖屏视频需要竖屏尺寸（H>W）
        target_height = max(base_height, base_width)
        target_width = min(base_height, base_width)
    else:
        # 横屏视频需要横屏尺寸（W>H）
        target_height = min(base_height, base_width)
        target_width = max(base_height, base_width)
    
    return resize_video(video, target_height, target_width)

