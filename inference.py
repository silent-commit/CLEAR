"""
CLEAR Inference: End-to-End Mask-Free Video Subtitle Removal

Performs fully mask-free inference using the trained CLEAR model.
Only requires the subtitled video as input - no external detection modules,
no segmentation models, no auxiliary networks needed.

The adaptive weighting strategy is internalized into LoRA-augmented 
attention maps during training.

Reference: Section 3.5 (Algorithm 1) in the CLEAR paper
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2

# DiffSynth-Studio imports (requires DiffSynth-Studio to be installed or in PYTHONPATH)
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


def calculate_target_resolution(orig_width, orig_height, max_pixels=1280*720, division_factor=16):
    """
    计算目标分辨率（保持横竖屏方向）
    
    根据视频原始分辨率和max_pixels计算目标分辨率：
    1. 如果原始分辨率 <= max_pixels，使用原始（对齐到division_factor）
    2. 如果超过，缩放到max_pixels（保持横竖屏方向）
    3. 确保尺寸是division_factor的倍数（VAE要求）
    4. 🔥 保持原始的横竖屏方向（竖屏→竖屏，横屏→横屏）
    """
    # 如果超过max_pixels，等比缩小
    if orig_width * orig_height > max_pixels:
        scale = ((orig_width * orig_height) / max_pixels) ** 0.5
        target_h = int(orig_height / scale)
        target_w = int(orig_width / scale)
    else:
        target_h = orig_height
        target_w = orig_width
    
    # 对齐到division_factor（VAE下采样要求）
    target_h = target_h // division_factor * division_factor
    target_w = target_w // division_factor * division_factor
    
    # 🔥 确保短边不超过720（保持横竖屏方向）
    is_portrait = orig_height > orig_width  # 竖屏
    if is_portrait:
        # 竖屏：宽度是短边，应该<=720
        if target_w > 720:
            scale = 720 / target_w
            target_w = 720
            target_h = int(target_h * scale)
            target_h = target_h // division_factor * division_factor
    else:
        # 横屏：高度是短边，应该<=720
        if target_h > 720:
            scale = 720 / target_h
            target_h = 720
            target_w = int(target_w * scale)
            target_w = target_w // division_factor * division_factor
    
    return target_w, target_h


def crop_and_resize_image(image, target_width, target_height):
    """
    智能裁剪和缩放（与训练时一致）
    
    1. 计算scale使得图像覆盖目标尺寸（scale = max(tw/w, th/h)）
    2. resize到scale后的尺寸
    3. center crop到目标尺寸
    
    这样可以保持宽高比，避免图像拉伸变形
    """
    import torchvision.transforms.functional as TF
    
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    
    # Resize to cover target size
    new_height = round(height * scale)
    new_width = round(width * scale)
    image = TF.resize(
        image,
        (new_height, new_width),
        interpolation=TF.InterpolationMode.BILINEAR
    )
    
    # Center crop to exact target size
    image = TF.center_crop(image, (target_height, target_width))
    return image


def load_video_pil(video_path, max_frames=None, target_resolution=None, max_pixels=1280*720):
    """
    加载视频，返回PIL Image列表
    
    Args:
        video_path: 视频路径
        max_frames: 最大帧数限制（None表示加载所有帧）
        target_resolution: 目标分辨率 (width, height)，None表示自动计算
        max_pixels: 最大像素数（默认720P，与训练一致）
    
    Returns:
        List[PIL.Image]: PIL Image列表
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    # 读取第一帧以确定原始分辨率
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"无法读取视频: {video_path}")
    
    orig_height, orig_width = first_frame.shape[:2]
    
    # 计算目标分辨率
    if target_resolution is None:
        target_width, target_height = calculate_target_resolution(orig_width, orig_height, max_pixels)
        print(f"  原始分辨率: {orig_width}x{orig_height}")
        if (orig_width, orig_height) != (target_width, target_height):
            print(f"  目标分辨率: {target_width}x{target_height} (resize到720P范围)")
        else:
            print(f"  目标分辨率: {target_width}x{target_height} (保持原始)")
    else:
        target_width, target_height = target_resolution
    
    need_resize = (orig_width, orig_height) != (target_width, target_height)
    
    # 重置到开头
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        
        # Resize if needed
        if need_resize:
            pil_img = crop_and_resize_image(pil_img, target_width, target_height)
        
        frames.append(pil_img)
        
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    return frames


def save_video_pil(pil_frames, output_path, fps=30):
    """
    将PIL Image列表保存为视频
    
    Args:
        pil_frames: List[PIL.Image]
        output_path: 输出路径
        fps: 帧率
    """
    if len(pil_frames) == 0:
        raise ValueError("No frames to save")
    
    # 获取第一帧的尺寸
    width, height = pil_frames[0].size
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )
    
    if not out_writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {output_path}")
    
    # 逐帧写入
    for pil_img in tqdm(pil_frames, desc="保存视频"):
        # PIL to numpy
        frame = np.array(pil_img)
        # RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out_writer.write(frame)
    
    out_writer.release()
    print(f"✓ 视频已保存: {output_path}")


def create_comparison_video(source_path, cleaned_path, output_path):
    """创建横向对比视频"""
    print(f"创建对比视频: {output_path}")
    
    # 加载视频（source也需要resize到720P范围，与推理时一致）
    source_frames = load_video_pil(source_path, max_pixels=1280*720)
    cleaned_frames = load_video_pil(cleaned_path)  # cleaned已经是处理后的尺寸
    
    # 🔥 裁剪到相同帧数（使用cleaned的帧数，丢弃source多余的帧）
    min_frames = min(len(source_frames), len(cleaned_frames))
    if len(source_frames) != len(cleaned_frames):
        dropped_frames = abs(len(source_frames) - len(cleaned_frames))
        print(f"📊 对比视频帧数对齐:")
        print(f"   Source: {len(source_frames)} 帧")
        print(f"   Cleaned: {len(cleaned_frames)} 帧")
        print(f"   对齐到: {min_frames} 帧 (丢弃末尾 {dropped_frames} 帧)")
        source_frames = source_frames[:min_frames]
        cleaned_frames = cleaned_frames[:min_frames]
    
    # 获取FPS
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # 获取尺寸
    width, height = source_frames[0].size
    
    # 创建标签高度
    label_height = 40
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width * 2, height + label_height)
    )
    
    # 逐帧处理
    for i in tqdm(range(min_frames), desc="拼接视频"):
        source_img = source_frames[i]
        cleaned_img = cleaned_frames[i]
        
        # PIL to numpy
        source_np = np.array(source_img)
        cleaned_np = np.array(cleaned_img)
        
        # 横向拼接
        combined = np.hstack([source_np, cleaned_np])
        
        # 创建标签区域
        label_area = np.ones((label_height, width * 2, 3), dtype=np.uint8) * 255
        
        # 添加文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(label_area, "Source", (width//2 - 50, 28), 
                    font, 1.0, (0, 0, 0), 2)
        cv2.putText(label_area, "Cleaned", (width + width//2 - 60, 28), 
                    font, 1.0, (0, 0, 0), 2)
        cv2.line(label_area, (width, 0), (width, label_height), (200, 200, 200), 2)
        
        # 拼接
        final_frame = np.vstack([label_area, combined])
        
        # RGB to BGR
        final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
        out_writer.write(final_frame)
    
    out_writer.release()
    print(f"✓ 对比视频已保存: {output_path}")


class Wan21InferencePipeline:
    """Wan2.1推理Pipeline"""
    
    def __init__(
        self,
        model_base_path,
        lora_checkpoint_path=None,
        lora_rank=64,
        lora_scale=1.0,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        device='cuda',
        multi_gpu=False,
    ):
        """
        初始化推理pipeline
        
        Args:
            model_base_path: Wan2.1-1.3B-Control模型路径
            lora_checkpoint_path: LoRA权重路径（可选）
            lora_rank: LoRA rank
            lora_scale: LoRA强度缩放因子（0.0-2.0，默认1.0）
            lora_target_modules: LoRA target modules（必须与训练时一致）
            device: 设备
            multi_gpu: 是否使用多GPU（DataParallel）
        """
        self.device = device
        self.multi_gpu = multi_gpu
        self.lora_rank = lora_rank
        self.lora_scale = lora_scale
        self.lora_target_modules = [m.strip() for m in lora_target_modules.split(',')]
        
        # 检测可用GPU数量
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            print(f"\n🖥️  检测到 {self.num_gpus} 个GPU")
            for i in range(self.num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            
            if self.multi_gpu and self.num_gpus > 1:
                print(f"✓ 将使用 {self.num_gpus} 个GPU进行推理")
            else:
                print(f"✓ 将使用单GPU推理 (GPU 0)")
        else:
            self.num_gpus = 0
            print("⚠️  未检测到GPU，将使用CPU")
        
        print(f"\n加载Wan2.1-1.3B-Control模型: {model_base_path}")
        
        # 🔥 使用本地文件，避免下载
        import glob
        
        diffusion_files = sorted(glob.glob(os.path.join(model_base_path, "diffusion_pytorch_model*.safetensors")))
        if not diffusion_files:
            raise FileNotFoundError(f"No diffusion model files found in {model_base_path}")
        
        model_configs = []
        model_configs.append(ModelConfig(path=diffusion_files, skip_download=True))
        
        t5_path = os.path.join(model_base_path, "models_t5_umt5-xxl-enc-bf16.pth")
        if os.path.exists(t5_path):
            model_configs.append(ModelConfig(path=t5_path, skip_download=True))
        
        vae_path = os.path.join(model_base_path, "Wan2.1_VAE.pth")
        if os.path.exists(vae_path):
            model_configs.append(ModelConfig(path=vae_path, skip_download=True))
        
        clip_path = os.path.join(model_base_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
        if os.path.exists(clip_path):
            model_configs.append(ModelConfig(path=clip_path, skip_download=True))
        
        print(f"✓ 配置了{len(model_configs)}个本地模型文件")
        
        # 🔥 Monkey patch: 完全禁用下载
        original_download = ModelConfig.download_if_necessary
        ModelConfig.download_if_necessary = lambda self, use_usp=False: None
        
        try:
            # 加载pipeline
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=model_configs
            )
        finally:
            # 恢复原函数
            ModelConfig.download_if_necessary = original_download
        
        print("✓ WanVideoPipeline加载完成")
        
        # 🔥 初始化 prompter（text_encoder + tokenizer）
        # 参考训练脚本的实现
        if self.pipe.text_encoder is not None:
            self.pipe.prompter.fetch_models(text_encoder=self.pipe.text_encoder)
            print("✓ Prompter已连接text_encoder")
        
        # 🔥 加载tokenizer（从本地模型目录）
        tokenizer_path = os.path.join(model_base_path, "google/umt5-xxl")
        if os.path.exists(tokenizer_path):
            self.pipe.prompter.fetch_tokenizer(tokenizer_path)
            print(f"✓ Prompter已加载tokenizer: {tokenizer_path}")
        else:
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        # 设置为评估模式
        self.pipe.dit.eval()
        
        # 加载LoRA（如果提供）
        if lora_checkpoint_path:
            self.load_lora(lora_checkpoint_path)
        
        # 🔥 多GPU支持：使用DataParallel包装DiT模型
        if self.multi_gpu and self.num_gpus > 1:
            print(f"\n🚀 启用DataParallel，使用 {self.num_gpus} 个GPU")
            # 确保模型在GPU 0上
            if not next(self.pipe.dit.parameters()).is_cuda:
                self.pipe.dit = self.pipe.dit.to('cuda:0')
            
            # 🔥 保存原始dit模型的引用
            original_dit = self.pipe.dit
            
            # 🔥 创建一个改进的DataParallel代理类
            class DataParallelProxy(torch.nn.DataParallel):
                def __getattr__(self, name):
                    # 先尝试从DataParallel本身获取
                    try:
                        return super().__getattr__(name)
                    except AttributeError:
                        pass
                    
                    # 如果失败，从原始模型获取
                    if hasattr(self, 'module'):
                        return getattr(self.module, name)
                    
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
            # 包装为代理类
            self.pipe.dit = DataParallelProxy(original_dit)
            
            print("✓ DiT模型已包装为DataParallel")
            print(f"  主设备: cuda:0")
            print(f"  并行设备: cuda:0-{self.num_gpus-1}")
            print(f"  ✓ 已修复属性访问兼容性（自动代理所有属性）")
        else:
            print(f"\n使用单GPU推理")
            if self.num_gpus > 1:
                print(f"  检测到 {self.num_gpus} 个GPU，但使用单GPU模式")
                print(f"  原因: DataParallel与WanVideo pipeline有兼容性问题")
                print(f"  单GPU已启用tiled=True优化，性能足够")
    
    def load_lora(self, checkpoint_path):
        """加载LoRA权重（支持 .pt 和 .safetensors 格式）"""
        print(f"\n加载LoRA权重: {checkpoint_path}")
        print(f"  LoRA scale: {self.lora_scale}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"LoRA checkpoint不存在: {checkpoint_path}")
        
        # 注入LoRA到DiT
        from peft import LoraConfig, inject_adapter_in_model
        
        # 🔥 通过调整lora_alpha来控制LoRA强度
        # scaling = lora_alpha / lora_rank
        # 要实现scale倍的效果: lora_alpha = scale * lora_rank
        effective_alpha = self.lora_scale * self.lora_rank
        
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=effective_alpha,  # 🔥 应用scale
            target_modules=self.lora_target_modules,  # 🔥 使用与训练一致的target_modules
            lora_dropout=0.0,
            bias="none",
        )
        
        self.pipe.dit = inject_adapter_in_model(lora_config, self.pipe.dit)
        print("✓ LoRA结构注入完成")
        
        # 🔥 根据文件扩展名选择加载方法
        checkpoint_ext = os.path.splitext(checkpoint_path)[1].lower()
        
        if checkpoint_ext == '.safetensors':
            # 加载 safetensors 格式
            print(f"  检测到 SafeTensors 格式")
            try:
                from safetensors.torch import load_file
                checkpoint = load_file(checkpoint_path, device='cpu')
                print(f"  ✓ SafeTensors 加载成功")
            except ImportError:
                raise ImportError(
                    "需要安装 safetensors 库来加载 .safetensors 文件。\n"
                    "请运行: pip install safetensors"
                )
        else:
            # 加载 .pt/.pth 格式
            print(f"  检测到 PyTorch 格式 ({checkpoint_ext})")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 🔥 处理不同的checkpoint格式
        if checkpoint_ext == '.safetensors':
            # SafeTensors 直接返回 state_dict
            state_dict = checkpoint
            print(f"  ℹ️  SafeTensors 直接包含 state_dict")
        else:
            # PyTorch checkpoint 可能包含额外的元数据
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"  ℹ️  从 'model_state_dict' 键提取权重")
                elif 'dit_state_dict' in checkpoint:
                    state_dict = checkpoint['dit_state_dict']
                    print(f"  ℹ️  从 'dit_state_dict' 键提取权重")
                else:
                    state_dict = checkpoint
                    print(f"  ℹ️  直接使用 checkpoint 作为 state_dict")
            else:
                state_dict = checkpoint
        
        # 🔥 处理key前缀（训练时可能有pipe.dit.前缀）
        processed_state_dict = {}
        for key, value in state_dict.items():
            # 移除可能的前缀
            new_key = key.replace('pipe.dit.', '').replace('module.', '').replace('_orig_mod.', '')
            processed_state_dict[new_key] = value
        
        print(f"处理了{len(processed_state_dict)}个权重")
        
        # 加载到DiT
        missing, unexpected = self.pipe.dit.load_state_dict(processed_state_dict, strict=False)
        
        print(f"✓ LoRA权重加载完成")
        if missing:
            print(f"  ⚠️ Missing keys: {len(missing)}")
            if len(missing) <= 10:
                for k in missing[:10]:
                    print(f"    - {k}")
        if unexpected:
            print(f"  ⚠️ Unexpected keys: {len(unexpected)}")
            if len(unexpected) <= 10:
                for k in unexpected[:10]:
                    print(f"    - {k}")
        
        # 验证LoRA参数已加载
        lora_params = sum(1 for name, _ in self.pipe.dit.named_parameters() if 'lora' in name.lower())
        print(f"  ✅ DiT中的LoRA参数: {lora_params}")
    
    @torch.no_grad()
    def inference(
        self,
        input_video,
        prompt="Please remove the subtitle text from the video while preserving the character appearance, background composition, and color style. Do not add any new elements.",
        num_steps=20,
        cfg_scale=1.0,
        chunk_size=81,
        chunk_overlap=8,
        frame_ratio=1.0,
        use_sliding_window=False,
    ):
        """
        对视频进行推理
        
        Args:
            input_video: List[PIL.Image] 或 视频路径
            prompt: 提示词
            num_steps: 去噪步数
            cfg_scale: CFG scale
            chunk_size: 每个chunk的帧数（手动chunk模式）或sliding_window窗口大小
            chunk_overlap: chunk之间的重叠帧数（手动chunk模式）或sliding_window步长计算
            frame_ratio: 只处理视频的前N%帧（0.0-1.0），用于快速测试
            use_sliding_window: 是否使用Pipeline的sliding_window（latent空间加权融合）
        
        Returns:
            List[PIL.Image]: 处理后的视频帧
        """
        # 加载视频（如果是路径）
        if isinstance(input_video, (str, Path)):
            print(f"\n加载视频: {input_video}")
            input_frames = load_video_pil(str(input_video), max_pixels=1280*720)  # 🔥 与训练一致的resize
        else:
            input_frames = input_video
        
        # 🔥 根据 frame_ratio 均匀采样帧数
        total_frames_original = len(input_frames)
        if 0.0 < frame_ratio < 1.0:
            num_frames_to_process = int(total_frames_original * frame_ratio)
            num_frames_to_process = max(1, num_frames_to_process)  # 至少处理1帧
            
            # ✅ 均匀采样：选取均匀分布的帧，而不是只取前N帧
            indices = np.linspace(0, total_frames_original - 1, num_frames_to_process, dtype=int)
            input_frames = [input_frames[i] for i in indices]
            print(f"\n⚡ 均匀采样模式: 从 {total_frames_original} 帧中均匀采样 {num_frames_to_process} 帧 ({frame_ratio*100:.0f}%)")
            print(f"   采样索引范围: [{indices[0]}, {indices[-1]}]")
        
        total_frames = len(input_frames)
        print(f"\n开始推理: {total_frames} 帧")
        print(f"  Num steps: {num_steps}")
        print(f"  CFG scale: {cfg_scale}")
        print(f"  Use sliding_window: {use_sliding_window}")
        
        # 获取视频尺寸
        width, height = input_frames[0].size
        
        # 🔥 方案1：使用Pipeline的sliding_window（latent空间加权融合）
        if use_sliding_window:
            print(f"\n🔥 使用Pipeline sliding_window模式（latent空间加权融合）")
            print(f"  Sliding window size: {chunk_size} 帧（与训练时的num_frames一致）")
            print(f"  Sliding window stride: {chunk_size - chunk_overlap} 帧（{chunk_overlap}帧重叠）")
            print(f"  ✅ 优势：自动在latent空间融合，边界平滑，与训练一致")
            
            # 🔥 计算实际能处理的最大帧数（最后不足window_size的部分将被丢弃）
            original_total_frames = total_frames
            stride = chunk_size - chunk_overlap
            
            # 在latent空间计算窗口（VAE编码后时间维度缩小4倍）
            # latent_frames = (pixel_frames - 1) // 4 + 1
            latent_total = (total_frames - 1) // 4 + 1
            latent_window_size = (chunk_size - 1) // 4 + 1
            latent_stride = (stride - 1) // 4 + 1
            
            # 🔥 计算最后一个完整窗口能覆盖到的帧数
            # 窗口起始位置: 0, stride, 2*stride, ...
            # 最后一个完整窗口: k*stride, 其中 k*stride + window_size <= latent_total
            if latent_total >= latent_window_size:
                # 最大的k值
                k_max = (latent_total - latent_window_size) // latent_stride
                # 最后能处理到的latent帧
                processed_latent_frames = k_max * latent_stride + latent_window_size
                num_complete_windows = k_max + 1
                
                # 🔥 转换回pixel空间：找最大的pixel帧数，使其latent表示 <= processed_latent_frames
                # latent_frames = (pixel_frames - 1) // 4 + 1
                # 反推: pixel_frames = (latent_frames - 1) * 4 + 1, 但要考虑上限
                expected_output_frames = min(total_frames, (processed_latent_frames - 1) * 4 + 1)
            else:
                # 视频太短，不足一个窗口
                num_complete_windows = 0
                expected_output_frames = 0
            
            if expected_output_frames < total_frames and expected_output_frames > 0:
                # 🔥 裁剪输入视频，避免sliding window处理不完整窗口
                dropped_frames = total_frames - expected_output_frames
                print(f"\n📊 视频处理策略:")
                print(f"   原始帧数: {total_frames}")
                print(f"   完整窗口数: {num_complete_windows}")
                print(f"   裁剪到: {expected_output_frames} 帧 (确保完整处理)")
                print(f"   末尾丢弃: {dropped_frames} 帧 (不足完整窗口)")
                input_frames = input_frames[:expected_output_frames]
                total_frames = expected_output_frames
            elif expected_output_frames == 0:
                print(f"\n⚠️  警告: 视频帧数 {total_frames} 不足一个窗口 ({chunk_size}帧)，无法处理")
                raise ValueError(f"视频太短，至少需要 {chunk_size} 帧")
            else:
                print(f"\n✓ 视频帧数 {total_frames} 可完整处理")
            
            # 🔥 显示GPU使用情况
            if self.multi_gpu and self.num_gpus > 1:
                print(f"\n💡 多GPU推理配置:")
                print(f"   - 使用GPU数量: {self.num_gpus}")
                print(f"   - 并行策略: DataParallel")
            else:
                print(f"\n💡 单GPU推理配置:")
                print(f"   - 使用GPU: cuda:0")
            
            # 🔥 参考官方脚本结构，使用sliding_window
            # Pipeline会自动在latent空间进行加权融合
            try:
                output_frames = self.pipe(
                    prompt=prompt,
                    control_video=input_frames,  # ✅ 整个视频，不分chunk
                    height=height,
                    width=width,
                    num_frames=total_frames,  # ✅ 总帧数
                    num_inference_steps=num_steps,
                    cfg_scale=cfg_scale,
                    tiled=True,  # 🔥 与训练时VAE的tiled=True一致
                    tile_size=(30, 52),  # 🔥 与训练完全一致
                    tile_stride=(15, 26),  # 🔥 与训练完全一致
                    # 🔥 使用sliding_window（latent空间加权融合）
                    sliding_window_size=chunk_size,      # 81帧窗口（与训练时的num_frames一致）
                    sliding_window_stride=chunk_size - chunk_overlap,  # 73帧步长（8帧重叠）
                )
                
                # output_frames应该是List[PIL.Image]
                if isinstance(output_frames, dict) and 'video' in output_frames:
                    output_frames = output_frames['video']
                
                print("\n" + "=" * 60)
                print(f"✓ 推理完成: {len(output_frames)} 帧（sliding_window模式）")
                return output_frames
                
            except Exception as e:
                print(f"\n✗ 错误：sliding_window推理失败: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # 🔥 方案2：手动分chunk（旧方式，保持向后兼容）
        else:
            print(f"\n使用手动chunk模式（旧方式）")
            print(f"  Chunk size: {chunk_size}")
            print(f"  Chunk overlap: {chunk_overlap}")
            
            output_frames = []
            stride = chunk_size - chunk_overlap
            chunk_indices = list(range(0, total_frames, stride))
            
            # 🔥 修复：预先调整所有chunk，然后移除重复的
            adjusted_indices = []
            for i, start in enumerate(chunk_indices):
                end = min(start + chunk_size, total_frames)
                # 如果不足chunk_size，向前扩展
                if end - start < chunk_size:
                    start = max(0, end - chunk_size)
                    print(f"⚠️  Chunk {i+1} 调整为 [{start}:{end}]")
                # 检查是否与前一个重复
                if not adjusted_indices or adjusted_indices[-1] != start:
                    adjusted_indices.append(start)
                else:
                    print(f"⚠️  Chunk {i+1} [{start}:{end}] 与前一个重复，跳过")
            
            chunk_indices = adjusted_indices
            num_chunks = len(chunk_indices)
            print(f"\n将处理 {num_chunks} 个chunks")
            
            # 🔥 显示GPU使用情况
            if self.multi_gpu and self.num_gpus > 1:
                print(f"\n💡 多GPU推理配置:")
                print(f"   - 使用GPU数量: {self.num_gpus}")
                print(f"   - 并行策略: DataParallel")
                print(f"   - 推理过程中可运行 'watch -n 1 nvidia-smi' 查看所有GPU利用率")
                print(f"   - 预期: 所有GPU利用率都会增长（不只是GPU 0）")
            else:
                print(f"\n💡 单GPU推理配置:")
                print(f"   - 使用GPU: cuda:0")
            
            # 🔥 保留chunks级别的进度条
            import sys
            for chunk_id, start_idx in enumerate(tqdm(chunk_indices, desc="处理chunks", file=sys.stdout), 1):
                # 🔥 确保每个chunk都是chunk_size帧，如果不足就向前扩展
                end_idx = min(start_idx + chunk_size, total_frames)
                if end_idx - start_idx < chunk_size:
                    # 向前扩展到chunk_size帧
                    start_idx = max(0, end_idx - chunk_size)
                
                chunk_frames = input_frames[start_idx:end_idx]
                
                # 获取帧尺寸
                chunk_width, chunk_height = chunk_frames[0].size
                
                # 显示当前chunk信息
                print(f"\n📹 Chunk {chunk_id}/{num_chunks}: 帧 [{start_idx}:{end_idx}] ({len(chunk_frames)} 帧), 分辨率 {chunk_width}x{chunk_height}")
                print(f"   ⚙️  Pipeline配置: tiled=True (与训练、官方一致)")
                
                # 调用pipeline推理
                try:
                    # 🔥 使用Pipeline的标准配置：
                    # - tiled=True: VAE空间分块（与训练、官方示例一致）
                    output_chunk = self.pipe(
                        prompt=prompt,
                        control_video=chunk_frames,  # ✅ 正确：控制信号
                        height=chunk_height,
                        width=chunk_width,
                        num_frames=len(chunk_frames),
                        num_inference_steps=num_steps,
                        cfg_scale=cfg_scale,
                        tiled=True,  # 🔥 VAE空间tiling
                        tile_size=(30, 52),
                        tile_stride=(15, 26),
                    )
                    
                    # output_chunk应该是List[PIL.Image]
                    if isinstance(output_chunk, dict) and 'video' in output_chunk:
                        output_chunk = output_chunk['video']
                    
                    # 处理重叠部分
                    if chunk_id == 1:
                        # 第一个chunk，保留全部
                        output_frames.extend(output_chunk)
                        print(f"   ✓ Chunk {chunk_id}/{num_chunks} 完成，添加 {len(output_chunk)} 帧，累计输出 {len(output_frames)} 帧")
                    else:
                        # 后续chunk，计算实际的重叠帧数
                        # 重叠帧数 = 当前chunk起始位置 - 上一个chunk结束位置的负数
                        # 如果start_idx < len(output_frames)，说明有重叠
                        if start_idx < len(output_frames):
                            actual_overlap = len(output_frames) - start_idx
                            output_chunk = output_chunk[actual_overlap:]
                            print(f"   ℹ️  与前面重叠 {actual_overlap} 帧，跳过")
                        
                        output_frames.extend(output_chunk)
                        print(f"   ✓ Chunk {chunk_id}/{num_chunks} 完成，添加 {len(output_chunk)} 帧，累计输出 {len(output_frames)} 帧")
                    
                except Exception as e:
                    print(f"\n   ✗ 错误：处理chunk [{start_idx}:{end_idx}] 时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    # 出错时使用原始帧
                    output_frames.extend(chunk_frames)
                    print(f"   ⚠ 使用原始帧替代，累计输出 {len(output_frames)} 帧")
            
            print("\n" + "=" * 60)
            print(f"✓ 推理完成: {len(output_frames)} 帧（手动chunk模式）")
            return output_frames


def main():
    parser = argparse.ArgumentParser()
    
    # 模型配置
    parser.add_argument('--model_base_path', type=str, required=True,
                        help='Wan2.1-1.3B-Control模型路径')
    parser.add_argument('--lora_checkpoint', type=str, default=None,
                        help='LoRA checkpoint路径')
    parser.add_argument('--lora_rank', type=int, default=64,
                        help='LoRA rank（必须与训练时一致）')
    parser.add_argument('--lora_scale', type=float, default=1.0,
                        help='LoRA强度缩放因子（0.0-2.0，默认1.0=完全应用）')
    parser.add_argument('--lora_target_modules', type=str, default="q,k,v,o,ffn.0,ffn.2",
                        help='LoRA target modules（必须与训练时一致）')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='使用多GPU推理（DataParallel）')
    
    # 输入输出
    parser.add_argument('--input_video', type=str, required=True,
                        help='输入视频路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    
    # 推理参数
    parser.add_argument('--prompt', type=str, 
                        default="Please remove the subtitle text from the video while preserving the character appearance, background composition, and color style. Do not add any new elements.",
                        help='提示词（必须与训练时一致）')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='去噪步数')
    parser.add_argument('--cfg_scale', type=float, default=1.0,
                        help='CFG scale')
    parser.add_argument('--chunk_size', type=int, default=81,
                        help='每个chunk的帧数')
    parser.add_argument('--chunk_overlap', type=int, default=8,
                        help='chunk重叠帧数')
    parser.add_argument('--frame_ratio', type=float, default=1.0,
                        help='只处理视频的前N%%帧（0.0-1.0），例如0.3表示只处理前30%%，用于快速测试')
    parser.add_argument('--use_sliding_window', action='store_true',
                        help='使用Pipeline的sliding_window（latent空间加权融合），推荐用于改善chunk边界闪烁问题')
    
    # 其他选项
    parser.add_argument('--create_comparison', action='store_true',
                        help='创建对比视频')
    parser.add_argument('--copy_source', action='store_true',
                        help='复制原始视频到输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化pipeline
    pipeline = Wan21InferencePipeline(
        model_base_path=args.model_base_path,
        lora_checkpoint_path=args.lora_checkpoint,
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,  # 🔥 LoRA强度
        lora_target_modules=args.lora_target_modules,  # 🔥 与训练保持一致
        device='cuda',
        multi_gpu=args.multi_gpu,
    )
    
    # 推理（记录时间）
    import time
    inference_start_time = time.time()
    
    output_frames = pipeline.inference(
        input_video=args.input_video,
        prompt=args.prompt,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        frame_ratio=args.frame_ratio,
        use_sliding_window=args.use_sliding_window,
    )
    
    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time
    
    # 保存结果
    video_name = Path(args.input_video).stem
    output_cleaned = output_dir / f"{video_name}_cleaned.mp4"
    
    # 获取原视频FPS
    cap = cv2.VideoCapture(args.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    save_video_pil(output_frames, output_cleaned, fps=fps)
    
    # 保存原始视频（resize到与cleaned一致的分辨率）
    if args.copy_source:
        output_source = output_dir / f"{video_name}_source.mp4"
        print(f"\n保存原始视频（resize后）: {output_source}")
        source_frames_resized = load_video_pil(args.input_video, max_pixels=1280*720)
        save_video_pil(source_frames_resized, output_source, fps=fps)
        print(f"✓ 原始视频已保存: {output_source}")
    
    # 创建对比视频
    if args.create_comparison:
        output_comparison = output_dir / f"{video_name}_comparison.mp4"
        source_path = args.input_video
        create_comparison_video(source_path, output_cleaned, output_comparison)
    
    # 保存推理时间信息
    import json
    num_frames = len(output_frames)
    time_per_frame = inference_time / num_frames if num_frames > 0 else 0
    
    timing_info = {
        'total_time_seconds': inference_time,
        'time_per_frame_seconds': time_per_frame,
        'num_frames': num_frames,
        'video_name': video_name
    }
    
    timing_file = output_dir / 'timing_info.json'
    with open(timing_file, 'w') as f:
        json.dump(timing_info, f, indent=2)
    
    print(f"\n✓ 所有处理完成！")
    print(f"  输出目录: {output_dir}")
    print(f"  推理时间: {inference_time:.2f}秒 (每帧 {time_per_frame:.3f}秒, 总帧数 {num_frames})")


if __name__ == '__main__':
    main()

