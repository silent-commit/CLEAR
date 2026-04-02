# CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal

<p align="center">
  <b>Qingdong He<sup>1</sup>*</b> · 
  <b>Chaoyi Wang<sup>2</sup>*</b> · 
  Peng Tang<sup>3</sup> · 
  Yifan Yang<sup>4</sup> · 
  <a href="mailto:ben0xiaobin0hu1@nus.edu.sg"><b>Xiaobin Hu<sup>5</sup>✉️</b></a>
</p>

<p align="center">
  <sup>1</sup>University of Electronic Science and Technology of China &nbsp;&nbsp;
  <sup>2</sup>University of Chinese Academy of Sciences &nbsp;&nbsp;
  <sup>3</sup>Technical University of Munich &nbsp;&nbsp;
  <sup>4</sup>Shanghai Jiao Tong University &nbsp;&nbsp;
  <sup>5</sup>National University of Singapore
</p>

<p align="center">
  * Equal contribution &nbsp;&nbsp; ✉️ Corresponding author
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.21901">
    <img src="https://img.shields.io/badge/arXiv-2603.21901-b31b1b.svg"/>
  </a>
  <a href="https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal">
    <img src="https://img.shields.io/badge/🤗 HuggingFace-Model-yellow"/>
  </a>
  <a href="https://qm.qq.com/q/j0QZUsjinu">
    <img src="https://img.shields.io/badge/💬 QQ群-1082514558-blue"/>
  </a>
</p>

<p align="center">
  💬 QQ Group: <code>1082514558</code>
</p>

<p align="center">
  <a href="assets/demo_videos/english1_demo.mp4">
    <img src="assets/demo_videos/english1_demo.gif" width="80%"/>
  </a>
</p>

<p align="center">
  ▶ Click to watch the full-resolution MP4
</p>

## 🔥 News

- **[2026-04-02]** Added support for **CogVideoX**. See the implementation and usage details in [CLEAR-CogVideoX](https://github.com/silent-commit/CLEAR/tree/main/CLEAR-CogVideoX).
- **[2026-03-24]** We released the [paper](https://arxiv.org/abs/2603.21901) and [model](https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal) for **CLEAR**.

## 📋 Overview

**CLEAR** is a mask-free video subtitle removal framework that achieves end-to-end inference through context-aware adaptive learning. By decoupling prior extraction from generative refinement in a two-stage design, CLEAR requires only **0.77%** of the base diffusion model's parameters for training while outperforming mask-dependent baselines by a large margin.

### Key Features

- 🎯 **End-to-End Mask-Free Inference**: No external text detection or segmentation modules needed at inference time
- 🚀 **Parameter Efficient**: Only 0.77% trainable parameters via LoRA adaptation
- 🌍 **Zero-Shot Cross-Lingual Generalization**: Trained on Chinese subtitles, generalizes to English, Korean, French, Japanese, Russian, and German
- 📊 **State-of-the-Art Performance**: +6.77 dB PSNR and -74.7% VFID over best baselines

## 🎬 Demo Videos

| Language | Demo |
|----------|------|
| English | [english1_demo.mp4](assets/demo_videos/english1_demo.mp4) |
| English | [english2_demo.mp4](assets/demo_videos/english2_demo.mp4) |
| English | [english3_demo.mp4](assets/demo_videos/english3_demo.mp4) |
| Japanese | [japanese_demo.mp4](assets/demo_videos/japanese_demo.mp4) |
| Arabic | [arabic_demo.mp4](assets/demo_videos/arabic_demo.mp4) |

> All demos show zero-shot cross-lingual generalization — the model was trained only on Chinese subtitle data.

## 📊 Performance

Quantitative results on the Chinese subtitle test set (default configuration: rank=64, steps=5, cfg=1.0, lora_scale=1.0):

| Metric | Category | CLEAR (Ours) |
|--------|----------|:------------:|
| PSNR ↑ | Reconstruction | **26.80** |
| SSIM ↑ | Reconstruction | **0.894** |
| LPIPS ↓ | Reconstruction | **0.101** |
| DISTS ↓ | Perceptual | **0.075** |
| VFID ↓ | Perceptual | **20.37** |
| TWE ↓ | Temporal | **1.227** |
| TC ↓ | Temporal | **1.049** |
| Flow Mean ↓ | Motion | **0.209** |
| Flow Var ↓ | Motion | **0.029** |
| Time (s/frame) | Efficiency | 4.86 |

### Ablation Study

| Configuration | PSNR ↑ | SSIM ↑ | LPIPS ↓ | VFID ↓ |
|--------------|:------:|:------:|:-------:|:------:|
| Baseline (LoRA-only) | 21.62 | 0.855 | 0.131 | 34.74 |
| + M1: Stage I Prior with Focal Weighting | 23.11 | 0.868 | 0.130 | 38.21 |
| + M2: Context Distillation | 24.72 | 0.890 | 0.110 | 31.73 |
| + M3: Context-Aware Adaptation | 25.09 | 0.891 | 0.109 | 31.56 |
| + M4: Context Consistency (CLEAR) | **26.80** | **0.894** | **0.101** | **20.37** |

## 🛠️ Installation

### Prerequisites

1. **Wan2.1 Environment**: CLEAR is built on top of [Wan2.1](https://github.com/Wan-Video/Wan2.1). Please follow the Wan2.1 installation guide first:

```bash
# Clone Wan2.1
git clone https://github.com/Wan-Video/Wan2.1.git
cd Wan2.1

# Install Wan2.1 dependencies (requires torch >= 2.4.0)
pip install -r requirements.txt
```

2. **DiffSynth-Studio**: CLEAR uses [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) for the video diffusion pipeline:

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

3. **Download Wan2.1 Model Weights**:

```bash
# Download Wan2.1-Fun-V1.1-1.3B-Control
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./models/Wan2.1-T2V-1.3B
```

4. **Download CLEAR Model Weights**:
```bash
huggingface-cli download charlesw09/CLEAR-mask-free-video-subtitle-removal \
    CLEAR-mask-free-subtitle-removal.pt \
    --local-dir ./checkpoints
```

### Install CLEAR
```bash
git clone https://github.com/your-repo/CLEAR.git
cd CLEAR
pip install -r requirements.txt
```

> ✅ After installation, your `checkpoints/` folder should contain `CLEAR-mask-free-subtitle-removal.pt`.

## 📁 Project Structure

```
CLEAR/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── .gitignore
│
├── configs/
│   └── stage1_config.yaml        # Stage I training configuration
│
├── models/
│   ├── __init__.py
│   ├── dual_encoder.py           # Dual ResNet-50 encoders with FPN fusion
│   ├── disentangled_modules.py   # Disentangled feature learning modules
│   └── occlusion_head.py         # Context-dependent occlusion head (Stage II)
│
├── utils/
│   ├── __init__.py
│   ├── video_utils.py            # Video loading and processing
│   ├── mask_utils.py             # Mask generation and processing
│   ├── contrastive_loss.py       # Contrastive learning utilities
│   └── vae_temporal_alignment.py # VAE temporal alignment
│
├── scripts/
│   ├── train_stage1.sh           # Stage I training launcher
│   ├── train_stage2.sh           # Stage II training launcher
│   └── inference.sh              # Inference launcher
│
├── train_stage1.py               # Stage I: Self-Supervised Prior Learning
├── train_stage2.py               # Stage II: Adaptive Weighting Learning
├── inference.py                  # End-to-End Mask-Free Inference
│
├── checkpoints/                  # Model checkpoints (download separately)
│   └── CLEAR-mask-free-subtitle-removal.pt            # Pre-trained CLEAR LoRA weights
│
└── assets/
    └── demo_videos/              # Demo comparison videos
        ├── english1_demo.mp4
        ├── english2_demo.mp4
        ├── english3_demo.mp4
        ├── japanese_demo.mp4
        └── arabic_demo.mp4
```

## 🚀 Quick Start

### Inference (Mask-Free)

The simplest way to use CLEAR — just provide a video with subtitles:

```bash
# Set model paths
export MODEL_BASE_PATH=/path/to/Wan2.1-Fun-V1.1-1.3B-Control
export LORA_CHECKPOINT=./checkpoints/CLEAR-mask-free-subtitle-removal.pt

# Run inference
bash scripts/inference.sh input_video.mp4 ./results
```

Or use Python directly:

```python
python inference.py \
    --model_base_path /path/to/Wan2.1-Fun-V1.1-1.3B-Control \
    --lora_checkpoint ./checkpoints/CLEAR-mask-free-subtitle-removal.pt \
    --lora_rank 64 \
    --lora_scale 1.0 \
    --input_video input_video.mp4 \
    --output_dir ./results \
    --num_steps 5 \
    --cfg_scale 1.0 \
    --use_sliding_window \
    --create_comparison
```

### Training

#### Stage I: Self-Supervised Prior Learning

Trains dual encoders to extract occlusion guidance from paired videos:

```bash
# Edit configs/stage1_config.yaml to set your data paths
bash scripts/train_stage1.sh
```

#### Stage II: Adaptive Weighting Learning

Trains LoRA-adapted diffusion model with context-dependent occlusion head:

```bash
# Set environment variables
export MODEL_BASE_PATH=/path/to/Wan2.1-Fun-V1.1-1.3B-Control
export ADAPTER_CHECKPOINT=./checkpoints/stage1/checkpoint_best.pt
export CLEAN_DIRS=/path/to/clean_videos
export SUBTITLE_DIRS=/path/to/subtitle_videos

bash scripts/train_stage2.sh
```

## ⚙️ Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_steps` | 5 | Denoising steps (5 recommended for speed-quality balance) |
| `cfg_scale` | 1.0 | Classifier-free guidance scale |
| `lora_scale` | 1.0 | LoRA strength (0.0-2.0) |
| `chunk_size` | 81 | Sliding window size (frames) |
| `chunk_overlap` | 16 | Overlap between chunks |

## 🔧 Method Overview

CLEAR consists of two training stages and a mask-free inference pipeline:

### Stage I: Self-Supervised Prior Learning
- Dual ResNet-50 encoders extract disentangled features from video pairs
- Orthogonality constraints ensure feature independence
- Pseudo-labels from pixel differences (no manual annotation needed)
- Output: Coarse occlusion prior M^prior

### Stage II: Adaptive Weighting Learning  
- LoRA adaptation (rank=64) on frozen Wan2.1 diffusion model
- Lightweight occlusion head (~2.1M params) predicts adaptive weights
- Joint optimization: L_distill + L_gen + 0.1 × L_sparse
- Dynamic alpha scheduling prevents over-reliance on noisy priors

### Inference: End-to-End Mask-Free
- Only requires subtitled video input
- No Stage I dependency, no external modules
- Adaptive weighting internalized into LoRA-augmented attention
- Single-pass generation via DDIM sampling

## Citation

If you find this work helpful, please consider citing:

```bibtex
@misc{he2026clearcontextawarelearningendtoend,
      title={CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal}, 
      author={Qingdong He and Chaoyi Wang and Peng Tang and Yifan Yang and Xiaobin Hu},
      year={2026},
      eprint={2603.21901},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.21901}, 
}
```

## 📧 Contact

- **Qingdong He** — [heqingdong@alu.uestc.edu.cn](mailto:heqingdong@alu.uestc.edu.cn) — [Google Scholar](https://scholar.google.com/citations?user=gUJWww0AAAAJ&hl=zh-CN)
- **Chaoyi Wang** — [chaoyiwang@mail.sim.ac.cn](mailto:chaoyiwang@mail.sim.ac.cn) — [Google Scholar](https://scholar.google.com/citations?user=e_wL1LsAAAAJ&hl=zh-CN)

## 🙏 Acknowledgements

This project is built upon the following excellent works:
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) — Open and advanced large-scale video generative models
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — Diffusion model training and inference framework
- [PEFT](https://github.com/huggingface/peft) — Parameter-Efficient Fine-Tuning

## ⚖️ License

This project is released under the [Apache 2.0 License](LICENSE). We claim no rights over your generated contents. Please use responsibly and ensure compliance with applicable laws.

## ⚠️ Disclaimer

We acknowledge the potential for misuse, particularly for generating misinformation. This code is released for research purposes with explicit stipulations against malicious use.

