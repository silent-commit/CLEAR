# CLEAR-CogVideoX: Subtitle Removal on CogVideoX-2b (Supplementary Experiment)

> **Note**: This is a **supplementary experiment** that migrates the CLEAR method to CogVideoX-2b. The primary results in our paper are based on Wan2.1-Control, which achieves better performance. This CogVideoX variant is provided for completeness and reproducibility.

## Overview

CogVideoX-2b is originally a **text-to-video** generation model that does not support video conditioning input. We modified the model architecture to enable **video-to-video** subtitle removal by expanding the input channels and training with LoRA.

**What we release:**
- Training script and inference script
- LoRA model weights (checkpoint)
- CogVideoX-2b base model weights (download from [HuggingFace](https://huggingface.co/THUDM/CogVideoX-2b))

---

## Architecture Modification

The key modification converts CogVideoX-2b from text-to-video to video-to-video:

```
Original:  patch_embed.proj = Conv2d(16 → 1920, kernel=2, stride=2)
Modified:  patch_embed.proj = Conv2d(32 → 1920, kernel=2, stride=2)
                                     ^^
                              First 16ch: noisy clean latent (pretrained weights preserved)
                              Last  16ch: subtitle condition latent (zero-initialized)
```

Data flow during training:

```
subtitle_video ──→ VAE encode ──→ subtitle_latent (16ch) ─┐
                                                           ├─ concat ──→ [32ch] ──→ patch_embed ──→ Transformer (LoRA) ──→ prediction
clean_video ──→ VAE encode ──→ clean_latent ──→ add_noise ─┘
                                    │
                                    └──→ target (velocity prediction loss)
```

Data flow during inference:

```
subtitle_video ──→ VAE encode ──→ subtitle_latent (16ch) ─┐
                                                           ├─ concat ──→ [32ch] ──→ Reverse Diffusion ──→ clean_latent ──→ VAE decode ──→ output_video
                                   random noise (16ch)  ───┘
```

---

## Training Method

Trains LoRA on the expanded CogVideoX-2b with CLEAR's structure.

**Trainable parameters:**
- LoRA: `to_k`, `to_q`, `to_v`, `to_out.0` (rank=64)
- Expanded `patch_embed.proj` (new 16-channel weights)

---

## Checkpoint Format

Each `.pt` checkpoint contains:

```python
{
    'lora_state_dict': {...},           # LoRA weights
    'proj_state_dict': {                # Expanded patch_embed.proj weights
        'patch_embed.proj.weight': ..., # [1920, 32, 2, 2]
        'patch_embed.proj.bias': ...,   # [1920] or None
    },
    'step': int,                        # Training step
}
```

---

## Dataset Format

Training requires paired video directories:

```
clean_dir/
    video_name.mp4              # Clean video (no subtitles)
subtitle_dir/
    video_name_with_subtitle.mp4 # Same video with subtitles burned in
```

Pairs are matched by filename: the subtitle video's name with `_with_subtitle` removed must match a clean video.

Optional `non_text.json` provides a whitelist filter (array of `"video_id.jpg"` strings; the `.jpg` extension is stripped for matching).

---

## Installation

```bash
pip install -r requirements.txt
```

Additionally, download the [CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b) base model.

---

## Quick Start: Inference

```bash
# Single video inference
CHECKPOINT=/path/to/cogvideox_2b_CLEAR_lora_checkpoint.pt \
MODEL_PATH=/path/to/CogVideoX-2b \
bash scripts/inference_cogvideox_2b.sh \
    --input_video /path/to/subtitle_video.mp4 \
    --output_dir ./output

# Or call the Python script directly
python inference_cogvideox_clear.py \
    --model_path /path/to/CogVideoX-2b \
    --checkpoint /path/to/cogvideox_2b_CLEAR_lora_checkpoint.pt \
    --input_video /path/to/subtitle_video.mp4 \
    --output_dir ./output \
    --num_steps 50 \
    --auto_resolution \
    --match_input_size
```

**Key inference parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_steps` | 50 | Denoising steps (DPM-Solver++ 2M) |
| `--chunk_size` | 49 | Frames per chunk (must be 4k+1) |
| `--chunk_overlap` | 9 | Overlap between chunks for blending |
| `--auto_resolution` | flag | Auto-detect portrait/landscape |
| `--match_input_size` | flag | Resize output to match input dimensions |
| `--lora_rank` | 64 | Must match training rank |

---

## Training

```bash
MODEL_PATH=/path/to/CogVideoX-2b \
ADAPTER_CHECKPOINT=/path/to/adapter_stage1.pt \
ADAPTER_CODE_PATH=/path/to/adapter_code \
CLEAN_DIRS="/data/clean_dir1 /data/clean_dir2" \
SUBTITLE_DIRS="/data/subtitle_dir1 /data/subtitle_dir2" \
NUM_SAMPLES_PER_DIR=500 \
NUM_EPOCHS=5 \
bash scripts/train_cogvideox_2b.sh
```

---

## Project Structure

```
CogVideoX-CLEAR/
├── README.md
├── requirements.txt
├── train_paper_method.py            # Training (focal + temporal + mask)
├── inference_cogvideox_clear.py     # Inference script
└── scripts/
    ├── train_cogvideox_2b.sh        # Training launch script
    └── inference_cogvideox_2b.sh    # Inference launch script
```

---

## Comparison with Wan2.1 Baseline

This CogVideoX experiment serves as a supplementary comparison. Key differences:

| Aspect | Wan2.1 (Primary) | CogVideoX-2b (This repo) |
|--------|-------------------|--------------------------|
| Base model | Wan2.1-Control | CogVideoX-2b |
| Conditioning | Native control architecture | Channel concatenation (modified input layer) |
| Originally | Image/Video conditioned | Text-to-video only (requires architecture change) |
| Noise schedule | Flow Matching | DPM-Solver++ (DDPM-based) |
| Precision | bf16 | fp16 |
| Frames | 81 | 49 |
| Performance | **Better** | Lower (supplementary) |

The CogVideoX variant underperforms the Wan2.1 baseline primarily because:
1. CogVideoX-2b was not designed for video conditioning — our channel concatenation is a simple but suboptimal adaptation
2. Fewer training frames (49 vs 81) limit temporal modeling capacity
3. The model capacity of CogVideoX-2b is smaller

---

## Environment Requirements

- GPU: NVIDIA (CUDA) or AMD (ROCm), recommended ≥ 24GB VRAM
- Python ≥ 3.10
- PyTorch ≥ 2.0
- Multi-GPU training supported via HuggingFace Accelerate

---

## Citation

If you find this code useful, please cite our paper:

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

---

## License

This project is released for academic research purposes, under the [Apache 2.0 License](LICENSE). We claim no rights over your generated contents. Please use responsibly and ensure compliance with applicable laws.
