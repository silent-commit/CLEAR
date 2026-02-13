#!/bin/bash

# ================================================================
# CLEAR Inference: End-to-End Mask-Free Video Subtitle Removal
#
# Performs fully mask-free inference - only requires the subtitled 
# video as input. No external detection modules needed.
#
# Key properties:
# - No Stage I dependency at inference time
# - No external text detection or segmentation modules
# - Single-pass generation via DDIM sampling
# - Sliding window support for long videos
#
# Reference: Section 3.5 (Algorithm 1) in the CLEAR paper
# ================================================================

set -e

# ========== Configuration ==========
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INFERENCE_SCRIPT="${PROJECT_ROOT}/inference.py"

# ========== Model Paths (MODIFY THESE) ==========
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/path/to/Wan2.1-Fun-V1.1-1.3B-Control}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/clear_lora.pt}"

# ========== Inference Parameters ==========
LORA_RANK=64
LORA_SCALE=${LORA_SCALE:-1.0}
NUM_STEPS=${NUM_STEPS:-5}
CFG_SCALE=${CFG_SCALE:-1.0}
CHUNK_SIZE=81
CHUNK_OVERLAP=16

# ========== Input/Output ==========
INPUT_VIDEO="${1:-}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/results}"

if [ -z "${INPUT_VIDEO}" ]; then
    echo "Usage: bash scripts/inference.sh <input_video> [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  input_video  Path to input video with subtitles"
    echo "  output_dir   Output directory (default: ./results)"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_BASE_PATH  Path to Wan2.1-Fun-V1.1-1.3B-Control"
    echo "  LORA_CHECKPOINT  Path to trained CLEAR LoRA weights"
    echo "  LORA_SCALE       LoRA strength (default: 1.0)"
    echo "  NUM_STEPS        Denoising steps (default: 5)"
    echo "  CFG_SCALE        CFG scale (default: 1.0)"
    exit 1
fi

if [ ! -f "${INPUT_VIDEO}" ]; then
    echo "Error: Input video not found: ${INPUT_VIDEO}"
    exit 1
fi

echo "=========================================="
echo "CLEAR: Mask-Free Video Subtitle Removal"
echo "=========================================="
echo "Input: ${INPUT_VIDEO}"
echo "Output: ${OUTPUT_DIR}"
echo "LoRA checkpoint: ${LORA_CHECKPOINT}"
echo "LoRA scale: ${LORA_SCALE}"
echo "Denoising steps: ${NUM_STEPS}"
echo "CFG scale: ${CFG_SCALE}"
echo ""

cd ${PROJECT_ROOT}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

mkdir -p ${OUTPUT_DIR}

PROMPT="Remove the text overlays and subtitles from this video while preserving the original background content and maintaining temporal consistency across frames."

python ${INFERENCE_SCRIPT} \
    --model_base_path ${MODEL_BASE_PATH} \
    --lora_checkpoint ${LORA_CHECKPOINT} \
    --lora_rank ${LORA_RANK} \
    --lora_scale ${LORA_SCALE} \
    --input_video "${INPUT_VIDEO}" \
    --output_dir ${OUTPUT_DIR} \
    --prompt "${PROMPT}" \
    --num_steps ${NUM_STEPS} \
    --cfg_scale ${CFG_SCALE} \
    --chunk_size ${CHUNK_SIZE} \
    --chunk_overlap ${CHUNK_OVERLAP} \
    --use_sliding_window \
    --create_comparison \
    --copy_source

echo ""
echo "Inference completed! Results saved to: ${OUTPUT_DIR}"

