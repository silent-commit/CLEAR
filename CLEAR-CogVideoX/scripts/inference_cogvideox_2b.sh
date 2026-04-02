#!/bin/bash

# ================================================================
# CogVideoX-CLEAR Inference — CogVideoX-2b
#
# Single-video inference with trained LoRA checkpoint.
#
# Usage:
#   # Single video:
#   CHECKPOINT=/path/to/cogvideox_2b_CLEAR_lora_checkpoint.pt \
#   bash scripts/inference_cogvideox_2b.sh --input_video /path/to/subtitle_video.mp4
#
#   # With options:
#   MODEL_PATH=/path/to/CogVideoX-2b \
#   CHECKPOINT=/path/to/cogvideox_2b_CLEAR_lora_checkpoint.pt \
#   NUM_STEPS=50 \
#   bash scripts/inference_cogvideox_2b.sh --input_video /path/to/video.mp4 --output_dir ./results
# ================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INFERENCE_SCRIPT="${PROJECT_ROOT}/inference_cogvideox_clear.py"

# ========== Configurable Paths ==========
MODEL_PATH="${MODEL_PATH:-/path/to/CogVideoX-2b}"
CHECKPOINT="${CHECKPOINT:-}"

# ========== Inference Defaults ==========
NUM_STEPS="${NUM_STEPS:-50}"
SEED="${SEED:-42}"
LORA_RANK="${LORA_RANK:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-49}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-9}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-720}"
AUTO_RESOLUTION="${AUTO_RESOLUTION:-true}"
MATCH_INPUT_SIZE="${MATCH_INPUT_SIZE:-true}"

# ========== Validate ==========
if [ ! -f "${INFERENCE_SCRIPT}" ]; then
    echo "Error: Inference script not found: ${INFERENCE_SCRIPT}"
    exit 1
fi
if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
    echo "Error: CHECKPOINT not set or file not found."
    echo "Usage: CHECKPOINT=/path/to/cogvideox_2b_CLEAR_lora_checkpoint.pt bash scripts/inference_cogvideox_2b.sh --input_video video.mp4"
    exit 1
fi
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: MODEL_PATH not found: ${MODEL_PATH}"
    exit 1
fi

export TOKENIZERS_PARALLELISM=false

# ========== Parse remaining args and run ==========
CMD="python -u ${INFERENCE_SCRIPT} \
    --model_path ${MODEL_PATH} \
    --checkpoint ${CHECKPOINT} \
    --num_steps ${NUM_STEPS} \
    --seed ${SEED} \
    --lora_rank ${LORA_RANK} \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --chunk_size ${CHUNK_SIZE} \
    --chunk_overlap ${CHUNK_OVERLAP} \
    --copy_source"

if [ "${AUTO_RESOLUTION}" = "true" ]; then
    CMD="${CMD} --auto_resolution"
fi
if [ "${MATCH_INPUT_SIZE}" = "true" ]; then
    CMD="${CMD} --match_input_size"
fi

# Append any extra arguments (e.g., --input_video, --output_dir)
CMD="${CMD} $@"

echo "=========================================="
echo "CogVideoX-CLEAR Inference (CogVideoX-2b)"
echo "=========================================="
echo "Model:      ${MODEL_PATH}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Steps:      ${NUM_STEPS}"
echo "Chunk:      ${CHUNK_SIZE} frames, overlap=${CHUNK_OVERLAP}"
echo "=========================================="
echo ""

eval ${CMD}
