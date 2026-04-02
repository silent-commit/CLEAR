#!/bin/bash

# ================================================================
# CogVideoX-CLEAR Training — CogVideoX-2b
#
# Focal loss + Mask guidance + Temporal loss
# Channel-concatenation V2V conditioning
# LoRA fine-tuning (rank=64)
#
# Usage:
#   # Edit the paths below, then:
#   bash scripts/train_cogvideox_2b.sh
#
#   # Override defaults via environment variables:
#   NUM_SAMPLES_PER_DIR=500 NUM_EPOCHS=5 bash scripts/train_cogvideox_2b.sh
# ================================================================

set -e

# ========== Paths (MUST be set before running) ==========
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_ROOT}/train_paper_method.py"

# CogVideoX-2b pretrained model (download from HuggingFace)
MODEL_PATH="${MODEL_PATH:-/path/to/CogVideoX-2b}"

# Stage 1 mask predictor checkpoint (from CLEAR adapter training)
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:-/path/to/adapter_stage1/checkpoint.pt}"

# Path to the directory containing models/dual_encoder.py
ADAPTER_CODE_PATH="${ADAPTER_CODE_PATH:-/path/to/adapter_code}"

# Data filtering list (optional, set to empty string to skip)
NON_TEXT_JSON="${NON_TEXT_JSON:-}"

# Training data directories (clean ↔ subtitle must be paired 1-to-1)
CLEAN_DIRS="${CLEAN_DIRS:-/path/to/clean_video_dir1 /path/to/clean_video_dir2}"
SUBTITLE_DIRS="${SUBTITLE_DIRS:-/path/to/subtitle_video_dir1 /path/to/subtitle_video_dir2}"

# ========== Training Config ==========
NUM_SAMPLES_PER_DIR="${NUM_SAMPLES_PER_DIR:-500}"
NUM_FRAMES=49
RANDOM_SEED="${RANDOM_SEED:-2026}"
LORA_RANK="${LORA_RANK:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
FOCAL_ALPHA="${FOCAL_ALPHA:-5.0}"
TEMPORAL_WEIGHT="${TEMPORAL_WEIGHT:-0.1}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUM="${GRADIENT_ACCUM:-1}"

SAVE_DIR="${PROJECT_ROOT}/checkpoints/cogvideox2b_${NUM_SAMPLES_PER_DIR}_seed${RANDOM_SEED}_ep${NUM_EPOCHS}"

# ========== Environment ==========
echo "=========================================="
echo "CogVideoX-CLEAR Training (CogVideoX-2b)"
echo "=========================================="
echo ""
echo "Config:"
echo "  Model:           ${MODEL_PATH}"
echo "  Adapter ckpt:    ${ADAPTER_CHECKPOINT}"
echo "  LoRA rank:       ${LORA_RANK}"
echo "  Focal alpha:     ${FOCAL_ALPHA}"
echo "  Temporal weight: ${TEMPORAL_WEIGHT}"
echo "  Frames:          ${NUM_FRAMES}"
echo "  Samples:         ${NUM_SAMPLES_PER_DIR} per dir (seed=${RANDOM_SEED})"
echo "  Epochs:          ${NUM_EPOCHS}"
echo "  LR:              ${LEARNING_RATE}"
echo "  Save:            ${SAVE_DIR}"
echo ""

# Validate paths
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: MODEL_PATH not found: ${MODEL_PATH}"
    echo "Please set MODEL_PATH to your CogVideoX-2b model directory."
    exit 1
fi
if [ ! -f "${ADAPTER_CHECKPOINT}" ]; then
    echo "Error: ADAPTER_CHECKPOINT not found: ${ADAPTER_CHECKPOINT}"
    exit 1
fi
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "Error: Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

export TOKENIZERS_PARALLELISM=false

# ========== GPU Detection ==========
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected ${NUM_GPUS} NVIDIA GPUs"
elif command -v rocm-smi &> /dev/null; then
    NUM_GPUS=$(rocm-smi --showid | grep -c "GPU")
    echo "Detected ${NUM_GPUS} AMD GPUs"
else
    echo "Error: No GPU detected"
    exit 1
fi

USE_GPUS=${USE_GPUS:-${NUM_GPUS}}
if [ ${USE_GPUS} -gt 8 ]; then
    USE_GPUS=8
fi
echo "Using ${USE_GPUS} GPUs"

# ========== Accelerate Config ==========
ACCEL_DIR="${PROJECT_ROOT}/configs"
mkdir -p ${ACCEL_DIR} ${SAVE_DIR}

ACCEL_FILE="${ACCEL_DIR}/accelerate_config.yaml"
cat > ${ACCEL_FILE} << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: ${USE_GPUS}
rdzv_backend: static
same_network: true
use_cpu: false
EOF

export ACCELERATE_CONFIG_FILE=${ACCEL_FILE}

# ========== Logging ==========
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_cogvideox2b_${TIMESTAMP}.log"

echo "Log: ${LOG_FILE}"
echo ""

# ========== Build Command ==========
CMD="accelerate launch \
    --config_file ${ACCEL_FILE} \
    --num_processes ${USE_GPUS} \
    --num_machines 1 \
    --mixed_precision fp16 \
    ${TRAIN_SCRIPT} \
    --model_path ${MODEL_PATH} \
    --adapter_checkpoint ${ADAPTER_CHECKPOINT} \
    --clean_dirs ${CLEAN_DIRS} \
    --subtitle_dirs ${SUBTITLE_DIRS} \
    --output_dir ${SAVE_DIR} \
    --num_samples_per_dir ${NUM_SAMPLES_PER_DIR} \
    --num_frames ${NUM_FRAMES} \
    --seed ${RANDOM_SEED} \
    --lora_rank ${LORA_RANK} \
    --focal_alpha ${FOCAL_ALPHA} \
    --temporal_weight ${TEMPORAL_WEIGHT} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUM} \
    --max_grad_norm 1.0 \
    --save_interval 25 \
    --log_interval 1 \
    --gradient_checkpointing \
    --enable_slicing \
    --enable_tiling \
    --mask_dilation_kernel 3"

if [ -n "${ADAPTER_CODE_PATH}" ] && [ -d "${ADAPTER_CODE_PATH}" ]; then
    CMD="${CMD} --adapter_code_path ${ADAPTER_CODE_PATH}"
fi
if [ -n "${NON_TEXT_JSON}" ] && [ -f "${NON_TEXT_JSON}" ]; then
    CMD="${CMD} --non_text_json ${NON_TEXT_JSON}"
fi

# ========== Start Training ==========
echo "Starting training..."
echo "Start: $(date)"
TRAIN_START=$(date +%s)

eval ${CMD} 2>&1 | tee ${LOG_FILE}

TRAIN_EXIT=$?
TRAIN_END=$(date +%s)
DURATION=$((TRAIN_END - TRAIN_START))

echo ""
if [ ${TRAIN_EXIT} -eq 0 ]; then
    echo "Training completed in $((DURATION / 3600))h $(((DURATION % 3600) / 60))m"
    echo "Checkpoint: ${SAVE_DIR}"
    echo ""
    echo "Next: Run inference"
    echo "  CHECKPOINT=${SAVE_DIR}/cogvideox_2b_CLEAR_lora_checkpoint.pt bash scripts/inference_cogvideox_2b.sh --input_video /path/to/video.mp4"
else
    echo "Training FAILED (exit ${TRAIN_EXIT})"
    echo "Check log: ${LOG_FILE}"
fi

exit ${TRAIN_EXIT}
