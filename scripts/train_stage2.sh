#!/bin/bash

# ================================================================
# CLEAR Stage II: Adaptive Weighting Learning
#
# Trains LoRA-adapted diffusion model with context-dependent
# occlusion head for dynamic context adjustment.
#
# Key features:
# - LoRA adaptation (rank=64) on frozen Wan2.1 diffusion model
# - Only 0.77% of base model parameters are trainable
# - Context-dependent occlusion head (~2.1M parameters)
# - Dynamic alpha scheduling (triangular oscillation)
# - Joint optimization: L_distill + L_gen + 0.1 * L_sparse
#
# Loss function (Eq. 22 in paper):
#   L_stage2 = L_distill + L_gen + 0.1 * L_sparse
#
# Reference: Section 3.3-3.4 in the CLEAR paper
# ================================================================

set -e

# ========== Configuration ==========
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_ROOT}/train_stage2.py"

# ========== Model Paths (MODIFY THESE) ==========
# Base model: Wan2.1-Fun-V1.1-1.3B-Control
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/path/to/Wan2.1-Fun-V1.1-1.3B-Control}"
# Stage I checkpoint (for prior mask prediction)
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/stage1/checkpoint_best.pt}"

# ========== Data Configuration (MODIFY THESE) ==========
NON_TEXT_JSON="${PROJECT_ROOT}/non_text.json"
CLEAN_DIRS="${CLEAN_DIRS:-/path/to/clean_videos}"
SUBTITLE_DIRS="${SUBTITLE_DIRS:-/path/to/subtitle_videos}"

# ========== Training Parameters ==========
NUM_SAMPLES=${NUM_SAMPLES:-500}
NUM_FRAMES=81
RANDOM_SEED=2026
LORA_RANK=64
LEARNING_RATE=1e-4
FOCAL_ALPHA=5.0
TEMPORAL_WEIGHT=0.1
NUM_EPOCHS=1
BATCH_SIZE=1
GRADIENT_ACCUM=1

SAVE_DIR="${PROJECT_ROOT}/checkpoints/stage2_lora"

echo "=========================================="
echo "CLEAR Stage II: Adaptive Weighting Learning"
echo "=========================================="
echo ""
echo "Key Configuration:"
echo "  Base model: ${MODEL_BASE_PATH}"
echo "  LoRA rank: ${LORA_RANK}"
echo "  LoRA targets: q,k,v,o,ffn.0,ffn.2"
echo "  Frames: ${NUM_FRAMES}"
echo "  Dynamic alpha: [${FOCAL_ALPHA}, 15.0] with period=40"
echo "  Loss: L_distill + L_gen + 0.1*L_sparse"
echo "  Samples: ${NUM_SAMPLES}"
echo "  Save dir: ${SAVE_DIR}"
echo ""

# Check dependencies
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "Error: Train script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

cd ${PROJECT_ROOT}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# ========== GPU Detection ==========
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected ${NUM_GPUS} NVIDIA GPUs"
elif command -v rocm-smi &> /dev/null; then
    NUM_GPUS=$(rocm-smi --showid | grep -c "GPU")
    echo "Detected ${NUM_GPUS} AMD GPUs"
else
    echo "Error: No GPU detected!"
    exit 1
fi

USE_GPUS=${USE_GPUS:-8}
if [ ${NUM_GPUS} -lt ${USE_GPUS} ]; then
    USE_GPUS=${NUM_GPUS}
fi
echo "Using ${USE_GPUS} GPUs"

# ========== Accelerate Configuration ==========
ACCELERATE_CONFIG_DIR="${PROJECT_ROOT}/configs/accelerate"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_DIR}/default_config.yaml"
mkdir -p ${ACCELERATE_CONFIG_DIR}

cat > ${ACCELERATE_CONFIG_FILE} << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: ${USE_GPUS}
rdzv_backend: static
same_network: true
use_cpu: false
EOF

export ACCELERATE_CONFIG_FILE=${ACCELERATE_CONFIG_FILE}

# ========== Logging ==========
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p ${LOG_DIR} ${SAVE_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_stage2_${TIMESTAMP}.log"
echo "Log file: ${LOG_FILE}"
echo ""

# ========== Start Training ==========
echo "Starting Stage II training..."
echo "Start time: $(date)"

TRAIN_START=$(date +%s)

accelerate launch \
    --config_file ${ACCELERATE_CONFIG_FILE} \
    --num_processes ${USE_GPUS} \
    --num_machines 1 \
    --mixed_precision bf16 \
    ${TRAIN_SCRIPT} \
    --model_base_path ${MODEL_BASE_PATH} \
    --adapter_checkpoint ${ADAPTER_CHECKPOINT} \
    --non_text_json ${NON_TEXT_JSON} \
    --clean_dirs ${CLEAN_DIRS} \
    --subtitle_dirs ${SUBTITLE_DIRS} \
    --num_samples ${NUM_SAMPLES} \
    --num_frames ${NUM_FRAMES} \
    --random_seed ${RANDOM_SEED} \
    --lora_rank ${LORA_RANK} \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --focal_loss_alpha ${FOCAL_ALPHA} \
    --temporal_loss_weight ${TEMPORAL_WEIGHT} \
    --mask_dilation_kernel 3 \
    --enable_random_blackout \
    --use_uniform_timestep_sampling \
    --use_custom_loss \
    --output_path ${SAVE_DIR} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUM} \
    --max_grad_norm 1.0 \
    --save_interval 25 \
    --log_interval 1 \
    --log_detail_interval 1 \
    --vis_interval 10 \
    2>&1 | tee ${LOG_FILE}

TRAIN_EXIT_CODE=$?
TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))

echo ""
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "Stage II training completed in $((TRAIN_DURATION / 3600))h $(((TRAIN_DURATION % 3600) / 60))m."
    echo "Checkpoint directory: ${SAVE_DIR}"
    echo ""
    echo "Next step: Run inference"
    echo "  bash ${PROJECT_ROOT}/scripts/inference.sh"
else
    echo "Stage II training failed (exit code: ${TRAIN_EXIT_CODE})"
    echo "Check log: ${LOG_FILE}"
fi

exit ${TRAIN_EXIT_CODE}
