#!/bin/bash

# ================================================================
# CLEAR Stage I: Self-Supervised Prior Learning
# 
# Trains dual encoders with disentangled feature learning to extract
# coarse occlusion guidance from paired videos.
#
# Key features:
# - Dual ResNet-50 encoders (E_sub, E_content)
# - Orthogonality constraint for feature independence
# - Multi-scale FPN fusion for small subtitle detection
# - Self-supervised pseudo-labels from pixel differences
#
# Loss function:
#   L_stage1 = L_ortho + 0.5 * L_adv + L_region + 0.1 * L_recon
#
# Reference: Section 3.2 in the CLEAR paper
# ================================================================

set -e

# ========== Configuration ==========
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/configs/stage1_config.yaml"

echo "=========================================="
echo "CLEAR Stage I: Self-Supervised Prior Learning"
echo "=========================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Config file: ${CONFIG_FILE}"
echo ""

# Check config file
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
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
echo "Using ${USE_GPUS} GPUs for training"

# ========== Distributed Configuration ==========
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29504}
export WORLD_SIZE=${USE_GPUS}

# ========== Logging ==========
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_stage1_${TIMESTAMP}.log"

echo ""
echo "Training Configuration:"
echo "  Backbone: ResNet-50 (dual encoders)"
echo "  Loss: L_ortho + 0.5*L_adv + L_region + 0.1*L_recon"
echo "  Multi-scale FPN: enabled (layer2 fusion)"
echo "  Learning rate: as specified in config"
echo "  Log file: ${LOG_FILE}"
echo ""

# ========== Start Training ==========
echo "Starting Stage I training..."
echo "Start time: $(date)"

TRAIN_START=$(date +%s)

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${USE_GPUS} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ${PROJECT_ROOT}/train_stage1.py \
    --config ${CONFIG_FILE} \
    2>&1 | tee ${LOG_FILE}

TRAIN_EXIT_CODE=$?
TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))

echo ""
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "Stage I training completed in $((TRAIN_DURATION / 60)) minutes."
    echo ""
    echo "Next step: Run Stage II training"
    echo "  bash ${PROJECT_ROOT}/scripts/train_stage2.sh"
else
    echo "Stage I training failed (exit code: ${TRAIN_EXIT_CODE})"
    echo "Check log: ${LOG_FILE}"
fi

exit ${TRAIN_EXIT_CODE}
