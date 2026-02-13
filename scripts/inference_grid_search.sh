#!/bin/bash

# ================================================================
# CLEAR Inference Grid Search
#
# Runs inference across different hyperparameter configurations:
# - Checkpoint steps
# - LoRA scales
# - Denoising steps
#
# Supports multi-GPU parallel execution.
#
# Usage:
#   bash scripts/inference_grid_search.sh
# ================================================================

set -e

# ========== Configuration ==========
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INFERENCE_SCRIPT="${PROJECT_ROOT}/inference.py"

# Model paths (MODIFY THESE)
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/path/to/Wan2.1-Fun-V1.1-1.3B-Control}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/stage2_lora}"

# Test videos directory (MODIFY THIS)
TESTVIDEOS_DIR="${TESTVIDEOS_DIR:-${PROJECT_ROOT}/test_videos}"

# ========== Grid Search Parameters ==========
# Checkpoint steps to evaluate
CHECKPOINT_STEPS=(350)

# LoRA strength values
LORA_SCALES=(1.0)

# Denoising steps
NUM_STEPS_LIST=(5 10)

# Test video files (list your test videos here)
TEST_VIDEOS=(
    "test1.mp4"
    "test2.mp4"
)

# ========== Fixed Parameters ==========
CFG_SCALE=1.0
CHUNK_SIZE=81
CHUNK_OVERLAP=16
LORA_RANK=64
USE_SLIDING_WINDOW=true

# GPU configuration
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    if [ ${NUM_GPUS} -gt 8 ]; then
        NUM_GPUS=8
    fi
else
    NUM_GPUS=${NUM_GPUS:-1}
fi
GPU_IDS=($(seq 0 $((NUM_GPUS - 1))))
echo "Using ${NUM_GPUS} GPUs: ${GPU_IDS[@]}"

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="${PROJECT_ROOT}/experiments/grid_search_${TIMESTAMP}"
mkdir -p ${OUTPUT_ROOT}/logs

cd ${PROJECT_ROOT}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ========== Run Experiment Function ==========
run_experiment() {
    local GPU_ID=$1
    local STEP=$2
    local LORA_SCALE=$3
    local NUM_STEPS=$4
    local VIDEO_FILE=$5
    local EXP_ID=$6
    local TOTAL=$7
    
    local CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoint_${STEP}_lora.pt"
    local VIDEO_PATH="${TESTVIDEOS_DIR}/${VIDEO_FILE}"
    
    if [ ! -f "${CHECKPOINT_PATH}" ] || [ ! -f "${VIDEO_PATH}" ]; then
        echo "[GPU${GPU_ID}] Skipping: missing checkpoint or video"
        return 1
    fi
    
    local SCALE_STR=$(echo ${LORA_SCALE} | sed 's/\./_/g')
    local EXP_NAME="step${STEP}_scale${SCALE_STR}_steps${NUM_STEPS}"
    local VIDEO_NAME=$(basename "${VIDEO_FILE}" .mp4)
    local OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}/${VIDEO_NAME}"
    mkdir -p "${OUTPUT_DIR}"
    
    local LOG_FILE="${OUTPUT_ROOT}/logs/${EXP_NAME}_${VIDEO_NAME}.log"
    
    echo "[GPU${GPU_ID}] Task ${EXP_ID}/${TOTAL}: step=${STEP} scale=${LORA_SCALE} steps=${NUM_STEPS} video=${VIDEO_FILE}"
    
    local SLIDING_FLAG=""
    if [ "${USE_SLIDING_WINDOW}" = "true" ]; then
        SLIDING_FLAG="--use_sliding_window"
    fi
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${INFERENCE_SCRIPT} \
        --model_base_path ${MODEL_BASE_PATH} \
        --lora_checkpoint ${CHECKPOINT_PATH} \
        --lora_rank ${LORA_RANK} \
        --lora_scale ${LORA_SCALE} \
        --input_video "${VIDEO_PATH}" \
        --output_dir ${OUTPUT_DIR} \
        --prompt "Remove the text overlays and subtitles from this video while preserving the original background content and maintaining temporal consistency across frames." \
        --num_steps ${NUM_STEPS} \
        --cfg_scale ${CFG_SCALE} \
        --chunk_size ${CHUNK_SIZE} \
        --chunk_overlap ${CHUNK_OVERLAP} \
        ${SLIDING_FLAG} \
        --create_comparison \
        --copy_source \
        > ${LOG_FILE} 2>&1
    
    echo "[GPU${GPU_ID}] Task ${EXP_ID}/${TOTAL} completed"
}

# ========== Execute Grid Search ==========
TOTAL_START=$(date +%s)

# Collect all tasks
declare -a TASKS
EXP_ID=0
for STEP in "${CHECKPOINT_STEPS[@]}"; do
    for SCALE in "${LORA_SCALES[@]}"; do
        for STEPS in "${NUM_STEPS_LIST[@]}"; do
            for VIDEO in "${TEST_VIDEOS[@]}"; do
                EXP_ID=$((EXP_ID + 1))
                TASKS+=("${STEP}:${SCALE}:${STEPS}:${VIDEO}:${EXP_ID}")
            done
        done
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo "Total tasks: ${TOTAL_TASKS}"
echo "Output: ${OUTPUT_ROOT}"
echo ""

# Parallel execution with GPU pool
TASK_IDX=0
declare -a PIDS

for GPU_IDX in "${!GPU_IDS[@]}"; do
    if [ ${TASK_IDX} -lt ${TOTAL_TASKS} ]; then
        IFS=':' read -r STEP SCALE STEPS VIDEO EID <<< "${TASKS[$TASK_IDX]}"
        (run_experiment ${GPU_IDS[$GPU_IDX]} ${STEP} ${SCALE} ${STEPS} "${VIDEO}" ${EID} ${TOTAL_TASKS}) &
        PIDS[$GPU_IDX]=$!
        TASK_IDX=$((TASK_IDX + 1))
    fi
done

while [ ${TASK_IDX} -lt ${TOTAL_TASKS} ]; do
    for GPU_IDX in "${!GPU_IDS[@]}"; do
        if [ -n "${PIDS[$GPU_IDX]}" ] && ! kill -0 ${PIDS[$GPU_IDX]} 2>/dev/null; then
            wait ${PIDS[$GPU_IDX]} 2>/dev/null
            if [ ${TASK_IDX} -lt ${TOTAL_TASKS} ]; then
                IFS=':' read -r STEP SCALE STEPS VIDEO EID <<< "${TASKS[$TASK_IDX]}"
                (run_experiment ${GPU_IDS[$GPU_IDX]} ${STEP} ${SCALE} ${STEPS} "${VIDEO}" ${EID} ${TOTAL_TASKS}) &
                PIDS[$GPU_IDX]=$!
                TASK_IDX=$((TASK_IDX + 1))
            else
                PIDS[$GPU_IDX]=""
            fi
        fi
    done
    sleep 1
done

# Wait for remaining tasks
for GPU_IDX in "${!GPU_IDS[@]}"; do
    if [ -n "${PIDS[$GPU_IDX]}" ]; then
        wait ${PIDS[$GPU_IDX]} 2>/dev/null
    fi
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo ""
echo "=========================================="
echo "Grid search completed!"
echo "Total time: $((TOTAL_DURATION / 60)) minutes"
echo "Results: ${OUTPUT_ROOT}/"
echo "=========================================="
