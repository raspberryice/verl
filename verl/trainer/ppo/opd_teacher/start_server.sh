#!/bin/bash
# Start OPD Teacher Server
# This script launches a teacher model server for On-Policy Distillation.
# The server computes full log probabilities on student-generated trajectories.

set -e

# Configuration
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"Qwen/Qwen3-32B"}
PORT=${PORT:-15555}
TP=${TP:-2}  # Tensor parallelism
GPU_MEM=${GPU_MEM:-0.9}
DTYPE=${DTYPE:-"bfloat16"}
MAX_LEN=${MAX_LEN:-16384}

# Optional: Set visible GPUs for teacher server
# export CUDA_VISIBLE_DEVICES=0,1  # Separate GPUs from student training

# Fix CUDA multiprocessing error: vLLM must use 'spawn' instead of 'fork'
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "============================================"
echo "Starting OPD Teacher Server"
echo "============================================"
echo "Model: ${TEACHER_MODEL_PATH}"
echo "Port: ${PORT}"
echo "TP: ${TP}"
echo "GPU Memory: ${GPU_MEM}"
echo "Dtype: ${DTYPE}"
echo "Max Length: ${MAX_LEN}"
echo "============================================"

python3 -m verl.trainer.ppo.opd_teacher.server \
    --model_path ${TEACHER_MODEL_PATH} \
    --port ${PORT} \
    --tp ${TP} \
    --gpu_memory_utilization ${GPU_MEM} \
    --dtype ${DTYPE} \
    --max_model_len ${MAX_LEN} \
    "$@"
