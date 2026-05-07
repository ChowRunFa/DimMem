#!/bin/bash
# Step 0: 用 Qwen3-32B (vLLM) 对所有对话窗口生成记忆抽取结果
# 然后构建 SFT 数据集

set -e

# ===== 配置 =====
CONDA_ENV="roll_env"
MODEL_PATH="/data/aios-weights/Qwen/Qwen3-32B"
VLLM_PORT=8199
TP_SIZE=8

CONV_DIR="/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/DimMem/evaluation/dimmem/locomo/results/generated_conversation"
MEM_OUTPUT_DIR="/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/DimMem/evaluation/dimmem/locomo/results/generated_conversation_memories_qwen3_32b"
SFT_OUTPUT="/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/DimMem/evaluation/dimmem/locomo/results/sft_memory_extraction_qwen3_32b.jsonl"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "Step 1: 启动 vLLM Server (Qwen3-32B, TP=$TP_SIZE)"
echo "=========================================="

# 启动 vLLM 服务（后台）
/root/miniconda3/envs/${CONDA_ENV}/bin/python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --tensor-parallel-size ${TP_SIZE} \
    --dtype bfloat16 \
    --port ${VLLM_PORT} \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --disable-log-requests &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# 等待服务就绪
echo "等待 vLLM 服务启动..."
for i in $(seq 1 120); do
    if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM 服务就绪！"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: vLLM 启动超时"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 5
done

echo "=========================================="
echo "Step 2: 教师推理 - 生成结构化记忆"
echo "=========================================="

cd ${SCRIPT_DIR}
/root/miniconda3/envs/${CONDA_ENV}/bin/python generate_structured_memories.py \
    --input-dir "${CONV_DIR}" \
    --output-dir "${MEM_OUTPUT_DIR}" \
    --base-url "http://localhost:${VLLM_PORT}/v1" \
    --api-key "EMPTY" \
    --model "${MODEL_PATH}" \
    --concurrency 32 \
    --temperature 0.1

echo "=========================================="
echo "Step 3: 构建 SFT 数据集"
echo "=========================================="

/root/miniconda3/envs/${CONDA_ENV}/bin/python build_sft_dataset.py \
    --conv-dir "${CONV_DIR}" \
    --mem-dir "${MEM_OUTPUT_DIR}" \
    --output "${SFT_OUTPUT}"

echo "=========================================="
echo "Step 4: 关闭 vLLM 服务"
echo "=========================================="
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "vLLM 已关闭"

echo ""
echo "====== 完成 ======"
echo "SFT 数据集: ${SFT_OUTPUT}"
echo "下一步: 运行 run_sft_and_opd.sh"
