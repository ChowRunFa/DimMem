#!/bin/bash
# UltraChat 5000条 Window-15 完整训练+评测流程
# Phase 1: 数据已准备 (format_ultrachat_windows.py)
# Phase 2: GPT-5.4 记忆标注
# Phase 3: 构建 SFT 数据集
# Phase 4: LoRA SFT 训练
# Phase 5: 合并 LoRA + vLLM 部署
# Phase 6: 记忆构建
# Phase 7: 三路检索 + QA + Judge
set -e

# ===== 路径配置 =====
export WANDB_API_KEY='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'

BASE_DIR="/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/DimMem/evaluation/dimmem/locomo"
SCRIPT_DIR="${BASE_DIR}/data_generation"
RESULTS_DIR="${BASE_DIR}/results"

ULTRACHAT_WINDOWS="${RESULTS_DIR}/ultrachat_windows_15"
ULTRACHAT_MEMORIES="${RESULTS_DIR}/ultrachat_windows_15_memories"
SFT_OUTPUT="${RESULTS_DIR}/sft_ultrachat_5000_w15.jsonl"

STUDENT_MODEL="/data/aios-weights/Qwen/Qwen3-4B"
SFT_MERGED_MODEL="/data/qwt/projects/HiDimMem/locomo_results/Qwen3-4B-sft-ultrachat-w15"

# API 配置
API_BASE="https://models-proxy.stepfun-inc.com/v1"
API_KEY="ak-ic8y499r1fkx40brlq70z3dlata25c0b"
TEACHER_MODEL="gpt-5.4"

# 评测配置
EVAL_MODEL_NAME="ultrachat-w15-sft"
VLLM_BASE_PORT=7790
VLLM_NUM_INSTANCES=8

echo "=========================================="
echo "Phase 2: GPT-5.4 记忆标注 (5000 windows)"
echo "=========================================="

cd ${SCRIPT_DIR}
/root/miniconda3/envs/roll_env/bin/python generate_ultrachat_memories.py \
    --input-dir "${ULTRACHAT_WINDOWS}" \
    --output-dir "${ULTRACHAT_MEMORIES}" \
    --base-url "${API_BASE}" \
    --api-key "${API_KEY}" \
    --model "${TEACHER_MODEL}" \
    --concurrency 16 \
    --temperature 0.1

echo "=========================================="
echo "Phase 3: 构建 SFT 数据集"
echo "=========================================="

/root/miniconda3/envs/roll_env/bin/python build_ultrachat_sft_dataset.py \
    --conv-dir "${ULTRACHAT_WINDOWS}" \
    --mem-dir "${ULTRACHAT_MEMORIES}" \
    --output "${SFT_OUTPUT}"

echo "SFT 数据集行数:"
wc -l "${SFT_OUTPUT}"

echo "=========================================="
echo "Phase 4: LoRA SFT 训练 (ms-swift, 8 GPU)"
echo "=========================================="

NPROC_PER_NODE=8 /root/miniconda3/envs/ms-swift/bin/swift sft \
  --model ${STUDENT_MODEL} \
  --tuner_type lora \
  --dataset ${SFT_OUTPUT} \
  --load_from_cache_file false \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_checkpointing true \
  --max_length 12288 \
  --save_steps 200 \
  --logging_steps 10 \
  --report_to wandb \
  --run_name qwen3-4b-ultrachat-w15-5000

# 找到最新的 checkpoint
SFT_OUTPUT_DIR=$(find ${SCRIPT_DIR}/output/Qwen3-4B -maxdepth 1 -type d -name "v*" | sort | tail -1)
SFT_CKPT=$(find ${SFT_OUTPUT_DIR} -maxdepth 1 -type d -name "checkpoint-*" | sort -t- -k2 -n | tail -1)
echo "SFT checkpoint: ${SFT_CKPT}"

echo "=========================================="
echo "Phase 5: 合并 LoRA + 部署 vLLM"
echo "=========================================="

/root/miniconda3/envs/ms-swift/bin/swift export \
  --model ${STUDENT_MODEL} \
  --adapters ${SFT_CKPT} \
  --output_dir ${SFT_MERGED_MODEL}

echo "合并后模型: ${SFT_MERGED_MODEL}"

# 启动 8 个 vLLM 进程
echo "启动 8 个 vLLM 实例..."
VLLM_PIDS=()
for i in $(seq 0 $((VLLM_NUM_INSTANCES-1))); do
    PORT=$((VLLM_BASE_PORT + i))
    CUDA_VISIBLE_DEVICES=$i /root/miniconda3/envs/roll_env/bin/python -m vllm.entrypoints.openai.api_server \
        --model ${SFT_MERGED_MODEL} \
        --port ${PORT} \
        --dtype bfloat16 \
        --max-model-len 40960 \
        --gpu-memory-utilization 0.90 \
        --trust-remote-code \
        --disable-log-requests &
    VLLM_PIDS+=($!)
    echo "  GPU $i → port ${PORT} (PID: ${VLLM_PIDS[-1]})"
done

# 等待所有实例就绪
echo "等待 vLLM 实例启动..."
for i in $(seq 0 $((VLLM_NUM_INSTANCES-1))); do
    PORT=$((VLLM_BASE_PORT + i))
    for attempt in $(seq 1 120); do
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "  Port ${PORT} ready!"
            break
        fi
        if [ $attempt -eq 120 ]; then
            echo "ERROR: vLLM port ${PORT} 启动超时"
            for pid in "${VLLM_PIDS[@]}"; do kill $pid 2>/dev/null; done
            exit 1
        fi
        sleep 5
    done
done
echo "所有 vLLM 实例就绪！"

echo "=========================================="
echo "Phase 6: 记忆构建 (LoCoMo 评测数据)"
echo "=========================================="

# 生成端口列表
PORTS=""
for i in $(seq 0 $((VLLM_NUM_INSTANCES-1))); do
    if [ -n "$PORTS" ]; then PORTS="${PORTS},"; fi
    PORTS="${PORTS}$((VLLM_BASE_PORT + i))"
done

# 使用 w15 压缩数据
COMPRESSED_ROOT="${RESULTS_DIR}/segment_results/compressed/20260427_153200_by_conv_compressed_w15_o3"
MEMORY_OUTPUT="${RESULTS_DIR}/memory_results/${EVAL_MODEL_NAME}"

cd ${BASE_DIR}/src/memory_constructor
/root/miniconda3/envs/roll_env/bin/python build_memories_opd_parallel.py \
    --compressed-root "${COMPRESSED_ROOT}" \
    --output-root "${MEMORY_OUTPUT}" \
    --ports "${PORTS}" \
    --api-key "EMPTY" \
    --model-name "${SFT_MERGED_MODEL}" \
    --max-tokens 8192 \
    --timeout 300 \
    --max-retries 3 \
    --workers 16

echo "=========================================="
echo "Phase 7a: 三路检索"
echo "=========================================="

QUERY_ROOT="${RESULTS_DIR}/query_analysis/20260427_011047"

# 结构化检索
cd ${BASE_DIR}/src/search
/root/miniconda3/envs/roll_env/bin/python run_retrieval_from_query_analysis.py \
    --query-root "${QUERY_ROOT}" \
    --memory-root "${MEMORY_OUTPUT}" \
    --output-base "${RESULTS_DIR}/retrieval_results" \
    --run-name "${EVAL_MODEL_NAME}_structure" \
    --top-n 15

# MiniLM 语义检索
/root/miniconda3/envs/roll_env/bin/python run_retrieval_minilm.py \
    --query-root "${QUERY_ROOT}" \
    --memory-root "${MEMORY_OUTPUT}" \
    --output-base "${RESULTS_DIR}/retrieval_results" \
    --run-name "${EVAL_MODEL_NAME}_minilm" \
    --top-n 15

# BM25 检索
/root/miniconda3/envs/roll_env/bin/python run_retrieval_bm25.py \
    --query-root "${QUERY_ROOT}" \
    --memory-root "${MEMORY_OUTPUT}" \
    --output-base "${RESULTS_DIR}/retrieval_results" \
    --run-name "${EVAL_MODEL_NAME}_bm25" \
    --top-n 15

echo "=========================================="
echo "Phase 7b: QA (gpt-4.1-mini, 三路合并 top15)"
echo "=========================================="

cd ${BASE_DIR}/src/qa
QA_RUN_NAME="${EVAL_MODEL_NAME}_qa_3routes_top15"

/root/miniconda3/envs/roll_env/bin/python run_qa_from_three_retrievals.py \
    --query-root "${QUERY_ROOT}" \
    --structured-root "${RESULTS_DIR}/retrieval_results/${EVAL_MODEL_NAME}_structure" \
    --minilm-root "${RESULTS_DIR}/retrieval_results/${EVAL_MODEL_NAME}_minilm" \
    --bm25-root "${RESULTS_DIR}/retrieval_results/${EVAL_MODEL_NAME}_bm25" \
    --output-base "${RESULTS_DIR}/qa_results" \
    --run-name "${QA_RUN_NAME}" \
    --top-n-each 15 \
    --max-merged 45 \
    --base-url "${API_BASE}" \
    --api-key "${API_KEY}" \
    --model-name "gpt-4.1-mini" \
    --timeout 120 \
    --max-retries 3

echo "=========================================="
echo "Phase 7c: Judge (gpt-4.1-mini)"
echo "=========================================="

cd ${BASE_DIR}/src/judge
JUDGE_RUN_NAME="${EVAL_MODEL_NAME}_judge"

# Judge 按 conv 并行
find "${RESULTS_DIR}/qa_results/${QA_RUN_NAME}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" \
  | while read -r conv; do
      [[ -f "/data/qwt/projects/data/locomo10_by_conv/$conv/locomo10.json" ]] || continue
      echo "$conv"
    done \
  | xargs -I{} -P 3 bash -lc \
    "/root/miniconda3/envs/roll_env/bin/python \"${BASE_DIR}/src/judge/run_judge_from_qa.py\" \
      --qa-root \"${RESULTS_DIR}/qa_results/${QA_RUN_NAME}\" \
      --conv-json \"/data/qwt/projects/data/locomo10_by_conv/{}/locomo10.json\" \
      --conv-name \"{}\" \
      --output-base \"${RESULTS_DIR}/judge_results/${JUDGE_RUN_NAME}\" \
      --run-name \"{}\" \
      --base-url \"${API_BASE}\" \
      --api-key \"${API_KEY}\" \
      --model-name \"gpt-4.1-mini\" \
      --timeout 180 \
      --max-retries 5 \
      --max-tokens 256"

echo "=========================================="
echo "关闭 vLLM 实例"
echo "=========================================="

for pid in "${VLLM_PIDS[@]}"; do
    kill $pid 2>/dev/null || true
done
wait 2>/dev/null || true
echo "vLLM 已关闭"

echo ""
echo "====== 全部完成 ======"
echo "SFT 数据: ${SFT_OUTPUT}"
echo "合并模型: ${SFT_MERGED_MODEL}"
echo "评测结果: ${RESULTS_DIR}/judge_results/${EVAL_MODEL_NAME}"
