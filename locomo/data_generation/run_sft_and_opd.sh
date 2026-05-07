#!/bin/bash
# SFT 训练 + 合并 LoRA + 构建 OPD 数据 + 启动 OPD
set -e

# ===== 配置 =====
export WANDB_API_KEY='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'

STUDENT_MODEL="/data/aios-weights/Qwen/Qwen3-4B"
TEACHER_MODEL="/data/aios-weights/Qwen/Qwen3-32B"
SFT_DATA="/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/DimMem/evaluation/dimmem/locomo/results/sft_memory_extraction_qwen3_32b.jsonl"
SFT_MERGED_MODEL="/data/qwt/projects/HiDimMem/locomo_results/Qwen3-4B-sft-merged-qwen3-32b"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROLL_DIR="/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/roll/ROLL"

echo "=========================================="
echo "Phase 1: SFT 训练 (ms-swift, 8 GPU)"
echo "=========================================="

NPROC_PER_NODE=8 /root/miniconda3/envs/ms-swift/bin/swift sft \
  --model ${STUDENT_MODEL} \
  --tuner_type lora \
  --dataset ${SFT_DATA} \
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
  --run_name qwen3-4b-locomo-sft-from-qwen3-32b

# 找到最新的 checkpoint
SFT_OUTPUT_DIR=$(find ${SCRIPT_DIR}/output/Qwen3-4B -maxdepth 1 -type d -name "v*" | sort | tail -1)
SFT_CKPT=$(find ${SFT_OUTPUT_DIR} -maxdepth 1 -type d -name "checkpoint-*" | sort -t- -k2 -n | tail -1)
echo "SFT checkpoint: ${SFT_CKPT}"

echo "=========================================="
echo "Phase 2: 合并 LoRA"
echo "=========================================="

/root/miniconda3/envs/ms-swift/bin/swift export \
  --model ${STUDENT_MODEL} \
  --adapters ${SFT_CKPT} \
  --output_dir ${SFT_MERGED_MODEL}

echo "合并后模型: ${SFT_MERGED_MODEL}"

echo "=========================================="
echo "Phase 3: 构建 OPD 数据"
echo "=========================================="

/root/miniconda3/envs/roll_env/bin/python ${SCRIPT_DIR}/build_locomo_opd_jsonl.py \
  --input ${SFT_DATA} \
  --output ${ROLL_DIR}/examples/locomo-opd/data/locomo_opd_messages.jsonl

echo "=========================================="
echo "Phase 4: OPD 训练 (ROLL)"
echo "=========================================="

cd ${ROLL_DIR}
export PYTHONPATH="${ROLL_DIR}:${PYTHONPATH}"
export RAY_DEDUP_LOGS=1

/root/miniconda3/envs/roll_env/bin/python examples/start_onpolicy_distill_pipeline.py \
  --config_path "examples/locomo-opd" \
  --config_name "locomo_opd_config"

echo "====== 全部完成 ======"
