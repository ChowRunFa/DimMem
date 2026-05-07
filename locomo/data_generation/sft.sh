export WANDB_API_KEY='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'

NPROC_PER_NODE=8 swift sft \
  --model /data/aios-weights/Qwen/Qwen3-4B \
  --tuner_type lora \
  --dataset /data/qwt/projects/HiDimMem/locomo_results/sft_memory_extraction.jsonl \
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
  --run_name qwen3-4b-memory-extract-lora
