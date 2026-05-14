# LoCoMo Memory Constructor

从分段窗口结果构建结构化记忆。

## 规则

- Prompt 默认使用英文版：
  - `locomo/prompts/prompts.py`
  - `LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT`
- 若是每个 conv 的第一个 window（`window_index == 0`）：
  - `OverlappingContextRules` 置空，`overlap_count` 为 0
- 其他 window：
  - 使用 `OverlappingContextRules` 原文，`overlap_count` 由 `--overlap` 参数指定

## 输入

- `--compressed-root` 指向分段目录（raw 或 compressed 均可）：
  - `.../results/locomo_segments/raw/<run_name>`

## 输出

每条记录目录：

- `experiment_config.json`
- `summary.json`
- `all_memories.json`
- `window_0000/`
  - `dialogue_input.txt`
  - `extract_prompt.txt`
  - `raw_response.json`
  - `raw_response.txt`
  - `parsed_payload.json`（可选）
  - `normalized_memories.json`
  - `result.json`

运行级文件：

- `status.json`（实时进度）
- `experiment_config.json`
- `failures.json`（如有）

## 用法

批量模式（所有 record 顺序处理）：

```bash
python locomo/memory_constructor/run_batch_extract.py \
  --compressed-root ./results/locomo_segments/raw/<run_name> \
  --output-root ./results/locomo_memory \
  --run-name <run_name> \
  --overlap 5 \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b
```

单条 record 并行模式：

```bash
python locomo/memory_constructor/extract_single_record.py \
  --record-dir ./results/locomo_segments/raw/<run_name>/<conv>/ \
  --output-record-dir ./results/locomo_memory/<conv>/ \
  --overlap 5 \
  --base-urls http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b \
  --workers 4
```

断点续跑默认开启：重复用同一个 `--run-name` 会自动跳过已有 `summary.json` 的记录。
