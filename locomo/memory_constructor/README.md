# LoCoMo Memory Constructor

从压缩窗口结果构建结构化记忆，输出到：

`evaluation/dimmem/locomo/results/memory_results/<run_name>/...`

## 规则

- Prompt 默认使用英文版：
  - `locomo/src/prompts/prompts.py`
  - `LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT`
- 若是每个 conv 的第一个 window（`window_index == 0`）：
  - `OverlappingContextRules` 置空
- 其他 window：
  - 使用 `OverlappingContextRules` 原文

## 输入

- `--compressed-root` 指向：
  - `.../results/segment_results/compressed/<timestamp>`

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

```bash
/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/miniconda3/envs/dimmem_v2/bin/python \
  /mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/DimMem/evaluation/dimmem/locomo/src/memory_constructor/build_memories_from_compressed.py \
  --compressed-root /mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/DimMem/evaluation/dimmem/locomo/results/segment_results/compressed/20260426_by_conv_compressed \
  --run-name 20260426_locomo_extract \
  --base-url http://127.0.0.1:7790/v1 \
  --model-name qwen3-30b-a3b
```

断点续跑默认开启：重复用同一个 `--run-name` 会自动跳过已有 `summary.json` 的记录。
