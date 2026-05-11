# LongMemEval Submit Pipeline

本目录按功能拆分 LongMemEval 的端到端流程：切分窗口、压缩窗口、构建结构化记忆、解析问题、检索、QA/Judge。

## 目录结构

| 目录 / 文件 | 作用 |
|---|---|
| `segmenter/` | 将 LongMemEval 原始样本切成 raw sliding windows，并生成 assistant reply 索引。 |
| `compressor/` | 使用 LLMLingua-2 压缩 raw windows。 |
| `memory_constructor/` | 从窗口文本中抽取结构化 memories，并归一化 `dimension` 字段。 |
| `query_parser/` | 将问题解析为结构化 query analysis，输出 `parsed.json`。 |
| `models/` | 公共数据模型，包含 `DimensionMemory` 和 `ParsedQuery`。 |
| `search/` | BM25 / structured / embedding / fusion 检索。 |
| `update/` | memory update / consolidate 相关逻辑。 |
| `qa_judge/`, `run_qa_judge_from_retrieval.py` | 基于检索结果回答并评测。 |
| `retrieve_from_parsed_query.py` | 从 `parsed.json` 和 memories 运行检索。 |

## 推荐流程

### 1. 切 raw windows

```bash
python projects/submit/longmemeval/segmenter/build_raw_segments.py \
  --input-path data/longmemeval_s_cleaned.json \
  --output-root results/segments/raw
```

输出：

```text
results/segments/raw/<run_name>/<question_type>/<sample_id>/windows/window_0000.json
```

### 2. 压缩 windows（可选）

```bash
python projects/submit/longmemeval/compressor/build_compressed_segments.py \
  --raw-run-root results/segments/raw/<run_name> \
  --output-root results/segments/compressed
```

### 3. 构建结构化记忆

记忆构建相关 helper 已放入：

```text
projects/submit/longmemeval/memory_constructor/
```

核心文件：

```text
memory_constructor/run_extract_windows_with_en_prompt.py
```

该文件负责：

- 构造 `LONGMEMEVAL_STRUCTURED_MEMORY_EXTRACTION_PROMPT`
- 调用 OpenAI-compatible chat API
- 解析 LLM JSON 输出
- 使用 `models.DimensionMemory` 归一化 memory 的 `dimension`

> 根目录的 `run_extract_windows_with_en_prompt.py` 仅保留为兼容 import 的 wrapper；新代码请优先使用 `memory_constructor/` 下的路径。

### 4. 解析 query

```bash
python projects/submit/longmemeval/query_parser/run_query_analysis.py \
  --input-root data/longmemeval_s_cleaned.json \
  --output-base results/query_analysis
```

输出：

```text
results/query_analysis/<run_name>/<question_type>/<sample_id>/parsed.json
```

`parsed.json` 会通过 `models.ParsedQuery` 在检索阶段统一读取和使用。

### 5. 检索

```bash
python projects/submit/longmemeval/retrieve_from_parsed_query.py \
  --query-parsed results/query_analysis/<run_name>/<question_type>/<sample_id>/parsed.json \
  --memory-dir results/memories/<question_type>/<sample_id> \
  --output-root results/retrieval
```

检索模块支持：

- `bm25`
- `structured`
- `minilm`
- `fusion / rrf_hybrid`

## 公共模型

`models/` 中提供两个公共类，避免各模块直接散落读取 dict 字段：

- `DimensionMemory`：封装 memory.dimension
- `ParsedQuery`：封装 query parser 输出

示例：

```python
from models import DimensionMemory, ParsedQuery

memory_dim = DimensionMemory.from_dict(memory.get("dimension"))
query = ParsedQuery.from_dict(parsed_json)
```

## 说明

- 新增的 `memory_constructor/` 和 `query_parser/` 与 LoCoMo 目录结构保持一致，便于维护。
- 旧的根目录 `run_extract_windows_with_en_prompt.py` 仍可被 import，但不建议继续作为主路径使用。
