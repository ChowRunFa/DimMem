# LongMemEval Query Parser

本目录用于把 LongMemEval 的自然语言问题解析成结构化 query analysis，供 `search/` 和 `retrieve_from_parsed_query.py` 使用。

## 文件

| 文件 | 作用 |
|---|---|
| `run_query_analysis.py` | 批量解析 LongMemEval 问题，输出每题的 `parsed.json`。 |
| `__init__.py` | 包标记文件。 |

## 输入

支持两种输入：

1. 单个 JSON 文件：

```text
data/longmemeval_s_cleaned.json
```

2. 一个目录，目录下包含：

```text
longmemeval_s_cleaned__*.json
```

每条样本主要读取字段：

- `question`
- `question_id`
- `question_type`

如果文件名形如 `longmemeval_s_cleaned__<question_type>.json`，会优先使用文件名中的 `<question_type>`。

## 输出结构

```text
<output-base>/<run-name>/
├── run_manifest.json
├── status.json
├── summary.json
└── <question_type>/
    └── <sample_id>/
        ├── input.json
        ├── prompt.txt
        ├── raw_response.json
        ├── raw_response.txt
        ├── parsed.json
        └── result.json
```

其中 `sample_id` 形如：

```text
0000_<question_id>
```

## 运行示例

```bash
python projects/submit/longmemeval/query_parser/run_query_analysis.py \
  --input-root data/longmemeval_s_cleaned.json \
  --output-base results/query_analysis \
  --base-url http://127.0.0.1:7790/v1 \
  --api-key EMPTY \
  --model-name qwen3-30b-a3b
```

调试参数：

```bash
python projects/submit/longmemeval/query_parser/run_query_analysis.py \
  --input-root data/longmemeval_s_cleaned.json \
  --output-base results/query_analysis \
  --max-convs 1 \
  --max-questions-per-conv 5
```

## `parsed.json` 格式

LLM 应输出类似：

```json
{
  "parse_mode": "structured",
  "query_anchor": "...",
  "dimension": {
    "target_memory_type": ["episodic"],
    "time": "",
    "location": "",
    "keywords": ["..."]
  },
  "answer_dim": "content"
}
```

代码中请优先使用公共模型读取：

```python
from models import ParsedQuery

query = ParsedQuery.from_dict(parsed_json)
```
