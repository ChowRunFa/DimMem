# LongMemEval Memory Constructor

本目录用于 LongMemEval 的结构化记忆构建。它承接 `segmenter/` 或 `compressor/` 产生的窗口文本，通过 LLM 抽取 memories，并统一归一化 memory 的 `dimension` 字段。

## 文件

| 文件 | 作用 |
|---|---|
| `run_extract_windows_with_en_prompt.py` | 记忆抽取 helper：构造 prompt、调用模型、解析 JSON、归一化 memory。 |
| `__init__.py` | 包标记文件。 |

## 与旧路径的关系

旧文件：

```text
projects/submit/longmemeval/run_extract_windows_with_en_prompt.py
```

现在仅保留为兼容 wrapper，新代码请使用：

```text
projects/submit/longmemeval/memory_constructor/run_extract_windows_with_en_prompt.py
```

## 输入来源

通常来自：

```text
results/segments/raw/<run_name>/<question_type>/<sample_id>/windows/window_0000.json
```

或压缩后：

```text
results/segments/compressed/<run_name>/<question_type>/<sample_id>/windows/window_0000.json
```

窗口文本格式由 `segmenter/` 生成，形如：

```text
[2023-06-25T13:22:00, Sun] 1.User: ...
[2023-06-25T13:22:00.500000, Sun] 2.User: ...
```

## 主要 helper

`run_extract_windows_with_en_prompt.py` 提供：

- `_build_prompt(...)`
- `_call_chat(...)`
- `_safe_json_fragment(...)`
- `_extract_text(...)`
- `_source_time_by_id_from_dialogue(...)`
- `_build_session_map_from_window(...)`
- `_normalize_memory_entry(...)`

其中 `_normalize_memory_entry(...)` 已使用公共模型：

```python
from models import DimensionMemory
```

来规范：

```json
{
  "dimension": {
    "memory_type": "episodic",
    "time": "",
    "location": "",
    "reason": "",
    "purpose": "",
    "keywords": []
  }
}
```

## 公共模型

记忆维度统一使用：

```python
from models import DimensionMemory

dim = DimensionMemory.from_dict(row.get("dimension"))
normalized = dim.to_dict()
```

这样后续 `search/`、`update/`、`qa_judge/` 不需要重复手写字段清洗逻辑。
