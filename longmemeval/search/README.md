# LongMemEval Search

`src/search` 提供三路检索 + 融合去重 + assistant 上下文附加功能。

## 模块概览

| 文件 | 功能 |
|------|------|
| `bm25_search.py` | BM25 关键词检索 |
| `structured_search.py` | 结构化加权打分 |
| `embedding_search.py` | MiniLM 向量相似度检索 |
| `fusion_search.py` | 三路合并 + content MD5 去重 |
| `time_constraints.py` | 时间约束解析与匹配 |
| `assistant_context.py` | assistant 回复上下文附加 |

## 1) BM25 Search (`bm25_search.py`)

自建 BM25Okapi 实现（无外部依赖）。

- **查询字段**：`query_anchor` + `dimension.keywords`
- **匹配字段**：`content` / `dimension.reason` / `dimension.purpose` / `dimension.keywords`
- 参数：`k1=1.2`, `b=0.75`

```python
from search import search_bm25
result = search_bm25(parsed_query=query, records=records, top_k=15)
# result["top_records"] -> List[Dict] with score, score_components
```

## 2) Structured Search (`structured_search.py`)

基于查询维度的加权结构化打分。

**权重分配**（自适应归一化：仅激活的维度参与归一化）：

| 维度 | 基础权重 | 说明 |
|------|---------|------|
| `memory_type` | 0.15 | 记忆类型匹配（fact/episodic/profile） |
| `time_constraint` | 0.30 | 时间约束匹配 |
| `location_constraint` | 0.20 | 地点约束匹配 |
| `keyword_phrase_match` | 0.15 | 关键词短语精确匹配 |
| `keyword_token_overlap` | 0.15 | 关键词 token 重叠率 |

```python
from search import search_structured
result = search_structured(parsed_query=query, records=records, embedding_client=client, top_k=15)
```

## 3) Embedding Search (`embedding_search.py`)

MiniLM (all-MiniLM-L6-v2) cosine similarity。

- **查询**：`query_anchor` → embedding
- **文档**：`content + reason + purpose` → embedding
- 支持 batch embedding + batch cosine similarity

```python
from search import search_embedding
result = search_embedding(parsed_query=query, records=records, embedding_client=client, top_k=15)
```

## 4) Fusion Search (`fusion_search.py`)

三路合并 + content MD5 去重。

执行顺序：`bm25 → structured → minilm`，每路各取 top_k，合并时按首次出现顺序排列，重复记录通过 `content` 的 MD5 去重。

去重后的记录保留 `fusion_sources` 字段追踪来源：

```json
{
  "retrieval_method": "bm25",
  "retrieval_rank": 1,
  "fusion_sources": [
    {"method": "bm25", "rank": 1, "score": 5.23},
    {"method": "minilm", "rank": 3, "score": 0.82}
  ]
}
```

```python
from search import search_top15_content_dedup
result = search_top15_content_dedup(parsed_query=query, records=records, embedding_client=client, top_k=15)
```

## 5) Time Constraints (`time_constraints.py`)

支持的时间约束格式：

| 模式 | 示例 |
|------|------|
| 精确日期 | `2023-05-20` |
| 月份 | `2023-05` |
| 年份 | `2023` |
| 前缀 | `on 2023-05-20`, `before 2023-06`, `after 2023-01`, `around 2023-03` |
| 区间 | `between 2023-01 and 2023-06` |

`around` 的容差：日 ±3天，月 ±31天，年 ±366天。

## 6) Assistant Context (`assistant_context.py`)

为检索到的 memory 附加对应的原始 AI 回复。

**映射链**：
```
memory.source_boundary_id
  → (window_index, source_id)  [from all_memories.json]
  → uid = w{window_index:04d}u{source_id:02d}
  → assistant_replies.json[uid].assistant_reply
```

```python
from search.assistant_context import (
    build_boundary_to_window_source,
    load_window_assistant_replies,
    attach_assistant_context,
)

# 1. 构建 boundary_id -> (window_index, source_id) 映射
boundary_index, source_record_dir = build_boundary_to_window_source(memory_dir)

# 2. 加载所有窗口的 assistant_replies
uid_map = load_window_assistant_replies(Path(source_record_dir) / "windows")

# 3. 为检索结果附加 assistant 上下文
records = attach_assistant_context(records, boundary_index, uid_map)
# 每条 record 新增: assistant_reply, assistant_uid, session_id, session_local_user_index
```

## 7) `__init__.py` 导出

```python
from search import (
    search_bm25, map_bm25_query,
    search_structured, map_structured_query,
    search_embedding, map_embedding_query,
    search_fused, search_top15_content_dedup,
    attach_assistant_context,
    build_boundary_to_window_source,
    load_window_assistant_replies,
)
```
