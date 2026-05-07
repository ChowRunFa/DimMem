# LongMemEval Segmenter

`src/segmenter` 负责把 LongMemEval 原始样本切成滑动窗口，并为每个窗口生成 assistant 回复索引。

输出到：`debug_query_memory/memory_results/segment_results/raw/<run_timestamp>/...`

## 1) 输入

支持两种输入：

- 单文件：`longmemeval_s_cleaned__<question-type>.json`
- 目录：包含多个 `longmemeval_s_cleaned__*.json`

默认输入目录：`/data/qwt/projects/data/longmemeval_s_cleaned_by_type`

每条样本依赖字段：

- `question_id`
- `haystack_sessions`（会话列表，每个会话是 `[{role, content}, ...]`）
- `haystack_dates`（每个会话起始时间，格式 `YYYY/MM/DD (Day) HH:MM`）
- `haystack_session_ids`（会话 ID 列表）
- `question` / `question_date` / `answer`（可选）

## 2) 输出结构

```
raw/<run_name>/
├── run_manifest.json
└── <question_type>/<sample_id>/
    ├── input_item.json          # 原始样本
    ├── all_user_messages.txt    # 全量用户消息
    ├── summary.json             # 窗口汇总
    └── windows/
        ├── window_0000.json     # 窗口数据
        ├── window_0000.txt      # 窗口对话文本
        ├── window_0000_assistant_replies.json  # AI 回复索引
        └── ...
```

其中 `sample_id` 形如：`0000_<question_id>`。

## 3) Window JSON 字段

每条 window message 包含：

| 字段 | 说明 |
|------|------|
| `global_user_index` | 全局用户消息序号（跨所有 session） |
| `session_id` | 所属 session ID |
| `session_local_user_index` | session 内用户消息序号（1-based） |
| `timestamp` | ISO 格式时间戳 |
| `weekday` | 星期几缩写 |
| `content` | 消息内容（可能截断） |

## 4) Assistant Replies 索引

每个窗口同时生成 `window_XXXX_assistant_replies.json`，为窗口内每条用户消息映射其对应的 AI 回复。

**uid 格式**：`w{window_index:04d}u{source_id:02d}`

- `window_index`：窗口编号（0-based）
- `source_id`：窗口内消息序号（1-based，对应对话文本中的 `1.User:`, `2.User:`, ...）

示例：`w0005u08` 表示第 5 个窗口的第 8 条用户消息

```json
{
  "window_index": 5,
  "source_count": 25,
  "replies": {
    "w0005u01": {
      "uid": "w0005u01",
      "source_id": 1,
      "global_user_index": 101,
      "session_id": "abc123_4",
      "session_local_user_index": 3,
      "assistant_reply": "Here is the assistant's response..."
    }
  }
}
```

## 5) 切分逻辑

- 默认仅取 `role=user` 消息（`--use-user-only`，默认启用）
- 一个会话内用户消息时间：以 `haystack_dates[i]` 为起点，每条 +0.5 秒
- 滑窗参数默认：
  - `window_size=25`
  - `overlap_ratio=0.2`（即 `overlap=5, step=20`）
- 长文本截断（默认）：
  - 超过 `8000` 字符时，保留 `头500 + 中间200 + 尾300`，中间用 `...` 分隔

## 6) 使用方法

```bash
/mnt/workspace/.../miniconda3/envs/dimmem_v2/bin/python \
  src/segmenter/build_raw_segments.py \
  --input-path /data/qwt/projects/data/longmemeval_s_cleaned_by_type \
  --output-root .../segment_results/raw
```

常用参数：

- `--run-name 20260425_xxx`：自定义输出子目录名
- `--max-items 5`：每个类型只跑前 N 条（调试用）
- `--window-size 25`：窗口大小
- `--overlap -1 --overlap-ratio 0.2`：自动计算 overlap
- `--truncate-threshold 8000`：长文本截断阈值
