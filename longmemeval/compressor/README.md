# LongMemEval Compressor

`src/compressor` 使用 LLMLingua-2 对 raw 窗口的用户消息内容进行压缩，减少 token 数同时保留关键信息。

输出到：`debug_query_memory/memory_results/segment_results/compressed/<run_name>/...`

## 1) 压缩模型

默认使用 **LLMLingua-2** (BERT-base multilingual)：

```
/data/aios-weights/LLM-Lingua/llmlingua-2-bert-base-multilingual-cased-meetingbank
```

- 基于 token 级别的重要性评估，逐 token 决定保留或丢弃
- `use_llmlingua2=True`：使用 v2 压缩算法
- 支持 GPU 加速（`--device-map cuda`）

## 2) 压缩逻辑

- **逐消息压缩**：对每条 user message 的 `content` 字段独立压缩
- 保留原始内容在 `original_content` 字段中
- 压缩后的文本写入 `content` 字段
- 窗口的 `text` 字段用压缩后内容重新生成
- `assistant_replies.json` 从 raw 窗口直接复制（消息映射关系不变）

## 3) 输出结构

```
compressed/<run_name>/
├── experiment_config.json
├── failures.json              # (可选) 失败记录
└── <question_type>/<sample_id>/
    ├── input_item.json        # 原始样本 (复制)
    ├── all_user_messages.txt  # (复制)
    ├── summary.json           # 含 compression 统计
    └── windows/
        ├── window_0000.json
        ├── window_0000.txt
        ├── window_0000_assistant_replies.json  # 从 raw 复制
        └── ...
```

## 4) 压缩后 Window Message 字段

每条压缩后的 message 新增：

| 字段 | 说明 |
|------|------|
| `original_content` | 压缩前的原始内容 |
| `content` | 压缩后的内容 |
| `compression_applied` | bool，是否实际发生了压缩 |
| `original_length` | 原始字符数 |
| `compressed_length` | 压缩后字符数 |

## 5) Summary 中的 compression 统计

```json
{
  "compression": {
    "model_name": "llmlingua-2-bert-base-multilingual-cased-meetingbank",
    "device_map": "cuda",
    "rate": 0.8,
    "target_token": -1,
    "window_count": 21,
    "message_count": 525,
    "compressed_message_count": 320,
    "original_chars": 152000,
    "compressed_chars": 121600,
    "compression_ratio": 0.8
  }
}
```

## 6) 使用方法

```bash
/mnt/workspace/.../miniconda3/envs/dimmem_v2/bin/python \
  src/compressor/build_compressed_segments.py \
  --raw-run-root .../segment_results/raw/20260425_020737 \
  --output-root .../segment_results/compressed
```

关键参数：

- `--run-name`：自定义输出目录名（默认当前时间戳，通常设为与 raw 同名）
- `--model-name`：压缩模型路径
- `--device-map`：`cuda` / `cpu`
- `--rate 0.8`：目标压缩率（保留 80%）
- `--target-token -1`：目标 token 数（-1 表示不强制）
- `--max-records 5`：调试用，仅处理前 N 条
- `--max-batch-size 50`：LLMLingua 内部 batch
- `--max-force-token 100`：强制保留的 token 数上限
