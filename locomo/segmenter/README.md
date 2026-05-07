# LoCoMo Segmenter

用于把 `/data/qwt/projects/data/locomo10_by_type` 数据切成滑动窗口。

默认参数（按你的要求）：

- `window_size = 25`
- `overlap = 5`
- `window_step = 20`

## 输入

目录结构：

- `/data/qwt/projects/data/locomo10_by_type/1Multi-hop/locomo10.json`
- `/data/qwt/projects/data/locomo10_by_type/2Temporal/locomo10.json`
- `/data/qwt/projects/data/locomo10_by_type/3Open-domain/locomo10.json`
- `/data/qwt/projects/data/locomo10_by_type/4Single-hop/locomo10.json`

每条样本会读取 `conversation` 下：

- `session_n_date_time`（例如 `8:56 pm on 20 July, 2023`）
- `session_n`（会话轮次列表）

## 输出

默认输出到：

`evaluation/dimmem/locomo/results/segment_results/raw/<run_name>/...`

每条样本目录：

- `input_item.json`
- `all_turns.txt`
- `summary.json`
- `windows/window_0000.json|txt`

## 运行

```bash
/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/miniconda3/envs/dimmem_v2/bin/python \
  /mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/DimMem/evaluation/dimmem/locomo/src/segmenter/build_raw_segments.py \
  --input-root /data/qwt/projects/data/locomo10_by_type \
  --window-size 25 \
  --overlap 5
```

可选：

- `--run-name <name>`：自定义输出目录名
- `--max-items N`：每个类型只跑前 N 条
