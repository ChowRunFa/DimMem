# LoCoMo Compressor

把 `locomo/results/segment_results/raw/<timestamp>` 压缩为：

`locomo/results/segment_results/compressed/<timestamp>`

并保持每条样本目录结构不变（`<type>/<sample>/windows/window_XXXX.json|txt`）。

## 运行

```bash
/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/miniconda3/envs/dimmem_v2/bin/python \
  /mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/DimMem/evaluation/dimmem/locomo/src/compressor/build_compressed_segments.py \
  --raw-run-root /mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/DimMem/evaluation/dimmem/locomo/results/segment_results/raw/20260425_221840
```

## 常用参数

- `--run-name`: 自定义输出子目录
- `--model-name`: 压缩模型（默认 llmlingua2 multilingual）
- `--device-map`: `cuda` / `cpu`
- `--rate`: 压缩率，默认 `0.8`
- `--target-token`: 目标 token，默认 `-1`
- `--max-records`: 仅压前 N 条（调试）
- `--no-resume`: 关闭断点续跑（默认开启 resume）

## 输出

- `experiment_config.json`
- `status.json`（实时进度，包含 `inflight_record`）
- `<type>/<sample>/summary.json`
- `<type>/<sample>/windows/window_XXXX.json|txt`
- `failures.json`（如有失败）

## 断点续跑

默认开启：如果你重复使用同一个 `--run-name`，脚本会自动跳过已存在 `summary.json` 的样本并继续剩余样本。
