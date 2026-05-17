<p align="center">
  <img src="assets/logo.png" alt="DimMem Logo" width="200">
</p>

<h1 align="center">DimMem</h1>

<p align="center">
  面向长对话 QA 评估的维度感知结构化记忆系统
  <br>
  <a href="README.md">English</a>
</p>

<p align="center">
  <img src="assets/framework_gpt.png" alt="DimMem Framework" width="800">
</p>

## 特性

- **维度记忆模型** — 每条记忆附带结构化维度字段（类型、时间、地点、原因、目的、关键词），支持多维精准检索
- **三路检索融合** — BM25 + 稠密向量（MiniLM）+ 结构化维度匹配，通过 RRF 融合排序
- **7 步评估流水线** — 分割 → 压缩 → 抽取 → 查询分析 → 检索 → QA → 评判
- **动态助手上下文** — 自动判断是否需要附加 AI 回复，节省约 89% 的上下文查找开销
- **双基准支持** — 完整支持 [LongMemEval](https://github.com/xiaowu0162/LongMemEval) 和 [LoCoMo](https://github.com/snap-stanford/LoCoMo) 两个基准测试

## 快速入门

### 安装

```bash
pip install -r requirements.txt
```

### Demo（无需数据集）

```bash
python quick_start/quickstart_extract.py \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b \
  --demo both    # longmemeval / locomo / both
```

### 一键完整流水线测试

```bash
# 先编辑 API 密钥
bash quick_start/run_quickstart.sh
```

在每个基准上运行一条样本的完整 7 步流水线。设置 `SKIP_COMPRESS=1` 可跳过 GPU 压缩步骤。

## 数据集准备

下载后放入 `data/` 目录：

| 文件 | 来源 | 说明 |
|------|------|------|
| `longmemeval_s_cleaned.json` | [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | 500 条 QA 记录 |
| `locomo10.json` | [LoCoMo](https://github.com/snap-stanford/LoCoMo) | 10 段多轮长对话 + QA |

## 完整流水线

### 环境变量

```bash
export BASE_URL="https://api.example.com/v1"
export API_KEY="your-api-key"
export MODEL="gpt-4.1-mini"
export EMBED_MODEL="/path/to/all-MiniLM-L6-v2"
```

> 所有 LLM 调用均通过 `requests.post` 直接请求 OpenAI 兼容 `/chat/completions` 接口，不依赖 OpenAI SDK。

---

### LongMemEval

#### 步骤 1：对话分割

```bash
python longmemeval/segmenter/build_raw_segments.py \
  --input-path ./data/longmemeval_s_cleaned.json \
  --output-root ./results/segments/raw \
  --run-name my_run --window-size 15 --overlap 3
```

#### 步骤 2（可选）：LLMLingua-2 压缩

```bash
python longmemeval/compressor/build_compressed_segments.py \
  --raw-run-root ./results/segments/raw/my_run \
  --output-root ./results/segments/compressed \
  --run-name my_run --rate 0.8 --device-map cuda
```

> 需要 GPU 环境及 `llmlingua` 包。

#### 步骤 3：结构化记忆抽取

```bash
python longmemeval/memory_constructor/run_batch_extract.py \
  --segments-root ./results/segments/raw/my_run \
  --output-root ./results/memories --run-name my_run \
  --overlap 3 \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL
```

#### 步骤 4：查询分析

```bash
python longmemeval/query_parser/run_query_analysis.py \
  --input-root ./data/longmemeval_s_cleaned.json \
  --output-base ./results/query_analysis --run-name my_run \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL
```

#### 步骤 5：多路检索

```bash
python longmemeval/search/retrieve_from_parsed_query.py \
  --query-parsed ./results/query_analysis/my_run/<question_type>/<sample_id>/parsed.json \
  --memory-dir ./results/memories/my_run/<question_type>/<sample_id> \
  --output-root ./results/retrieval/my_run \
  --top-k 15 --embedding-model $EMBED_MODEL
```

#### 步骤 6+7：QA + 评判

```bash
python longmemeval/qa_judge/run_qa_judge_from_retrieval.py \
  --retrieval-root ./results/retrieval/my_run \
  --query-root ./results/query_analysis/my_run \
  --output-base ./results --run-name my_run \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL
```

#### 生成报告

```bash
python longmemeval/qa_judge/run_report.py --judge-root ./results/judge/my_run
```

---

### LoCoMo

#### 步骤 1：对话分割

```bash
python locomo/segmenter/build_raw_segments.py \
  --input-root ./data/locomo10.json \
  --output-root ./results/locomo_segments/raw \
  --run-name my_run --window-size 25 --overlap 5
```

#### 步骤 2（可选）：压缩

```bash
python locomo/compressor/build_compressed_segments.py \
  --raw-run-root ./results/locomo_segments/raw/my_run \
  --output-root ./results/locomo_segments/compressed \
  --run-name my_run --rate 0.8 --device-map cuda
```

#### 步骤 3：记忆抽取

```bash
python locomo/memory_constructor/run_batch_extract.py \
  --compressed-root ./results/locomo_segments/raw/my_run \
  --output-root ./results/locomo_memory --run-name my_run \
  --overlap 5 \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL
```

#### 步骤 4：查询分析

```bash
python locomo/query_parser/run_query_analysis_by_conv.py \
  --input-root ./data/locomo10.json \
  --output-base ./results/query_analysis --run-name locomo_my_run \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL \
  --exclude-categories 5
```

#### 步骤 5：三路检索

```bash
# BM25
python locomo/search/run_retrieval_bm25.py \
  --query-run-root ./results/query_analysis/locomo_my_run \
  --memory-root ./results/locomo_memory/my_run \
  --output-base ./results/retrieval/bm25 --run-name my_run --top-k 15

# MiniLM 稠密向量
python locomo/search/run_retrieval_minilm.py \
  --query-run-root ./results/query_analysis/locomo_my_run \
  --memory-root ./results/locomo_memory/my_run \
  --output-base ./results/retrieval/minilm --run-name my_run --top-k 15 \
  --embedding-model $EMBED_MODEL

# 结构化维度
python locomo/search/run_retrieval_from_query_analysis.py \
  --query-run-root ./results/query_analysis/locomo_my_run \
  --memory-root ./results/locomo_memory/my_run \
  --output-base ./results/retrieval/structured --run-name my_run --top-k 15
```

#### 步骤 6：QA 生成

```bash
python locomo/qa/run_qa_from_three_retrievals.py \
  --query-root ./results/query_analysis/locomo_my_run \
  --bm25-root ./results/retrieval/bm25/my_run \
  --minilm-root ./results/retrieval/minilm/my_run \
  --structured-root ./results/retrieval/structured/my_run \
  --output-base ./results/qa --run-name locomo_my_run \
  --top-n-each 15 \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL
```

#### 步骤 7：评判

```bash
python locomo/judge/run_judge_from_qa.py \
  --qa-root ./results/qa/locomo_my_run \
  --conv-json ./data/locomo10.json \
  --output-base ./results/judge --run-name locomo_my_run \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL
```

---

### 记忆更新（可选）

离线矛盾检测与记录合并：

```bash
python longmemeval/update/run_update.py \
  --method dimmem \
  --memory-root ./results/memories/my_run \
  --dataset longmemeval \       # 或 locomo
  --output ./results/update_output/ \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL
```

## 项目结构

```
DimMem/
├── data/                   # 数据集（见"数据集准备"）
├── quick_start/            # Demo 与快速启动脚本
├── longmemeval/            # LongMemEval 流水线
│   ├── models/             #   DimensionMemory & ParsedQuery
│   ├── segmenter/          #   步骤 1
│   ├── compressor/         #   步骤 2
│   ├── memory_constructor/ #   步骤 3
│   ├── query_parser/       #   步骤 4
│   ├── search/             #   步骤 5 (BM25 + 向量 + 结构化 + RRF)
│   ├── qa_judge/           #   步骤 6+7
│   └── update/             #   记忆更新
├── locomo/                 # LoCoMo 流水线（同上结构）
├── assets/                 # Logo 与架构图
└── requirements.txt
```

## 使用的模型

| 组件 | 模型 | 说明 |
|------|------|------|
| 记忆抽取 / 查询分析 | Qwen3-30B-A3B 或 gpt-4.1-mini | OpenAI 兼容 API |
| 嵌入向量 | all-MiniLM-L6-v2 (384 维) | 本地，sentence-transformers |
| QA + 评判 | gpt-4.1-mini | 任意 OpenAI 兼容模型 |
| 文本压缩 | LLMLingua-2 (bert-base-multilingual) | 本地 GPU |
