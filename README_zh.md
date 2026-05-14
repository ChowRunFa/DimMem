<p align="center">
  <img src="assets/logo.png" alt="DimMem Logo" width="200">
</p>

<h1 align="center">DimMem 评估流水线</h1>

<p align="center">基于 <b>LongMemEval</b> 和 <b>LoCoMo</b> 基准测试的记忆增强问答评估系统</p>

## 概述

DimMem（Dimension Memory）将长对话转化为结构化记忆记录，并通过多路检索和 LLM 判定来评估问答准确率。

### 系统架构

<p align="center">
  <img src="assets/framework_gpt.png" alt="DimMem Framework" width="800">
</p>

流水线包含以下阶段：

```
原始对话
    │
    ▼
┌─────────────────┐
│  1. 分割器       │  将对话切分为滑动窗口片段
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. 压缩器       │  (可选) 使用 LLMLingua-2 压缩对话片段
└────────┬────────┘
         ▼
┌──────────────────────────┐
│  3. 记忆抽取              │  LLM 从窗口中提取结构化记忆
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  4. 查询分析              │  将问题解析为结构化检索查询
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  5. 检索                  │  多路搜索 (BM25 + 稠密向量 + 结构化维度)
│     + 助手上下文回补       │  按需附加 AI 原始回复
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  6. 问答生成              │  基于检索到的记忆生成回答
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  7. 评判                  │  LLM 将回答与标准答案对比评分
└──────────────────────────┘
```

## 目录结构

```
DimMem/
├── data/                  # 基准数据集（需自行获取，见 data/README.md）
│   ├── README.md
│   ├── longmemeval_s_cleaned.json    # LongMemEval: 500 条 QA 记录
│   └── locomo10.json                 # LoCoMo: 10 段多轮长对话 + QA
│
├── quick_start/           # 快速入门演示
│   └── quickstart_extract.py         # 自包含记忆抽取 Demo（内置测试数据）
│
├── longmemeval/           # LongMemEval 基准流水线
│   ├── models/            # DimensionMemory & ParsedQuery 数据模型
│   ├── utils/             # LocalEmbeddingClient (sentence-transformers)
│   ├── segmenter/         # 步骤 1：滑动窗口分割
│   ├── compressor/        # 步骤 2：LLMLingua-2 压缩
│   ├── prompts/           # 提示词模板（抽取、QA、评判、查询分析）
│   ├── memory_constructor/# 步骤 3：结构化记忆抽取
│   ├── query_parser/      # 步骤 4：查询解析
│   ├── search/            # 步骤 5：多路检索 (BM25、向量、结构化、融合、助手上下文)
│   ├── update/            # 记忆更新检测与合并
│   └── qa_judge/          # 步骤 6+7：QA 生成、评判与报告
│
├── locomo/                # LoCoMo 基准流水线
│   ├── models/            # DimensionMemory & ParsedQuery 数据模型
│   ├── segmenter/         # 步骤 1
│   ├── compressor/        # 步骤 2
│   ├── memory_constructor/# 步骤 3：记忆抽取（支持多线程并行）
│   ├── prompts/           # 提示词模板
│   ├── query_parser/      # 步骤 4
│   ├── search/            # 步骤 5：检索 (BM25、MiniLM、结构化)
│   ├── qa/                # 步骤 6：QA 生成
│   ├── judge/             # 步骤 7：评判
│   └── update/            # 记忆更新
│
├── requirements.txt
├── README.md              # 英文文档
└── README_zh.md           # 中文文档（本文件）
```

## 核心概念

### DimensionMemory（维度记忆）

每条记忆记录包含以下结构化字段：

| 字段 | 说明 | 示例 |
|------|------|------|
| `source_id` | 对话中的消息编号 | `7` |
| `content` | 记忆主文本（自包含的完整句子） | `用户使用 LLaMA2-7B 作为基座模型` |
| `dimension.memory_type` | 记忆类型：`fact` / `episodic` / `profile` | `fact` |
| `dimension.time` | 时间信息（绝对日期） | `2023-05-08` |
| `dimension.location` | 地点/平台/场景 | `北京` |
| `dimension.reason` | 原因/动机 | `为了降低推理成本` |
| `dimension.purpose` | 目的/意图 | `部署到生产环境` |
| `dimension.keywords` | 检索关键词列表 | `["LLaMA2", "基座模型"]` |

**记忆类型说明**：
- **fact**：稳定事实——身份、背景、关系、状态、工具、模型等
- **episodic**：具体事件——经历、动作、进展、计划等
- **profile**：长期画像——偏好、习惯、兴趣、价值观、风格等

### ParsedQuery（解析查询）

查询分析阶段将自然语言问题转化为结构化检索查询：

| 字段 | 说明 |
|------|------|
| `query_anchor` | 改写后的检索友好文本 |
| `need_assistant_context` | 是否需要检索助手回复 |
| `dimension.target_memory_type` | 目标记忆类型 |
| `dimension.keywords` | 检索关键词 |
| `dimension.time` | 时间约束 |
| `dimension.location` | 地点约束 |
| `answer_dim` | 答案对应的字段（content/time/location/...） |

## 快速入门

### 环境安装

```bash
pip install -r requirements.txt
```

依赖项：`numpy`、`requests`、`httpx`、`openai`、`tqdm`、`sentence-transformers`、`torch`

可选依赖：
- `llmlingua`：压缩步骤所需（步骤 2）
- `onnxruntime`：ONNX 嵌入推理加速

> **注意**：压缩步骤需要 GPU 环境和 LLMLingua-2 模型权重。

### 运行 Demo

无需完整数据集，即可体验维度记忆抽取：

```bash
python quick_start/quickstart_extract.py \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b \
  --demo both
```

选项：`--demo longmemeval`、`--demo locomo` 或 `--demo both`（默认）。

脚本内置了 LongMemEval 和 LoCoMo 的测试对话，调用 LLM 抽取结构化记忆后打印标准化输出。

## 数据集准备

两个数据集均为公开可用：

1. **LongMemEval**：https://github.com/xiaowu0162/LongMemEval
2. **LoCoMo**：https://github.com/snap-stanford/LoCoMo

下载处理后的 JSON 文件并放置到 `data/` 目录下：

- `data/longmemeval_s_cleaned.json` — JSON 数组，500 条记录，每条包含：`question_id`、`question_type`、`question`、`answer`、`haystack_sessions`、`haystack_dates`、`haystack_session_ids`
- `data/locomo10.json` — JSON 数组，10 段对话，每段包含：`sample_id`、`conversation`、`qa`、`event_summary`、`observation`、`session_summary`

## 完整流水线运行

### 环境变量配置

```bash
# LLM API（OpenAI 兼容接口）
export BASE_URL="https://api.example.com/v1"    # vLLM / API 代理
export API_KEY="your-api-key"                    # API 密钥
export MODEL="gpt-4.1-mini"                      # 模型名称

# 本地模型路径
export EMBED_MODEL="/path/to/all-MiniLM-L6-v2"                      # 嵌入模型
export COMPRESS_MODEL="/path/to/llmlingua-2-bert-base-multilingual"  # 压缩模型（可选）
```

> **API 说明**：本系统直接使用 `requests.post` 调用 OpenAI 兼容的 `/chat/completions` 接口，不依赖 OpenAI Python SDK。任何兼容该接口的服务均可使用（vLLM、StepFun、mnapi 等）。

---

### LongMemEval 流水线

#### 步骤 1：对话分割

将长对话切分为滑动窗口片段，窗口之间有重叠以保持上下文连贯。

```bash
python longmemeval/segmenter/build_raw_segments.py \
  --input-path ./data/longmemeval_s_cleaned.json \
  --output-root ./results/segments/raw \
  --run-name my_run \
  --window-size 15 \
  --overlap 3
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--window-size` | 每个窗口的消息数 | 25 |
| `--overlap` | 窗口重叠消息数 | 5 |
| `--max-items` | 最多处理的记录数（0=全部） | 0 |

#### 步骤 2（可选）：LLMLingua-2 压缩

使用 LLMLingua-2 对分割后的片段进行文本压缩，减少 LLM 输入长度。

```bash
python longmemeval/compressor/build_compressed_segments.py \
  --raw-run-root ./results/segments/raw/my_run \
  --output-root ./results/segments/compressed \
  --run-name my_run \
  --model-name /path/to/llmlingua-2-model \
  --device-map cuda \
  --rate 0.8
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--rate` | 压缩率（0.8 = 保留 80% 文本） | 0.5 |
| `--device-map` | 设备：`cuda` 或 `cpu` | cuda |
| `--max-records` | 最多处理的记录数（0=全部） | 0 |

> **注意**：此步骤需要 `dimmem_v2` conda 环境（含 `llmlingua` 和 GPU 支持的 PyTorch）。

#### 步骤 3：结构化记忆抽取

LLM 从每个窗口中提取结构化记忆记录。

```bash
python longmemeval/memory_constructor/run_batch_extract.py \
  --segments-root ./results/segments/compressed/my_run \
  --output-root ./results/memories \
  --run-name my_run \
  --overlap 3 \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL \
  --max-tokens 16384 --timeout 120 --max-retries 3
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--segments-root` | 分割/压缩后的片段目录 | — |
| `--overlap` | 重叠消息数（提示词中标注不抽取） | 5 |
| `--max-records` | 最多处理的记录数（0=全部） | 0 |

#### 步骤 4：查询分析

将自然语言问题解析为包含维度约束的结构化查询。

```bash
python longmemeval/query_parser/run_query_analysis.py \
  --input-root ./data/longmemeval_s_cleaned.json \
  --output-base ./results/query_analysis \
  --run-name my_run \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL \
  --max-tokens 4096 --timeout 120 --max-retries 3 \
  --max-convs 1 --max-questions-per-conv 1
```

> `--max-convs 0` 和 `--max-questions-per-conv 0` 表示处理全部记录。

#### 步骤 5：多路检索

```bash
python longmemeval/search/retrieve_from_parsed_query.py \
  --query-parsed ./results/query_analysis/my_run/<question_type>/<sample_id>/parsed.json \
  --memory-dir ./results/memories/my_run/<question_type>/<sample_id> \
  --output-root ./results/retrieval/my_run \
  --top-k 15 \
  --embedding-model /path/to/all-MiniLM-L6-v2 \
  --embedding-device cpu
```

LongMemEval 的检索整合了 BM25、稠密向量（MiniLM）和结构化维度匹配，通过 RRF（Reciprocal Rank Fusion）融合排序。

#### 步骤 6+7：QA 生成 + 评判

```bash
python longmemeval/qa_judge/run_qa_judge_from_retrieval.py \
  --retrieval-root ./results/retrieval/my_run \
  --query-root ./results/query_analysis/my_run \
  --output-base ./results \
  --run-name my_run \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL
```

评判结果输出到 `results/judge/my_run/`，包含 `summary.json`（判定为 `CORRECT` 或 `INCORRECT`）。

#### 生成报告

```bash
python longmemeval/qa_judge/run_report.py \
  --judge-root ./results/judge/my_run
```

生成 `report.md`，包含总体准确率以及按 `question_type` 和检索方法的细分统计。

---

### LoCoMo 流水线

#### 步骤 1：对话分割

```bash
python locomo/segmenter/build_raw_segments.py \
  --input-root ./data/locomo10.json \
  --output-root ./results/locomo_segments/raw \
  --run-name my_run \
  --window-size 25 \
  --overlap 5
```

#### 步骤 2（可选）：压缩

```bash
python locomo/compressor/build_compressed_segments.py \
  --raw-run-root ./results/locomo_segments/raw/my_run \
  --output-root ./results/locomo_segments/compressed \
  --run-name my_run \
  --model-name /path/to/llmlingua-2-model \
  --device-map cuda \
  --rate 0.8
```

#### 步骤 3：记忆抽取

**批量模式**（所有对话顺序处理）：

```bash
python locomo/memory_constructor/run_batch_extract.py \
  --compressed-root ./results/locomo_segments/compressed/my_run \
  --output-root ./results/locomo_memory \
  --run-name my_run \
  --overlap 5 \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL \
  --max-tokens 16384 --timeout 120 --max-retries 3
```

**单条对话并行模式**（支持多端口负载均衡）：

```bash
python locomo/memory_constructor/extract_single_record.py \
  --record-dir ./results/locomo_segments/raw/my_run/<conv>/ \
  --output-record-dir ./results/locomo_memory/<conv>/ \
  --overlap 5 \
  --base-urls http://localhost:7790/v1,http://localhost:7791/v1 \
  --model-name qwen3-30b-a3b \
  --workers 4
```

> `--compressed-root` 同时接受原始或压缩后的片段目录。

#### 步骤 4：查询分析

```bash
python locomo/query_parser/run_query_analysis_by_conv.py \
  --input-root ./data/locomo10.json \
  --output-base ./results/query_analysis \
  --run-name locomo_my_run \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL \
  --max-tokens 4096 --timeout 120 --max-retries 3 \
  --exclude-categories 5
```

| 参数 | 说明 |
|------|------|
| `--exclude-categories` | 排除的问题类别（如 `5`） |
| `--max-convs` | 最多处理的对话数（0=全部） |
| `--max-questions-per-conv` | 每段对话最多处理的问题数（0=全部） |

#### 步骤 5：三路检索

LoCoMo 使用三条独立的检索路径：

```bash
# BM25 词频检索
python locomo/search/run_retrieval_bm25.py \
  --query-run-root ./results/query_analysis/locomo_my_run \
  --memory-root ./results/locomo_memory/my_run \
  --output-base ./results/retrieval/bm25 \
  --run-name my_run --top-k 15

# MiniLM 稠密向量检索
python locomo/search/run_retrieval_minilm.py \
  --query-run-root ./results/query_analysis/locomo_my_run \
  --memory-root ./results/locomo_memory/my_run \
  --output-base ./results/retrieval/minilm \
  --run-name my_run --top-k 15 \
  --embedding-model /path/to/all-MiniLM-L6-v2

# 结构化维度检索
python locomo/search/run_retrieval_from_query_analysis.py \
  --query-run-root ./results/query_analysis/locomo_my_run \
  --memory-root ./results/locomo_memory/my_run \
  --output-base ./results/retrieval/structured \
  --run-name my_run --top-k 15
```

#### 步骤 6：QA 生成（三路融合）

合并三路检索结果后生成回答：

```bash
python locomo/qa/run_qa_from_three_retrievals.py \
  --query-root ./results/query_analysis/locomo_my_run \
  --bm25-root ./results/retrieval/bm25/my_run \
  --minilm-root ./results/retrieval/minilm/my_run \
  --structured-root ./results/retrieval/structured/my_run \
  --output-base ./results/qa \
  --run-name locomo_my_run \
  --top-n-each 15 --max-merged 30 \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL \
  --timeout 120 --max-retries 3
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--top-n-each` | 每路取前 N 条 | 15 |
| `--max-merged` | 融合后最多保留条数 | 30 |

#### 步骤 7：评判

```bash
python locomo/judge/run_judge_from_qa.py \
  --qa-root ./results/qa/locomo_my_run \
  --conv-json ./data/locomo10.json \
  --output-base ./results/judge \
  --run-name locomo_my_run \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL \
  --timeout 120 --max-tokens 512 --max-retries 3
```

生成 `report.md`，包含总体准确率、按对话和按类别的细分统计。

> 可使用 `--conv-name conv-26` 仅评判指定对话。

---

### 记忆更新模块

两个基准共享一个离线记忆更新模块，用于检测矛盾并合并记录：

```bash
# LongMemEval
python longmemeval/update/run_update.py \
  --method dimmem \
  --memory-root ./results/memories/my_run \
  --dataset longmemeval \
  --output ./results/update_output/ \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL

# LoCoMo
python longmemeval/update/run_update.py \
  --method dimmem \
  --memory-root ./results/locomo_memory/my_run \
  --dataset locomo \
  --output ./results/update_output/ \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL
```

支持的 `--method`：`lightmem`、`dimmem`。支持的 `--dataset`：`longmemeval`、`locomo`。

---

## 完整运行示例

项目提供了端到端测试脚本 `run_test_41mini.sh`，使用 `gpt-4.1-mini` 模型运行完整的 7 步流水线（含压缩步骤），可作为参考：

```bash
bash run_test_41mini.sh
```

该脚本使用以下配置：
- LongMemEval：`window_size=15, overlap=3`
- LoCoMo：`window_size=25, overlap=5`
- 压缩率：0.8（保留 80% 文本）
- 检索：三路 top-15，融合后最多 30 条
- 各步骤限制 `--max-items 1` / `--max-records 1`（测试模式）

## 结果输出

所有中间结果和最终结果保存在 `./results/` 目录下：

```
results/
├── segments/                    # LongMemEval 分割结果
│   ├── raw/<run_name>/
│   └── compressed/<run_name>/
├── locomo_segments/             # LoCoMo 分割结果
│   ├── raw/<run_name>/
│   └── compressed/<run_name>/
├── memories/<run_name>/         # LongMemEval 抽取的记忆
├── locomo_memory/<run_name>/    # LoCoMo 抽取的记忆
├── query_analysis/
│   ├── <run_name>/              # LongMemEval 查询分析
│   └── locomo_<run_name>/       # LoCoMo 查询分析
├── retrieval/
│   ├── <run_name>/              # LongMemEval 检索结果
│   ├── bm25/<run_name>/         # LoCoMo BM25 检索
│   ├── minilm/<run_name>/       # LoCoMo MiniLM 检索
│   └── structured/<run_name>/   # LoCoMo 结构化检索
├── qa/
│   ├── <run_name>/              # LongMemEval QA 结果
│   └── locomo_<run_name>/       # LoCoMo QA 结果
└── judge/
    ├── <run_name>/              # LongMemEval 评判结果
    └── locomo_<run_name>/       # LoCoMo 评判结果 + report.md
```

每个结果目录下包含 `run_command.sh`，记录生成该结果的完整命令，方便复现。

## 设计要点

1. **滑动窗口分割**：对话被切分为有重叠的窗口（LongMemEval 默认 25 条消息 / 20% 重叠，LoCoMo 默认 25 条 / 5 条重叠），在保持上下文的同时控制 LLM 输入长度。

2. **维度记忆模型**：每条记忆附带结构化维度字段（类型、时间、地点、原因、目的、关键词），支持多维度检索和精确匹配。

3. **多路检索融合**：三条检索路径（BM25 词频、稠密向量嵌入、结构化维度匹配）通过 RRF 融合排序，兼顾召回率和准确率。

4. **动态助手上下文**：查询分类器（`need_assistant_context`）判断是否需要附加 AI 原始回复，节省约 89% 的上下文查找开销。

5. **记忆更新**：离线矛盾检测与合并机制，在新会话到来时保持记忆库的一致性。

6. **重叠上下文规则**：抽取提示词中明确标注窗口重叠部分仅用于理解上下文，不作为新记忆来源，避免跨窗口重复抽取。

## 使用的模型

| 用途 | 模型 | 说明 |
|------|------|------|
| 记忆抽取 / 查询分析 | Qwen3-30B-A3B 或 gpt-4.1-mini | 通过 OpenAI 兼容 API 调用 |
| 嵌入向量 | all-MiniLM-L6-v2 (384 维) | 本地加载，sentence-transformers |
| QA + 评判 | gpt-4.1-mini | 或任意 OpenAI 兼容模型 |
| 文本压缩 | LLMLingua-2 (bert-base-multilingual) | 本地 GPU 推理 |
