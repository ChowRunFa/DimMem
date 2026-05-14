<p align="center">
  <img src="assets/logo.png" alt="DimMem Logo" width="200">
</p>

<h1 align="center">DimMem Evaluation Pipeline</h1>

<p align="center">Memory-augmented QA evaluation on <b>LongMemEval</b> and <b>LoCoMo</b> benchmarks</p>

## Overview

DimMem (Dimension Memory) transforms long conversations into structured memory records and evaluates QA accuracy through multi-route retrieval and LLM judging.

### System Architecture

<p align="center">
  <img src="assets/framework_gpt.png" alt="DimMem Framework" width="800">
</p>

The pipeline consists of the following stages:

```
Raw Conversations
    │
    ▼
┌─────────────────┐
│  1. Segmenter   │  Split conversations into overlapping windows
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. Compressor  │  (Optional) Compress segments with LLMLingua-2
└────────┬────────┘
         ▼
┌──────────────────────────┐
│  3. Memory Extraction    │  LLM extracts structured memories from windows
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  4. Query Analysis       │  Parse questions into structured queries
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  5. Retrieval            │  Multi-route search (BM25 + Dense + Structured)
│     + Assistant Context  │  Dynamic recall of AI responses when needed
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  6. QA Generation        │  Answer questions from retrieved memories
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  7. Judge                │  LLM judges correctness against gold answers
└──────────────────────────┘
```

## Directory Structure

```
DimMem/
├── data/                  # Benchmark datasets (see data/README.md)
│   ├── README.md
│   ├── longmemeval_s_cleaned.json    # LongMemEval: 500 QA records
│   └── locomo10.json                 # LoCoMo: 10 multi-turn conversations + QA
│
├── quick_start/           # Quickstart demo
│   └── quickstart_extract.py         # Self-contained memory extraction demo (inline test data)
│
├── longmemeval/           # LongMemEval benchmark pipeline
│   ├── models/            # DimensionMemory & ParsedQuery data models
│   ├── utils/             # LocalEmbeddingClient (sentence-transformers)
│   ├── segmenter/         # Step 1: Windowed segmentation
│   ├── compressor/        # Step 2: LLMLingua-2 compression
│   ├── prompts/           # Prompt templates (extraction, QA, judge, query)
│   ├── memory_constructor/# Step 3: Memory extraction (library module)
│   ├── query_parser/      # Step 4: Query analysis
│   ├── search/            # Step 5: Multi-route retrieval (BM25, embedding, structured, fusion, assistant context)
│   ├── update/            # Memory update detection & consolidation
│   └── qa_judge/          # Steps 6+7: QA, Judge & Report
│
├── locomo/                # LoCoMo benchmark pipeline
│   ├── models/            # DimensionMemory & ParsedQuery data models
│   ├── segmenter/         # Step 1
│   ├── compressor/        # Step 2
│   ├── memory_constructor/# Step 3: Memory extraction (parallel)
│   ├── prompts/           # Prompt templates
│   ├── query_parser/      # Step 4: Query analysis
│   ├── search/            # Step 5: Retrieval (BM25, MiniLM, structured)
│   ├── qa/                # Step 6: QA generation
│   ├── judge/             # Step 7: Judge
│   └── update/            # Memory update
│
├── requirements.txt
├── README.md              # English documentation
└── README_zh.md           # Chinese documentation
```

## Core Concepts

### DimensionMemory

Each memory record contains the following structured fields:

| Field | Description | Example |
|-------|-------------|---------|
| `source_id` | Message number in the conversation | `7` |
| `content` | Core memory text (self-contained sentence) | `User uses LLaMA2-7B as the base model` |
| `dimension.memory_type` | Memory type: `fact` / `episodic` / `profile` | `fact` |
| `dimension.time` | Time information (absolute date) | `2023-05-08` |
| `dimension.location` | Location/platform/scene | `Beijing` |
| `dimension.reason` | Reason/motivation | `To reduce inference cost` |
| `dimension.purpose` | Purpose/intention | `Deploy to production` |
| `dimension.keywords` | Retrieval keyword list | `["LLaMA2", "base model"]` |

**Memory type descriptions**:
- **fact**: Stable facts — identity, background, relationships, status, tools, models, etc.
- **episodic**: Specific events — experiences, actions, progress, plans, etc.
- **profile**: Long-term profiles — preferences, habits, interests, values, style, etc.

### ParsedQuery

The query analysis stage converts natural language questions into structured retrieval queries:

| Field | Description |
|-------|-------------|
| `query_anchor` | Rewritten retrieval-friendly text |
| `need_assistant_context` | Whether to retrieve assistant responses |
| `dimension.target_memory_type` | Target memory types |
| `dimension.keywords` | Retrieval keywords |
| `dimension.time` | Time constraint |
| `dimension.location` | Location constraint |
| `answer_dim` | Answer field (content/time/location/...) |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `requests`, `httpx`, `openai`, `tqdm`, `sentence-transformers`, `torch`

Optional:
- `llmlingua`: Required for compression step (Step 2)
- `onnxruntime`: ONNX embedding inference acceleration

> **Note**: The compression step requires a GPU environment and LLMLingua-2 model weights.

### Run Demo

Experience dimension memory extraction without the full datasets:

```bash
python quick_start/quickstart_extract.py \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b \
  --demo both
```

Options: `--demo longmemeval`, `--demo locomo`, or `--demo both` (default).

The script includes inline test conversations for both benchmarks, calls the LLM to extract structured memories, and prints the normalized output.

## Dataset Preparation

Both datasets are publicly available:

1. **LongMemEval**: https://github.com/xiaowu0162/LongMemEval
2. **LoCoMo**: https://github.com/snap-stanford/LoCoMo

Download the processed JSON files and place them in the `data/` directory:

- `data/longmemeval_s_cleaned.json` — JSON array, 500 records, each containing: `question_id`, `question_type`, `question`, `answer`, `haystack_sessions`, `haystack_dates`, `haystack_session_ids`
- `data/locomo10.json` — JSON array, 10 conversations, each containing: `sample_id`, `conversation`, `qa`, `event_summary`, `observation`, `session_summary`

## Full Pipeline

### Environment Variables

```bash
# LLM API (OpenAI-compatible)
export BASE_URL="https://api.example.com/v1"    # vLLM / API proxy
export API_KEY="your-api-key"                    # API key
export MODEL="gpt-4.1-mini"                      # Model name

# Local model paths
export EMBED_MODEL="/path/to/all-MiniLM-L6-v2"                      # Embedding model
export COMPRESS_MODEL="/path/to/llmlingua-2-bert-base-multilingual"  # Compression model (optional)
```

> **API Note**: This system calls the OpenAI-compatible `/chat/completions` endpoint directly via `requests.post`, without depending on the OpenAI Python SDK. Any compatible service can be used (vLLM, StepFun, mnapi, etc.).

---

### LongMemEval Pipeline

#### Step 1: Conversation Segmentation

Split long conversations into overlapping sliding windows.

```bash
python longmemeval/segmenter/build_raw_segments.py \
  --input-path ./data/longmemeval_s_cleaned.json \
  --output-root ./results/segments/raw \
  --run-name my_run \
  --window-size 15 \
  --overlap 3
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--window-size` | Messages per window | 25 |
| `--overlap` | Overlapping messages between windows | 5 |
| `--max-items` | Max records to process (0 = all) | 0 |

#### Step 2 (Optional): LLMLingua-2 Compression

Compress segmented windows to reduce LLM input length.

```bash
python longmemeval/compressor/build_compressed_segments.py \
  --raw-run-root ./results/segments/raw/my_run \
  --output-root ./results/segments/compressed \
  --run-name my_run \
  --model-name /path/to/llmlingua-2-model \
  --device-map cuda \
  --rate 0.8
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--rate` | Compression rate (0.8 = keep 80% text) | 0.5 |
| `--device-map` | Device: `cuda` or `cpu` | cuda |
| `--max-records` | Max records to process (0 = all) | 0 |

> **Note**: This step requires the `dimmem_v2` conda environment (with `llmlingua` and GPU-enabled PyTorch).

#### Step 3: Structured Memory Extraction

LLM extracts structured memory records from each window.

```bash
python longmemeval/memory_constructor/run_batch_extract.py \
  --segments-root ./results/segments/compressed/my_run \
  --output-root ./results/memories \
  --run-name my_run \
  --overlap 3 \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL \
  --max-tokens 16384 --timeout 120 --max-retries 3
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--segments-root` | Segmented/compressed window directory | — |
| `--overlap` | Overlap count (marked as no-extract in prompt) | 5 |
| `--max-records` | Max records to process (0 = all) | 0 |

#### Step 4: Query Analysis

Parse natural language questions into structured queries with dimension constraints.

```bash
python longmemeval/query_parser/run_query_analysis.py \
  --input-root ./data/longmemeval_s_cleaned.json \
  --output-base ./results/query_analysis \
  --run-name my_run \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL \
  --max-tokens 4096 --timeout 120 --max-retries 3 \
  --max-convs 1 --max-questions-per-conv 1
```

> `--max-convs 0` and `--max-questions-per-conv 0` means process all records.

#### Step 5: Multi-Route Retrieval

```bash
python longmemeval/search/retrieve_from_parsed_query.py \
  --query-parsed ./results/query_analysis/my_run/<question_type>/<sample_id>/parsed.json \
  --memory-dir ./results/memories/my_run/<question_type>/<sample_id> \
  --output-root ./results/retrieval/my_run \
  --top-k 15 \
  --embedding-model /path/to/all-MiniLM-L6-v2 \
  --embedding-device cpu
```

LongMemEval retrieval integrates BM25, dense embedding (MiniLM), and structured dimension matching, fused via RRF (Reciprocal Rank Fusion).

#### Steps 6+7: QA Generation + Judge

```bash
python longmemeval/qa_judge/run_qa_judge_from_retrieval.py \
  --retrieval-root ./results/retrieval/my_run \
  --query-root ./results/query_analysis/my_run \
  --output-base ./results \
  --run-name my_run \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL
```

Judge results are output to `results/judge/my_run/`, containing `summary.json` (verdict: `CORRECT` or `INCORRECT`).

#### Generate Report

```bash
python longmemeval/qa_judge/run_report.py \
  --judge-root ./results/judge/my_run
```

Generates `report.md` with overall accuracy and breakdowns by `question_type` and retrieval method.

---

### LoCoMo Pipeline

#### Step 1: Conversation Segmentation

```bash
python locomo/segmenter/build_raw_segments.py \
  --input-root ./data/locomo10.json \
  --output-root ./results/locomo_segments/raw \
  --run-name my_run \
  --window-size 25 \
  --overlap 5
```

#### Step 2 (Optional): Compression

```bash
python locomo/compressor/build_compressed_segments.py \
  --raw-run-root ./results/locomo_segments/raw/my_run \
  --output-root ./results/locomo_segments/compressed \
  --run-name my_run \
  --model-name /path/to/llmlingua-2-model \
  --device-map cuda \
  --rate 0.8
```

#### Step 3: Memory Extraction

**Batch mode** (all conversations sequentially):

```bash
python locomo/memory_constructor/run_batch_extract.py \
  --compressed-root ./results/locomo_segments/compressed/my_run \
  --output-root ./results/locomo_memory \
  --run-name my_run \
  --overlap 5 \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL \
  --max-tokens 16384 --timeout 120 --max-retries 3
```

**Single conversation parallel mode** (multi-port load balancing):

```bash
python locomo/memory_constructor/extract_single_record.py \
  --record-dir ./results/locomo_segments/raw/my_run/<conv>/ \
  --output-record-dir ./results/locomo_memory/<conv>/ \
  --overlap 5 \
  --base-urls http://localhost:7790/v1,http://localhost:7791/v1 \
  --model-name qwen3-30b-a3b \
  --workers 4
```

> `--compressed-root` accepts either raw or compressed segment directories.

#### Step 4: Query Analysis

```bash
python locomo/query_parser/run_query_analysis_by_conv.py \
  --input-root ./data/locomo10.json \
  --output-base ./results/query_analysis \
  --run-name locomo_my_run \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL \
  --max-tokens 4096 --timeout 120 --max-retries 3 \
  --exclude-categories 5
```

| Parameter | Description |
|-----------|-------------|
| `--exclude-categories` | Question categories to exclude (e.g., `5`) |
| `--max-convs` | Max conversations to process (0 = all) |
| `--max-questions-per-conv` | Max questions per conversation (0 = all) |

#### Step 5: Three-Route Retrieval

LoCoMo uses three independent retrieval paths:

```bash
# BM25 lexical retrieval
python locomo/search/run_retrieval_bm25.py \
  --query-run-root ./results/query_analysis/locomo_my_run \
  --memory-root ./results/locomo_memory/my_run \
  --output-base ./results/retrieval/bm25 \
  --run-name my_run --top-k 15

# MiniLM dense embedding retrieval
python locomo/search/run_retrieval_minilm.py \
  --query-run-root ./results/query_analysis/locomo_my_run \
  --memory-root ./results/locomo_memory/my_run \
  --output-base ./results/retrieval/minilm \
  --run-name my_run --top-k 15 \
  --embedding-model /path/to/all-MiniLM-L6-v2

# Structured dimension retrieval
python locomo/search/run_retrieval_from_query_analysis.py \
  --query-run-root ./results/query_analysis/locomo_my_run \
  --memory-root ./results/locomo_memory/my_run \
  --output-base ./results/retrieval/structured \
  --run-name my_run --top-k 15
```

#### Step 6: QA Generation (Three-Route Merge)

Merge three retrieval routes and generate answers:

```bash
python locomo/qa/run_qa_from_three_retrievals.py \
  --query-root ./results/query_analysis/locomo_my_run \
  --bm25-root ./results/retrieval/bm25/my_run \
  --minilm-root ./results/retrieval/minilm/my_run \
  --structured-root ./results/retrieval/structured/my_run \
  --output-base ./results/qa \
  --run-name locomo_my_run \
  --top-n-each 15 \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL \
  --timeout 120 --max-retries 3
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--top-n-each` | Top N records per route | 10 |
| `--max-merged` | Max records after dedup merge (0 = no limit) | 0 |

#### Step 7: Judge

```bash
python locomo/judge/run_judge_from_qa.py \
  --qa-root ./results/qa/locomo_my_run \
  --conv-json ./data/locomo10.json \
  --output-base ./results/judge \
  --run-name locomo_my_run \
  --base-url $BASE_URL --api-key $API_KEY --model-name $MODEL \
  --timeout 120 --max-tokens 512 --max-retries 3
```

Generates `report.md` with overall accuracy and breakdowns by conversation and category.

> Use `--conv-name conv-26` to judge a specific conversation only.

---

### Memory Update Module

Both benchmarks share an offline memory update module for contradiction detection and record consolidation:

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

Supported `--method`: `lightmem`, `dimmem`. Supported `--dataset`: `longmemeval`, `locomo`.

---

## Full Run Example

The project provides an end-to-end test script `run_test_41mini.sh` that runs the full 7-step pipeline (including compression) with `gpt-4.1-mini`:

```bash
bash run_test_41mini.sh
```

Configuration used:
- LongMemEval: `window_size=15, overlap=3`
- LoCoMo: `window_size=25, overlap=5`
- Compression rate: 0.8 (keep 80% text)
- Retrieval: three-route top-15
- Each step limited to `--max-items 1` / `--max-records 1` (test mode)

## Results Output

All intermediate and final results are saved under `./results/`:

```
results/
├── segments/                    # LongMemEval segmentation results
│   ├── raw/<run_name>/
│   └── compressed/<run_name>/
├── locomo_segments/             # LoCoMo segmentation results
│   ├── raw/<run_name>/
│   └── compressed/<run_name>/
├── memories/<run_name>/         # LongMemEval extracted memories
├── locomo_memory/<run_name>/    # LoCoMo extracted memories
├── query_analysis/
│   ├── <run_name>/              # LongMemEval query analysis
│   └── locomo_<run_name>/       # LoCoMo query analysis
├── retrieval/
│   ├── <run_name>/              # LongMemEval retrieval results
│   ├── bm25/<run_name>/         # LoCoMo BM25 retrieval
│   ├── minilm/<run_name>/       # LoCoMo MiniLM retrieval
│   └── structured/<run_name>/   # LoCoMo structured retrieval
├── qa/
│   ├── <run_name>/              # LongMemEval QA results
│   └── locomo_<run_name>/       # LoCoMo QA results
└── judge/
    ├── <run_name>/              # LongMemEval judge results
    └── locomo_<run_name>/       # LoCoMo judge results + report.md
```

Each result directory contains `run_command.sh` recording the exact command used, for reproducibility.

## Key Design Decisions

1. **Windowed Segmentation**: Conversations are split into overlapping windows (LongMemEval default: 25 messages / 20% overlap; LoCoMo default: 25 messages / 5 overlap) to maintain context while keeping LLM input manageable.

2. **Dimension Memory Model**: Each memory record carries structured dimension fields (type, time, location, reason, purpose, keywords), enabling multi-dimensional retrieval and precise matching.

3. **Multi-Route Retrieval Fusion**: Three retrieval paths (BM25 lexical, dense embedding, structured dimension matching) are fused via RRF (Reciprocal Rank Fusion), balancing recall and precision.

4. **Dynamic Assistant Context**: A query classifier (`need_assistant_context`) determines whether to attach original AI responses, saving ~89% of context lookups.

5. **Memory Update**: Offline contradiction detection and consolidation keeps the memory store consistent as new sessions arrive.

6. **Overlapping Context Rule**: The extraction prompt explicitly marks window overlap sections as context-only (not for new memory extraction), preventing cross-window duplication.

## Models Used

| Purpose | Model | Description |
|---------|-------|-------------|
| Memory Extraction / Query Analysis | Qwen3-30B-A3B or gpt-4.1-mini | Via OpenAI-compatible API |
| Embedding | all-MiniLM-L6-v2 (384-dim) | Local, sentence-transformers |
| QA + Judge | gpt-4.1-mini | Or any OpenAI-compatible model |
| Text Compression | LLMLingua-2 (bert-base-multilingual) | Local GPU inference |
