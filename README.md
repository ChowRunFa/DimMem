# DimMem Evaluation Pipeline

Memory-augmented QA evaluation on **LongMemEval** and **LoCoMo** benchmarks.

## Overview

The pipeline processes long conversations into structured memory records and evaluates QA accuracy through:

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
├── data/                  # Benchmark datasets (raw, not included)
│   ├── README.md                              # How to obtain the datasets
│   ├── longmemeval_s_cleaned.json             # LongMemEval: 500 items with question_type field
│   └── locomo10.json                          # LoCoMo: 10 conversations with qa + conversation
│
├── quick_start/           # Quickstart demo
│   └── quickstart_extract.py                  # Self-contained memory extraction demo (inline test data)
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
└── README.md
```

## Quick Start

Run the self-contained demo to understand dimension memory extraction without the full datasets:

```bash
python quick_start/quickstart_extract.py \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b \
  --demo both
```

The script includes inline test conversations for both LongMemEval and LoCoMo, calls the LLM to extract structured memories, and prints the normalized output with dimension fields (`memory_type`, `time`, `location`, `reason`, `purpose`, `keywords`).

Options: `--demo longmemeval`, `--demo locomo`, or `--demo both` (default).

## Results Output

All intermediate and final results are saved under `./results/` (auto-created). Example structure after a full run:

```
results/
├── segments/              # LongMemEval segmented windows
│   └── raw/<run_name>/
├── compressed/            # (Optional) compressed segments
│   └── <run_name>/
├── memories/              # LongMemEval extracted structured memories
│   └── <run_name>/
├── query_analysis/        # LongMemEval parsed queries
│   └── <run_name>/
├── retrieval/             # LongMemEval multi-route retrieval results
│   └── <run_name>/
├── qa/                    # LongMemEval QA answers
│   └── <run_name>/
├── judge/                 # LongMemEval judge verdicts
│   └── <run_name>/
│
├── segments/              # LoCoMo segmented windows
│   └── raw/<run_name>/
├── memory/                # LoCoMo extracted memories
│   └── <run_name>/
├── query_analysis/        # LoCoMo parsed queries (shared path with LongMemEval)
│   └── <run_name>/
├── retrieval/             # LoCoMo retrieval (bm25/minilm/structured) (shared path)
│   ├── bm25/<run_name>/
│   ├── minilm/<run_name>/
│   └── structured/<run_name>/
├── qa/                    # LoCoMo QA answers (shared path)
│   └── <run_name>/
└── judge/                 # LoCoMo judge verdicts (shared path)
    └── <run_name>/
```

**Note**: Both benchmarks can share the same `results/` directory. Use different `<run_name>` values to keep experiments separate, or use the same run name if processing different datasets in the same experiment.

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `requests`, `httpx`, `openai`, `tqdm`, `sentence-transformers`, `torch`

Optional: `onnxruntime` (for ONNX embedding inference), `llmlingua` (for compression step)

## Running the Pipeline

### Environment Variables

```bash
export OPENAI_API_BASE="http://your-vllm-server:8000/v1"   # vLLM / OpenAI-compatible endpoint
export OPENAI_API_KEY="EMPTY"                                # API key (or "EMPTY" for local vLLM)
export EMBEDDING_MODEL="/path/to/all-MiniLM-L6-v2"          # Local embedding model path
```

---

### LongMemEval Pipeline

#### Step 1: Segment conversations into windows

```bash
python longmemeval/segmenter/build_raw_segments.py \
  --input-path ./data/longmemeval_s_cleaned.json \
  --output-root ./results/segments/raw \
  --window-size 25 \
  --overlap-ratio 0.2
```

#### Step 2 (Optional): Compress segments

```bash
python longmemeval/compressor/build_compressed_segments.py \
  --raw-run-root ./results/segments/raw/<run_name> \
  --output-root ./results/compressed/ \
  --rate 0.5
```

#### Step 3: Extract structured memories

```bash
python longmemeval/memory_constructor/run_batch_extract.py \
  --segments-root ./results/segments/raw/<run_name> \
  --output-root ./results/memories \
  --run-name <run_name> \
  --overlap 5 \
  --base-url http://localhost:7790/v1 \
  --api-key EMPTY \
  --model-name qwen3-30b-a3b \
  --max-tokens 16384 \
  --timeout 600 \
  --max-retries 5
```

The memory extraction module (`longmemeval/memory_constructor/extract_helpers.py`) provides helper functions for prompt-building and normalization. See `quick_start/quickstart_extract.py` for a self-contained usage example.

#### Step 4: Query analysis

```bash
python longmemeval/query_parser/run_query_analysis.py \
  --input-root ./data/longmemeval_s_cleaned.json \
  --output-base ./results/query_analysis \
  --run-name <run_name> \
  --base-url http://localhost:7790/v1 \
  --api-key EMPTY \
  --model-name qwen3-30b-a3b \
  --max-samples 1
```

> Use `--max-samples 0` to process all samples, or specify a number to limit processing.

#### Step 5: Retrieval

```bash
python longmemeval/search/retrieve_from_parsed_query.py \
  --query-parsed ./results/query_analysis/<run_name>/<question_type>/<sample_id>/parsed.json \
  --memory-dir ./results/memories/<run_name>/<question_type>/<sample_id> \
  --output-root ./results/retrieval/<run_name> \
  --embedding-model /path/to/all-MiniLM-L6-v2
```

Example for a specific sample:

```bash
python longmemeval/search/retrieve_from_parsed_query.py \
  --query-parsed ./results/query_analysis/demo/single-session-user/0000_e47becba/parsed.json \
  --memory-dir ./results/memories/demo/single-session-user/0000_e47becba \
  --output-root ./results/retrieval/demo \
  --embedding-model /path/to/all-MiniLM-L6-v2
```

#### Steps 6+7: QA + Judge

```bash
python longmemeval/qa_judge/run_qa_judge_from_retrieval.py
```

**Important**: This script uses hardcoded paths and API configuration. Before running, edit the script to set:

- `RETRIEVAL_ROOT`: Path to retrieval results (e.g., `SUBMIT_ROOT / "results/retrieval/<run_name>/<question_type>/<sample_id>"`)
- `QUERY_ANALYSIS_ROOT`: Path to query analysis results (e.g., `SUBMIT_ROOT / "results/query_analysis/<run_name>"`)
- `QA_ROOT`: Output path for QA results (e.g., `SUBMIT_ROOT / "results/qa/<run_name>"`)
- `JUDGE_ROOT`: Output path for judge results (e.g., `SUBMIT_ROOT / "results/judge/<run_name>"`)
- `MODEL_NAME`, `BASE_URL`, `API_KEY`: LLM API configuration

The script iterates through `RETRIEVAL_ROOT/<question_type>/<method>/<sample_id>/top_records.json`, generates QA answers, and judges them against gold answers.

#### Generate Report

```bash
python longmemeval/qa_judge/run_report.py \
  --judge-root ./results/judge/<run_name>
```

Example:

```bash
python longmemeval/qa_judge/run_report.py \
  --judge-root ./results/judge/demo
```

Generates `report.md` with overall accuracy and breakdowns by question type and retrieval method.

---

### LoCoMo Pipeline

#### Step 1: Segment conversations

```bash
python locomo/segmenter/build_raw_segments.py \
  --input-root ./data/locomo10.json \
  --output-root ./results/locomo_segments/raw \
  --window-size 25 \
  --overlap 5
```

#### Step 2 (Optional): Compress segments

```bash
python locomo/compressor/build_compressed_segments.py \
  --raw-run-root ./results/locomo_segments/raw/<run_name> \
  --output-root ./results/locomo_segments/compressed \
  --rate 0.5
```

#### Step 3: Extract memories

Single record (parallel per window):

```bash
python locomo/memory_constructor/extract_single_record.py \
  --record-dir ./results/locomo_segments/raw/<run_name>/<conv>/ \
  --output-record-dir ./results/locomo_memory/<conv>/ \
  --overlap 5 \
  --base-urls http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b \
  --workers 4
```

Batch mode (all records sequentially):

```bash
python locomo/memory_constructor/run_batch_extract.py \
  --compressed-root ./results/locomo_segments/raw/<run_name> \
  --output-root ./results/locomo_memory \
  --run-name <run_name> \
  --overlap 5 \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b
```

> Note: `--compressed-root` accepts either raw or compressed segment directories.

#### Step 4: Query analysis

```bash
python locomo/query_parser/run_query_analysis_by_conv.py \
  --input-root ./data/locomo10.json \
  --output-base ./results/query_analysis \
  --run-name <run_name> \
  --base-url http://localhost:7790/v1 \
  --api-key EMPTY \
  --model-name qwen3-30b-a3b \
  --max-tokens 4096 \
  --timeout 600 \
  --max-retries 3 \
  --exclude-categories 5
```

Use `--exclude-categories 5` to skip category 5 questions. Use `--max-convs` and `--max-questions-per-conv` to limit processing (0 means all).

#### Step 5: Retrieval (multi-route)

```bash
# BM25
python locomo/search/run_retrieval_bm25.py \
  --query-run-root ./results/query_analysis/<run_name> \
  --memory-root ./results/memory \
  --output-base ./results/retrieval \
  --run-name <run_name>

# Dense (MiniLM)
python locomo/search/run_retrieval_minilm.py \
  --query-run-root ./results/query_analysis/<run_name> \
  --memory-root ./results/memory \
  --output-base ./results/retrieval \
  --run-name <run_name> \
  --embedding-model /path/to/all-MiniLM-L6-v2

# Structured
python locomo/search/run_retrieval_from_query_analysis.py \
  --query-run-root ./results/query_analysis/<run_name> \
  --memory-root ./results/memory \
  --output-base ./results/retrieval \
  --run-name <run_name>
```

#### Step 6: QA generation

**Option A: Single retrieval route**

```bash
python locomo/qa/run_qa_from_retrieval.py \
  --retrieval-root ./results/retrieval/<run_name> \
  --query-root ./results/query_analysis/<run_name> \
  --output-base ./results/qa \
  --run-name <run_name> \
  --base-url http://localhost:7790/v1 \
  --api-key EMPTY \
  --model-name qwen3-30b-a3b \
  --timeout 600 \
  --max-tokens 2048 \
  --max-retries 3
```

**Option B: Merge three retrieval routes**

```bash
python locomo/qa/run_qa_from_three_retrievals.py \
  --query-root ./results/query_analysis/<run_name> \
  --bm25-root ./results/retrieval/bm25/<run_name> \
  --minilm-root ./results/retrieval/minilm/<run_name> \
  --structured-root ./results/retrieval/structured/<run_name> \
  --output-base ./results/qa \
  --run-name <run_name> \
  --base-url http://localhost:7790/v1 \
  --api-key EMPTY \
  --model-name qwen3-30b-a3b \
  --timeout 600 \
  --max-retries 3
```

#### Step 7: Judge

```bash
python locomo/judge/run_judge_from_qa.py \
  --qa-root ./results/qa/<run_name> \
  --conv-json ./data/locomo10.json \
  --output-base ./results/judge \
  --run-name <run_name> \
  --base-url http://localhost:7790/v1 \
  --api-key EMPTY \
  --model-name gpt-4.1-mini \
  --timeout 600 \
  --max-tokens 512 \
  --max-retries 3
```

Use `--conv-name <conv_name>` to judge only a specific conversation (e.g., `--conv-name conv-26`).

Generates `report.md` with overall accuracy and breakdowns by conversation and category.

---

### Memory Update Module

Both benchmarks share an offline memory update module that detects contradictions and consolidates records:

```bash
python longmemeval/update/run_update.py \
  --method dimmem \
  --memory-root ./results/memories/<run_name> \
  --dataset longmemeval \
  --output ./results/update_output/ \
  --base-url http://localhost:7790/v1 \
  --api-key EMPTY \
  --model-name qwen3-30b-a3b
```

For LoCoMo:

```bash
python longmemeval/update/run_update.py \
  --method dimmem \
  --memory-root ./results/memory/<run_name> \
  --dataset locomo \
  --output ./results/update_output/ \
  --base-url http://localhost:7790/v1 \
  --api-key EMPTY \
  --model-name qwen3-30b-a3b
```

Supported `--method`: `lightmem`, `dimmem`. Supported `--dataset`: `longmemeval`, `locomo`.

---

## Key Design Decisions

1. **Windowed Segmentation**: Conversations are split into overlapping windows (default 25 messages, 20% overlap for LongMemEval / 5 messages overlap for LoCoMo) to maintain context while keeping LLM input manageable.

2. **Multi-Route Retrieval**: Three search paths (BM25 lexical, dense embedding, structured dimension matching) are fused via RRF for robust recall.

3. **Dynamic Assistant Context**: A query classifier (`need_assistant_context`) determines whether to attach original assistant replies to retrieved memories — saving ~89% of context lookups while maintaining accuracy on assistant-knowledge questions.

4. **Memory Update**: An offline contradiction detection + consolidation pass keeps the memory store consistent as new sessions arrive.

## Models Used

- **Memory Extraction / Query Analysis**: Qwen3-30B-A3B (or fine-tuned Qwen3-4B), via OpenAI-compatible API (e.g., vLLM, StepFun)
- **Embedding**: all-MiniLM-L6-v2 (384-dim), loaded locally via `sentence-transformers`
- **QA + Judge**: gpt-4.1-mini (or any OpenAI-compatible model)
