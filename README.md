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
├── compressed/            # (Optional) compressed segments
├── memories/              # Extracted structured memories
├── query_analysis/        # Parsed queries
├── retrieval/             # Multi-route retrieval results
├── qa/                    # QA answers
├── judge/                 # Judge verdicts
│
├── locomo_segments/       # LoCoMo segmented windows
├── locomo_memory/         # LoCoMo extracted memories
├── locomo_query_analysis/ # LoCoMo parsed queries
├── locomo_retrieval/      # LoCoMo retrieval (bm25/minilm/structured)
├── locomo_qa/             # LoCoMo QA answers
└── locomo_judge/          # LoCoMo judge verdicts
```

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

The memory extraction module (`longmemeval/memory_constructor/run_extract_windows_with_en_prompt.py`) is a library that provides prompt-building and normalization functions. It is imported by the extraction pipeline scripts rather than invoked directly from the CLI. See `quick_start/quickstart_extract.py` for a self-contained usage example.

Key functions:
- `_build_prompt(conversation, window_index, overlap_count)` — fills the extraction prompt template
- `_call_chat(base_url, api_key, model_name, prompt, max_tokens)` — calls the LLM
- `_safe_json_fragment(text)` — parses JSON from LLM response
- `_normalize_memory_entry(row, source_time_map)` — normalizes each extracted memory with `DimensionMemory`

#### Step 4: Query analysis

```bash
python longmemeval/query_parser/run_query_analysis.py \
  --input-root ./data/longmemeval_s_cleaned.json \
  --output-base ./results/query_analysis \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b
```

#### Step 5: Retrieval

```bash
python longmemeval/search/retrieve_from_parsed_query.py \
  --query-parsed ./results/query_analysis/<question_type>/<sample_id>/parsed.json \
  --memory-dir ./results/memories/<question_type>/<sample_id> \
  --output-root ./results/retrieval \
  --embedding-model /path/to/all-MiniLM-L6-v2
```

#### Steps 6+7: QA + Judge

```bash
python longmemeval/qa_judge/run_qa_judge_from_retrieval.py
```

This script reads from hardcoded paths under `results/` (`retrieval/`, `query_analysis/`, `qa/`, `judge/`). Ensure the previous steps have populated these directories.

#### Generate Report

```bash
python longmemeval/qa_judge/run_report.py \
  --judge-root ./results/judge/<run_name>
```

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

#### Step 3: Extract memories (parallel)

```bash
python locomo/memory_constructor/build_one_record_parallel.py \
  --record-dir ./results/locomo_segments/raw/<run_name>/<conv>/ \
  --output-record-dir ./results/locomo_memory/<conv>/ \
  --base-urls http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b \
  --workers 4
```

Or process all records from compressed segments sequentially:

```bash
python locomo/memory_constructor/build_memories_from_compressed.py \
  --compressed-root ./results/locomo_segments/compressed/<run_name> \
  --output-root ./results/locomo_memory \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b
```

#### Step 4: Query analysis

```bash
python locomo/query_parser/run_query_analysis_by_conv.py \
  --input-root ./data/locomo10.json \
  --output-base ./results/locomo_query_analysis \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b
```

#### Step 5: Retrieval (multi-route)

```bash
# BM25
python locomo/search/run_retrieval_bm25.py \
  --query-run-root ./results/locomo_query_analysis/<run_name> \
  --memory-root ./results/locomo_memory \
  --output-base ./results/locomo_retrieval

# Dense (MiniLM)
python locomo/search/run_retrieval_minilm.py \
  --query-run-root ./results/locomo_query_analysis/<run_name> \
  --memory-root ./results/locomo_memory \
  --output-base ./results/locomo_retrieval \
  --embedding-model /path/to/all-MiniLM-L6-v2

# Structured
python locomo/search/run_retrieval_from_query_analysis.py \
  --query-run-root ./results/locomo_query_analysis/<run_name> \
  --memory-root ./results/locomo_memory \
  --output-base ./results/locomo_retrieval
```

#### Step 6: QA generation

```bash
python locomo/qa/run_qa_from_three_retrievals.py \
  --query-root ./results/locomo_query_analysis/<run_name> \
  --bm25-root ./results/locomo_retrieval/bm25/<run_name> \
  --minilm-root ./results/locomo_retrieval/minilm/<run_name> \
  --structured-root ./results/locomo_retrieval/structured/<run_name> \
  --output-base ./results/locomo_qa \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b
```

#### Step 7: Judge

```bash
python locomo/judge/run_judge_from_qa.py \
  --qa-root ./results/locomo_qa/<run_name> \
  --conv-json ./data/locomo10.json \
  --output-base ./results/locomo_judge \
  --base-url https://api.example.com/v1 \
  --api-key <key> \
  --model-name gpt-4.1-mini
```

---

### Memory Update Module

Both benchmarks share an offline memory update module that detects contradictions and consolidates records:

```bash
python longmemeval/update/run_update.py \
  --method dimmem \
  --memory-root ./results/memories/ \
  --dataset longmemeval \
  --output ./results/update_output/ \
  --base-url http://localhost:7790/v1 \
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

- **Memory Extraction**: Qwen3-30B-A3B (or fine-tuned Qwen3-4B)
- **Embedding**: all-MiniLM-L6-v2 (384-dim)
- **QA + Judge**: gpt-4.1-mini (or any OpenAI-compatible model)
