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
├── longmemeval/           # LongMemEval benchmark pipeline
│   ├── utils/             # LocalEmbeddingClient (sentence-transformers)
│   ├── segmenter/         # Step 1: Windowed segmentation
│   ├── compressor/        # Step 2: LLMLingua-2 compression
│   ├── prompts/           # Prompt templates (extraction, QA, judge, query)
│   ├── search/            # Step 5: Multi-route retrieval (BM25, embedding, structured, fusion)
│   ├── update/            # Memory update detection & consolidation
│   ├── qa_judge/          # Step 7: Report generation
│   ├── run_extract_windows_with_en_prompt.py   # Step 3: Memory extraction
│   ├── retrieve_from_parsed_query.py           # Step 5: Retrieval orchestrator
│   └── run_qa_judge_from_retrieval.py          # Steps 6+7: QA & Judge
│
├── locomo/                # LoCoMo benchmark pipeline
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

## Results Output

All intermediate and final results are saved under `./results/` (auto-created). Example structure after a full run:

```
results/
├── segments/              # LongMemEval segmented windows
├── compressed/            # (Optional) compressed segments
├── memories/              # Extracted structured memories
├── query_analysis/        # Parsed queries
├── retrieval/             # Multi-route retrieval results
├── qa_judge/              # QA answers + judge verdicts + report.md
│
├── locomo_segments/       # LoCoMo segmented windows
├── locomo_memory/         # LoCoMo extracted memories
├── locomo_query_analysis/ # LoCoMo parsed queries
├── locomo_retrieval/      # LoCoMo retrieval (bm25/minilm/structured)
├── locomo_qa/             # LoCoMo QA answers
└── locomo_judge/          # LoCoMo judge verdicts + report.md
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
  --output-root ./results/segments/ \
  --window-size 25 \
  --overlap-ratio 0.2
```

#### Step 2 (Optional): Compress segments

```bash
python longmemeval/compressor/build_compressed_segments.py \
  --input-root ./results/segments/<run_name> \
  --output-root ./results/compressed/ \
  --rate 0.5
```

#### Step 3: Extract structured memories

```bash
python longmemeval/run_extract_windows_with_en_prompt.py \
  --input-root ./results/segments/<run_name> \
  --output-root ./results/memories/ \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b \
  --max-tokens 4096 \
  --workers 8
```

#### Step 4: Query analysis

```bash
python longmemeval/run_qa_judge_from_retrieval.py \
  --mode query-analysis \
  --data-dir ./data/longmemeval_s_cleaned.json \
  --output-root ./results/query_analysis/ \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b
```

#### Step 5: Retrieval

```bash
python longmemeval/retrieve_from_parsed_query.py \
  --query-parsed ./results/query_analysis/<sample>/parsed.json \
  --memory-dir ./results/memories/<sample>/ \
  --output-root ./results/retrieval/ \
  --embedding-model /path/to/all-MiniLM-L6-v2
```

#### Steps 6+7: QA + Judge

```bash
python longmemeval/run_qa_judge_from_retrieval.py \
  --mode qa-judge \
  --retrieval-root ./results/retrieval/<run_name> \
  --output-root ./results/qa_judge/ \
  --base-url https://api.example.com/v1 \
  --api-key <key> \
  --model-name gpt-4.1-mini
```

#### Generate Report

```bash
python longmemeval/qa_judge/run_report.py \
  --judge-root ./results/qa_judge/<run_name>
```

---

### LoCoMo Pipeline

#### Step 1: Segment conversations

```bash
python locomo/segmenter/build_raw_segments.py \
  --input-root ./data/locomo10.json \
  --output-root ./results/locomo_segments/ \
  --window-size 25 \
  --overlap 5
```

#### Step 3: Extract memories (parallel)

```bash
python locomo/memory_constructor/build_one_record_parallel.py \
  --input-dir ./results/locomo_segments/<run>/<conv>/windows/ \
  --output-dir ./results/locomo_memories/<conv>/ \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b \
  --workers 8
```

#### Step 4: Query analysis

```bash
python locomo/query_parser/run_query_analysis_by_conv.py \
  --input-root ./data/locomo10.json \
  --output-base ./results/locomo_query_analysis/ \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b
```

#### Step 5: Retrieval (multi-route)

```bash
# BM25
python locomo/search/run_retrieval_bm25.py \
  --memory-root ./results/locomo_memories/<conv>/ \
  --query-root ./results/locomo_query_analysis/<conv>/ \
  --output-root ./results/locomo_retrieval/bm25/

# Dense (MiniLM)
python locomo/search/run_retrieval_minilm.py \
  --memory-root ./results/locomo_memories/<conv>/ \
  --query-root ./results/locomo_query_analysis/<conv>/ \
  --output-root ./results/locomo_retrieval/minilm/ \
  --embedding-model /path/to/all-MiniLM-L6-v2

# Structured
python locomo/search/run_retrieval_from_query_analysis.py \
  --memory-root ./results/locomo_memories/<conv>/ \
  --query-root ./results/locomo_query_analysis/<conv>/ \
  --output-root ./results/locomo_retrieval/structured/
```

#### Step 6: QA generation

```bash
python locomo/qa/run_qa_from_three_retrievals.py \
  --bm25-root ./results/locomo_retrieval/bm25/<conv>/ \
  --minilm-root ./results/locomo_retrieval/minilm/<conv>/ \
  --structured-root ./results/locomo_retrieval/structured/<conv>/ \
  --conv-json ./data/locomo10.json \
  --output-root ./results/locomo_qa/ \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b \
  --top-k 15
```

#### Step 7: Judge

```bash
python locomo/judge/run_judge_from_qa.py \
  --qa-root ./results/locomo_qa/ \
  --conv-json ./data/locomo10.json \
  --conv-name <conv> \
  --output-base ./results/locomo_judge/ \
  --run-name <conv> \
  --base-url https://api.example.com/v1 \
  --api-key <key> \
  --model-name gpt-4.1-mini
```

---

### Memory Update Module

Both benchmarks include an offline memory update module that detects contradictions and consolidates records:

```bash
python longmemeval/update/run_update.py \
  --memory-dir ./results/memories/<sample>/ \
  --base-url http://localhost:7790/v1 \
  --model-name qwen3-30b-a3b
```

---

### SFT Data Generation (LoCoMo)

Scripts for generating training data for memory extraction fine-tuning:

```bash
# Generate structured memory SFT pairs from teacher model outputs
python locomo/data_generation/build_sft_dataset.py \
  --input-dir ./teacher_outputs/ \
  --output-path ./sft_data.jsonl

# Run SFT training
bash locomo/data_generation/sft.sh
```

## Key Design Decisions

1. **Windowed Segmentation**: Conversations are split into overlapping windows (default 25 messages, 20% overlap) to maintain context while keeping LLM input manageable.

2. **Multi-Route Retrieval**: Three search paths (BM25 lexical, dense embedding, structured dimension matching) are fused via RRF for robust recall.

3. **Dynamic Assistant Context**: A query classifier (`need_assistant_context`) determines whether to attach original assistant replies to retrieved memories — saving ~89% of context lookups while maintaining accuracy on assistant-knowledge questions.

4. **Memory Update**: An offline contradiction detection + consolidation pass keeps the memory store consistent as new sessions arrive.

## Models Used

- **Memory Extraction**: Qwen3-30B-A3B (or fine-tuned Qwen3-4B)
- **Embedding**: all-MiniLM-L6-v2 (384-dim)
- **QA + Judge**: gpt-4.1-mini (or any OpenAI-compatible model)
