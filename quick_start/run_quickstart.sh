#!/bin/bash
# ===========================================================================
# DimMem Quick Start — End-to-End Pipeline Test (1 record each benchmark)
#
# Runs the full 7-step pipeline on ONE real LongMemEval record and ONE real
# LoCoMo conversation, including optional LLMLingua-2 compression.
#
# Usage:
#   # 1) Edit the configuration section below, then:
#   bash quick_start/run_quickstart.sh
#
#   # 2) Or override via environment variables:
#   BASE_URL="https://api.example.com/v1" API_KEY="sk-xxx" MODEL="gpt-4.1-mini" \
#     bash quick_start/run_quickstart.sh
#
#   # 3) Skip compression (no GPU / no llmlingua):
#   SKIP_COMPRESS=1 bash quick_start/run_quickstart.sh
# ===========================================================================
set -e

# ── locate project root (always resolve relative to this script) ──────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ======================== Configuration ====================================
# Override any of these via environment variables before running the script.

# LLM API (OpenAI-compatible)
BASE_URL="${BASE_URL:-https://xxx.xxxx.com/v1}"
API_KEY="${API_KEY:-sk-xxxxxxC}"
MODEL="${MODEL:-gpt-4.1-mini}"

# Local models
EMBED_MODEL="${EMBED_MODEL:-/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/models/all-MiniLM-L6-v2}"
COMPRESS_MODEL="${COMPRESS_MODEL:-/data/aios-weights/LLM-Lingua/llmlingua-2-bert-base-multilingual-cased-meetingbank}"

# Python for compression step (needs llmlingua + GPU torch)
# Set to empty string or SKIP_COMPRESS=1 to skip compression entirely.
PYTHON_COMPRESS="${PYTHON_COMPRESS:-/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/miniconda3/envs/dimmem_v2/bin/python}"

# Pipeline parameters
LME_WINDOW=15          # LongMemEval window size
LME_OVERLAP=3          # LongMemEval overlap
LOCOMO_WINDOW=25       # LoCoMo window size
LOCOMO_OVERLAP=5       # LoCoMo overlap
COMPRESS_RATE=0.8      # Compression rate (0.8 = keep 80%)
TOP_K=15               # Retrieval top-k per route
# MAX_MERGED removed — default is now no truncation

# Run name
RUN_NAME="${RUN_NAME:-quickstart}"

# Skip compression? (set to 1 to skip)
SKIP_COMPRESS="${SKIP_COMPRESS:-0}"

# ==========================================================================

RESULTS="./results"
rm -rf "$RESULTS"
mkdir -p "$RESULTS"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          DimMem Quick Start — Full Pipeline Test            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model    : $MODEL"
echo "║  API      : $BASE_URL"
echo "║  Run Name : $RUN_NAME"
echo "║  Compress : $([ "$SKIP_COMPRESS" = "1" ] && echo "SKIP" || echo "rate=$COMPRESS_RATE")"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Helper: determine which segment root to use ──────────────────────────
# If compression ran, use compressed; otherwise use raw.
lme_seg_root() {
    if [ "$SKIP_COMPRESS" = "1" ]; then
        echo "$RESULTS/segments/raw/$RUN_NAME"
    else
        echo "$RESULTS/segments/compressed/$RUN_NAME"
    fi
}
locomo_seg_root() {
    if [ "$SKIP_COMPRESS" = "1" ]; then
        echo "$RESULTS/locomo_segments/raw/$RUN_NAME"
    else
        echo "$RESULTS/locomo_segments/compressed/$RUN_NAME"
    fi
}

# ===========================================================================
# Step 1: Segmentation
# ===========================================================================
echo "============ Step 1/7: Segmentation ============"

python longmemeval/segmenter/build_raw_segments.py \
  --input-path ./data/longmemeval_s_cleaned.json \
  --output-root "$RESULTS/segments/raw" \
  --run-name "$RUN_NAME" \
  --window-size $LME_WINDOW --overlap $LME_OVERLAP --max-items 1

python locomo/segmenter/build_raw_segments.py \
  --input-root ./data/locomo10.json \
  --output-root "$RESULTS/locomo_segments/raw" \
  --run-name "$RUN_NAME" \
  --window-size $LOCOMO_WINDOW --overlap $LOCOMO_OVERLAP --max-items 1

echo "[OK] Step 1 done — segments created"

# ===========================================================================
# Step 2: Compression (optional)
# ===========================================================================
if [ "$SKIP_COMPRESS" = "1" ]; then
    echo "============ Step 2/7: Compression — SKIPPED ============"
else
    echo "============ Step 2/7: Compression (rate=$COMPRESS_RATE, GPU) ============"

    $PYTHON_COMPRESS longmemeval/compressor/build_compressed_segments.py \
      --raw-run-root "$RESULTS/segments/raw/$RUN_NAME" \
      --output-root "$RESULTS/segments/compressed" \
      --run-name "$RUN_NAME" \
      --model-name "$COMPRESS_MODEL" \
      --device-map cuda --rate $COMPRESS_RATE --max-records 1

    $PYTHON_COMPRESS locomo/compressor/build_compressed_segments.py \
      --raw-run-root "$RESULTS/locomo_segments/raw/$RUN_NAME" \
      --output-root "$RESULTS/locomo_segments/compressed" \
      --run-name "$RUN_NAME" \
      --model-name "$COMPRESS_MODEL" \
      --device-map cuda --rate $COMPRESS_RATE --max-records 1

    echo "[OK] Step 2 done — compression complete"
fi

# ===========================================================================
# Step 3: Memory Extraction
# ===========================================================================
echo "============ Step 3/7: Memory Extraction ============"

python longmemeval/memory_constructor/run_batch_extract.py \
  --segments-root "$(lme_seg_root)" \
  --output-root "$RESULTS/memories" \
  --run-name "$RUN_NAME" \
  --overlap $LME_OVERLAP \
  --base-url "$BASE_URL" --api-key "$API_KEY" --model-name "$MODEL" \
  --max-tokens 16384 --timeout 120 --max-retries 3 --max-records 1

python locomo/memory_constructor/run_batch_extract.py \
  --compressed-root "$(locomo_seg_root)" \
  --output-root "$RESULTS/locomo_memory" \
  --run-name "$RUN_NAME" \
  --overlap $LOCOMO_OVERLAP \
  --base-url "$BASE_URL" --api-key "$API_KEY" --model-name "$MODEL" \
  --max-tokens 16384 --timeout 120 --max-retries 3 --max-records 1

echo "[OK] Step 3 done — memories extracted"

# ===========================================================================
# Step 4: Query Analysis
# ===========================================================================
echo "============ Step 4/7: Query Analysis ============"

python longmemeval/query_parser/run_query_analysis.py \
  --input-root ./data/longmemeval_s_cleaned.json \
  --output-base "$RESULTS/query_analysis" \
  --run-name "$RUN_NAME" \
  --base-url "$BASE_URL" --api-key "$API_KEY" --model-name "$MODEL" \
  --max-tokens 4096 --timeout 120 --max-retries 3 \
  --max-convs 1 --max-questions-per-conv 1

python locomo/query_parser/run_query_analysis_by_conv.py \
  --input-root ./data/locomo10.json \
  --output-base "$RESULTS/query_analysis" \
  --run-name "locomo_$RUN_NAME" \
  --base-url "$BASE_URL" --api-key "$API_KEY" --model-name "$MODEL" \
  --max-tokens 4096 --timeout 120 --max-retries 3 \
  --max-convs 1 --max-questions-per-conv 0 \
  --exclude-categories 5

echo "[OK] Step 4 done — queries parsed"

# ===========================================================================
# Step 5: Retrieval
# ===========================================================================
echo "============ Step 5/7: Retrieval ============"

# ── LongMemEval: find the first sample automatically ──────────────────────
LME_QA_ROOT="$RESULTS/query_analysis/$RUN_NAME"
LME_FIRST_TYPE=$(ls "$LME_QA_ROOT" | grep -v '\.json$' | grep -v '\.sh$' | grep -v 'status' | grep -v 'summary' | head -1)
LME_FIRST_SAMPLE=$(ls "$LME_QA_ROOT/$LME_FIRST_TYPE" | head -1)

python longmemeval/search/retrieve_from_parsed_query.py \
  --query-parsed "$LME_QA_ROOT/$LME_FIRST_TYPE/$LME_FIRST_SAMPLE/parsed.json" \
  --memory-dir "$RESULTS/memories/$RUN_NAME/$LME_FIRST_TYPE/$LME_FIRST_SAMPLE" \
  --output-root "$RESULTS/retrieval/$RUN_NAME" \
  --top-k $TOP_K \
  --embedding-model "$EMBED_MODEL" --embedding-device cpu

# ── LoCoMo: 3-route retrieval ────────────────────────────────────────────
python locomo/search/run_retrieval_bm25.py \
  --query-run-root "$RESULTS/query_analysis/locomo_$RUN_NAME" \
  --memory-root "$RESULTS/locomo_memory/$RUN_NAME" \
  --output-base "$RESULTS/retrieval/bm25" \
  --run-name "$RUN_NAME" --top-k $TOP_K

python locomo/search/run_retrieval_minilm.py \
  --query-run-root "$RESULTS/query_analysis/locomo_$RUN_NAME" \
  --memory-root "$RESULTS/locomo_memory/$RUN_NAME" \
  --output-base "$RESULTS/retrieval/minilm" \
  --run-name "$RUN_NAME" --top-k $TOP_K \
  --embedding-model "$EMBED_MODEL"

python locomo/search/run_retrieval_from_query_analysis.py \
  --query-run-root "$RESULTS/query_analysis/locomo_$RUN_NAME" \
  --memory-root "$RESULTS/locomo_memory/$RUN_NAME" \
  --output-base "$RESULTS/retrieval/structured" \
  --run-name "$RUN_NAME" --top-k $TOP_K

echo "[OK] Step 5 done — retrieval complete"

# ===========================================================================
# Step 6: QA
# ===========================================================================
echo "============ Step 6/7: QA Generation ============"

# LME QA + Judge (combined script)
python longmemeval/qa_judge/run_qa_judge_from_retrieval.py \
  --retrieval-root "$RESULTS/retrieval/$RUN_NAME" \
  --query-root "$RESULTS/query_analysis/$RUN_NAME" \
  --output-base "$RESULTS" \
  --run-name "$RUN_NAME" \
  --base-url "$BASE_URL" --api-key "$API_KEY" --model-name "$MODEL"

# LoCoMo QA (merge 3 routes)
python locomo/qa/run_qa_from_three_retrievals.py \
  --query-root "$RESULTS/query_analysis/locomo_$RUN_NAME" \
  --bm25-root "$RESULTS/retrieval/bm25/$RUN_NAME" \
  --minilm-root "$RESULTS/retrieval/minilm/$RUN_NAME" \
  --structured-root "$RESULTS/retrieval/structured/$RUN_NAME" \
  --output-base "$RESULTS/qa" \
  --run-name "locomo_$RUN_NAME" \
  --top-n-each $TOP_K \
  --base-url "$BASE_URL" --api-key "$API_KEY" --model-name "$MODEL" \
  --timeout 120 --max-retries 3

echo "[OK] Step 6 done — QA answers generated"

# ===========================================================================
# Step 7: Judge
# ===========================================================================
echo "============ Step 7/7: Judge ============"

python locomo/judge/run_judge_from_qa.py \
  --qa-root "$RESULTS/qa/locomo_$RUN_NAME" \
  --conv-json ./data/locomo10.json \
  --output-base "$RESULTS/judge" \
  --run-name "locomo_$RUN_NAME" \
  --base-url "$BASE_URL" --api-key "$API_KEY" --model-name "$MODEL" \
  --timeout 120 --max-tokens 512 --max-retries 3

echo "[OK] Step 7 done — judging complete"

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ALL STEPS COMPLETE                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "=== LongMemEval Judge Result ==="
cat "$RESULTS/judge/$RUN_NAME/$LME_FIRST_TYPE/structured/$LME_FIRST_SAMPLE/summary.json" 2>/dev/null || echo "(not found)"
echo ""
echo "=== LoCoMo Judge Report ==="
cat "$RESULTS/judge/locomo_$RUN_NAME/report.md" 2>/dev/null || echo "(not found)"
echo ""
echo "=== All result directories ==="
find "$RESULTS" -mindepth 1 -maxdepth 3 -type d | sort
