#!/bin/bash
# Run memory construction for all records in parallel using 8 vLLM ports

COMPRESSED_ROOT="/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/DimMem/evaluation/dimmem/locomo/results/segment_results/compressed/20260426_by_conv_compressed"
OUTPUT_ROOT="/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/DimMem/evaluation/dimmem/locomo/results/memory_results/opd-4b-5702-teacher-32b-ckpt299"
MODEL_NAME="/data/aios-weights/Qwen/Qwen3-4B-opd-5702-teacher-32b-ckpt299"
PORTS="7790,7791,7792,7793,7794,7795,7796,7797"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/build_one_record_parallel.py"

mkdir -p "$OUTPUT_ROOT"

# Find all record dirs (those with summary.json)
RECORD_DIRS=$(find "$COMPRESSED_ROOT" -name "summary.json" -exec dirname {} \; | sort)

echo "Found records:"
echo "$RECORD_DIRS"
echo "---"
echo "Output: $OUTPUT_ROOT"
echo "Model: $MODEL_NAME"
echo "Ports: $PORTS"
echo "---"

PIDS=""
for RECORD_DIR in $RECORD_DIRS; do
    # Get the conv name (e.g., Audrey-conv44)
    CONV_NAME=$(basename "$(dirname "$RECORD_DIR")")
    OUT_DIR="${OUTPUT_ROOT}/${CONV_NAME}"

    # Skip if already done
    if [ -f "${OUT_DIR}/summary.json" ]; then
        echo "SKIP (already done): $CONV_NAME"
        continue
    fi

    echo "Starting: $CONV_NAME -> $OUT_DIR"
    python3 "$SCRIPT" \
        --record-dir "$RECORD_DIR" \
        --output-record-dir "$OUT_DIR" \
        --ports "$PORTS" \
        --model-name "$MODEL_NAME" \
        --max-tokens 16384 \
        --timeout 600 \
        --max-retries 3 \
        --workers 8 &
    PIDS="$PIDS $!"
done

echo "---"
echo "Waiting for all records to complete... PIDs: $PIDS"

# Wait for all
FAIL=0
for PID in $PIDS; do
    wait $PID || FAIL=$((FAIL+1))
done

echo "=== ALL DONE ==="
echo "Failed: $FAIL"
echo "Output dir: $OUTPUT_ROOT"
