#!/usr/bin/env python3
"""Memory update ablation V2 — main entry point.

Usage:
  python -m longmemeval.src.update.run_update \
    --method lightmem \
    --memory-root <path> \
    --output results/ablation/memory_update_v2/lightmem_longmemeval/

  python -m longmemeval.src.update.run_update \
    --method dimmem \
    --memory-root <path> \
    --output results/ablation/memory_update_v2/dimmem_longmemeval/

Or run directly:
  python run_update.py --method lightmem --memory-root ... --output ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Allow running as script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from detect import load_embedding_model, detect_lightmem, detect_dimmem
from consolidate import process_pair, apply_decisions, DEFAULT_BASE_URL, DEFAULT_API_KEY, DEFAULT_MODEL


def _clean(v: Any) -> str:
    return str(v or "").strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _count_tokens_approx(text: str) -> int:
    words = len(text.split())
    chars = len(text)
    return max(words, int(chars / 3))


def _load_memories(path: Path) -> Any:
    """Load raw JSON (preserving wrapper dict if present)."""
    return json.loads(path.read_text(encoding="utf-8"))


def _get_mem_list(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict) and isinstance(raw.get("memories"), list):
        return raw["memories"]
    return raw if isinstance(raw, list) else []


# ─── Process one sample ──────────────────────────────────────────────────────

def process_sample(
    sample_key: str,
    mem_file: Path,
    method: str,
    embedder,
    output_root: Path,
    base_url: str,
    api_key: str,
    model_name: str,
) -> Dict[str, Any]:
    raw_data = _load_memories(mem_file)
    mem_list = _get_mem_list(raw_data)

    if len(mem_list) < 2:
        return {"sample_key": sample_key, "pair_count": 0, "skipped": True}

    # Detect conflicts
    if method == "lightmem":
        pairs, detect_stats = detect_lightmem(mem_list, embedder)
    else:
        pairs, detect_stats = detect_dimmem(mem_list, embedder)

    if not pairs:
        return {
            "sample_key": sample_key,
            "original_count": len(mem_list),
            "updated_count": len(mem_list),
            "pair_count": 0,
            "detect_stats": detect_stats,
            "action_stats": {"merge": 0, "supersede": 0, "keep_both": 0},
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_llm_tokens": 0,
        }

    # LLM decisions
    decisions = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    errors = 0

    for pair in pairs:
        try:
            result = process_pair(pair, method, base_url, api_key, model_name)
            decisions.append(result)
            total_prompt_tokens += result.get("prompt_tokens", 0)
            total_completion_tokens += result.get("completion_tokens", 0)
        except Exception as exc:
            errors += 1
            decisions.append({
                "idx_new": pair["idx_new"],
                "idx_old": pair["idx_old"],
                "decision": "KEEP_BOTH",
                "error": str(exc),
            })

    # Apply decisions
    updated_memories, action_stats = apply_decisions(mem_list, decisions, method)

    # Save updated memory bank
    safe_key = sample_key.replace("/", "_")
    sample_output = output_root / safe_key
    sample_output.mkdir(parents=True, exist_ok=True)

    if isinstance(raw_data, dict):
        updated_payload = dict(raw_data)
        updated_payload["memories"] = updated_memories
        updated_payload["_update_method"] = method
        updated_payload["_original_count"] = len(mem_list)
        updated_payload["_updated_count"] = len(updated_memories)
    else:
        updated_payload = {"memories": updated_memories, "_update_method": method}

    _write_json(sample_output / "all_memories.json", updated_payload)
    _write_json(sample_output / "decisions.json", decisions)

    original_tokens = sum(_count_tokens_approx(_clean(m.get("content"))) for m in mem_list)
    updated_tokens = sum(_count_tokens_approx(_clean(m.get("content"))) for m in updated_memories)

    return {
        "sample_key": sample_key,
        "original_count": len(mem_list),
        "updated_count": len(updated_memories),
        "pair_count": len(pairs),
        "detect_stats": detect_stats,
        "action_stats": action_stats,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_llm_tokens": total_prompt_tokens + total_completion_tokens,
        "original_content_tokens": original_tokens,
        "updated_content_tokens": updated_tokens,
        "errors": errors,
    }


# ─── Collect samples ─────────────────────────────────────────────────────────

def collect_longmemeval_samples(memory_root: Path) -> List[Dict[str, Any]]:
    """Collect LongMemEval samples: question_type/sample_id/all_memories.json"""
    samples = []
    for qt_dir in sorted(memory_root.iterdir()):
        if not qt_dir.is_dir() or qt_dir.name.startswith(("MERGE", "README")):
            continue
        for sample_dir in sorted(qt_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            mem_file = sample_dir / "all_memories.json"
            if mem_file.exists():
                samples.append({
                    "key": f"{qt_dir.name}/{sample_dir.name}",
                    "mem_file": mem_file,
                })
    return samples


def collect_locomo_samples(memory_root: Path) -> List[Dict[str, Any]]:
    """Collect LoCoMo samples: conv_name/all_memories.json"""
    samples = []
    for conv_dir in sorted(memory_root.iterdir()):
        if not conv_dir.is_dir():
            continue
        mem_file = conv_dir / "all_memories.json"
        if mem_file.exists():
            samples.append({
                "key": conv_dir.name,
                "mem_file": mem_file,
            })
    return samples


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Memory update ablation V2.")
    parser.add_argument("--method", required=True, choices=["lightmem", "dimmem"])
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True, choices=["longmemeval", "locomo"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    print(f"Loading embedding model...", flush=True)
    embedder = load_embedding_model()

    if args.dataset == "longmemeval":
        samples = collect_longmemeval_samples(args.memory_root)
    else:
        samples = collect_locomo_samples(args.memory_root)

    args.output.mkdir(parents=True, exist_ok=True)
    updated_root = args.output / "updated_memories"
    updated_root.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(samples)} samples with method={args.method} dataset={args.dataset}", flush=True)
    started = time.time()

    all_results = []
    done = 0
    total = len(samples)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for s in samples:
            fut = executor.submit(
                process_sample,
                s["key"], s["mem_file"], args.method, embedder,
                updated_root, args.base_url, args.api_key, args.model_name,
            )
            futures[fut] = s["key"]

        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as exc:
                all_results.append({
                    "sample_key": futures[future],
                    "error": str(exc),
                })
            done += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - started
                print(f"  [{done}/{total}] elapsed={elapsed:.0f}s", flush=True)

    # Aggregate stats
    total_original = sum(r.get("original_count", 0) for r in all_results if "original_count" in r)
    total_updated = sum(r.get("updated_count", 0) for r in all_results if "updated_count" in r)
    total_pairs = sum(r.get("pair_count", 0) for r in all_results)
    total_prompt_tokens = sum(r.get("total_prompt_tokens", 0) for r in all_results)
    total_completion_tokens = sum(r.get("total_completion_tokens", 0) for r in all_results)
    total_original_content_tokens = sum(r.get("original_content_tokens", 0) for r in all_results)
    total_updated_content_tokens = sum(r.get("updated_content_tokens", 0) for r in all_results)

    agg_actions = {"merge": 0, "supersede": 0, "keep_both": 0}
    for r in all_results:
        for k, v in (r.get("action_stats") or {}).items():
            agg_actions[k] = agg_actions.get(k, 0) + v

    # Aggregate detect stats
    total_embedding_count = sum(
        r.get("detect_stats", {}).get("embedding_count", 0) for r in all_results
    )
    total_pairs_checked = sum(
        r.get("detect_stats", {}).get("pairs_checked", 0) or
        r.get("detect_stats", {}).get("keyword_pairs_checked", 0)
        for r in all_results
    )
    total_keyword_candidates = sum(
        r.get("detect_stats", {}).get("keyword_candidates", 0) for r in all_results
    )
    total_time_filtered = sum(
        r.get("detect_stats", {}).get("time_filtered", 0) for r in all_results
    )

    memories_reduced = total_original - total_updated
    total_llm_tokens = total_prompt_tokens + total_completion_tokens
    tokens_per_reduced = round(total_llm_tokens / max(1, memories_reduced), 1)

    report = {
        "method": args.method,
        "dataset": args.dataset,
        "version": "v2",
        "created_at": datetime.now().isoformat(),
        "elapsed_seconds": round(time.time() - started, 1),
        "sample_count": len(all_results),
        # Memory counts
        "total_original_memories": total_original,
        "total_updated_memories": total_updated,
        "memory_reduction": memories_reduced,
        "compression_rate": round(memories_reduced / max(1, total_original) * 100, 2),
        # Detection efficiency
        "total_embedding_count": total_embedding_count,
        "total_pairs_checked": total_pairs_checked,
        "total_keyword_candidates": total_keyword_candidates,
        "total_time_filtered": total_time_filtered,
        "total_pairs_to_llm": total_pairs,
        # LLM cost
        "action_distribution": agg_actions,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_llm_tokens": total_llm_tokens,
        "tokens_per_memory_reduced": tokens_per_reduced,
        # Content
        "original_content_tokens": total_original_content_tokens,
        "updated_content_tokens": total_updated_content_tokens,
        "content_token_reduction": total_original_content_tokens - total_updated_content_tokens,
    }
    _write_json(args.output / "update_report.json", report)
    _write_json(args.output / "sample_results.json", all_results)

    print(f"\nDone in {report['elapsed_seconds']}s", flush=True)
    print(f"Original: {total_original} -> Updated: {total_updated} ({report['compression_rate']}%)", flush=True)
    print(f"Actions: {agg_actions}", flush=True)
    print(f"Embeddings computed: {total_embedding_count} / {total_original} total memories", flush=True)
    print(f"Pairs checked: {total_pairs_checked}, to LLM: {total_pairs}", flush=True)
    print(f"LLM tokens: {total_llm_tokens} ({tokens_per_reduced} per memory reduced)", flush=True)


if __name__ == "__main__":
    main()
