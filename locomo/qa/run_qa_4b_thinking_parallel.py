#!/usr/bin/env python3
"""
Parallel QA using Qwen3-4B with thinking enabled.
Extracts answer from after </think> tag.
Runs 8 shards in parallel, each hitting its own vLLM port.
"""
from __future__ import annotations

import argparse
import json
import re
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Set file descriptor limit high for all processes
resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

THIS_FILE = Path(__file__).resolve()
LOCOMO_SRC_ROOT = THIS_FILE.parents[1]
if str(LOCOMO_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCOMO_SRC_ROOT))

from prompts.qa_prompts import build_qa_payload

TOKEN_RE = re.compile(r"\s+")


def _clean(v: Any) -> str:
    return str(v or "").strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_content(text: str) -> str:
    s = _clean(text).lower()
    return TOKEN_RE.sub(" ", s).strip()


def _chat_with_thinking(*, base_url: str, api_key: str, model_name: str, prompt: str, max_tokens: int, timeout: int) -> Dict[str, Any]:
    """Call vLLM with thinking enabled (default for Qwen3)."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,  # Qwen3 thinking works better with some temperature
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    session = requests.Session()
    session.trust_env = False
    resp = session.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _extract_message(resp_json: Dict[str, Any]) -> str:
    try:
        return _clean(resp_json["choices"][0]["message"]["content"])
    except Exception:
        return ""


def _parse_thinking_answer(raw_text: str) -> Dict[str, Any]:
    """Parse answer from model output with <think>...</think> prefix."""
    thinking = ""
    answer_part = raw_text

    # Extract thinking content
    if "<think>" in raw_text:
        think_end = raw_text.find("</think>")
        if think_end >= 0:
            think_start = raw_text.find("<think>") + len("<think>")
            thinking = raw_text[think_start:think_end].strip()
            answer_part = raw_text[think_end + len("</think>"):].strip()
        else:
            # Thinking not closed (truncated), try to get content after last complete sentence
            thinking = raw_text[raw_text.find("<think>") + len("<think>"):].strip()
            answer_part = ""

    # Parse "Reasoning:" and "Answer:" from the answer part
    reasoning = ""
    answer = answer_part.strip()
    for line in answer_part.splitlines():
        t = line.strip()
        if t.lower().startswith("reasoning:"):
            reasoning = t.split(":", 1)[1].strip()
        elif t.lower().startswith("answer:"):
            answer = t.split(":", 1)[1].strip()

    return {
        "thinking": thinking,
        "reasoning": reasoning,
        "answer": answer,
        "raw_text": raw_text,
    }


def _read_top_records(root: Path, conv: str, sample_id: str, top_n: int) -> List[Dict[str, Any]]:
    p = root / conv / sample_id / "top_records.json"
    if not p.exists():
        return []
    try:
        rows = _load_json(p)
    except Exception:
        return []
    if not isinstance(rows, list):
        return []
    return rows[:top_n]


def _merge_three_routes(
    *,
    conv: str,
    sample_id: str,
    roots: List[Tuple[str, Path]],
    top_n_each: int,
    max_merged: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen_content = set()
    route_counts: Dict[str, int] = {}

    for route_name, route_root in roots:
        rows = _read_top_records(route_root, conv, sample_id, top_n_each)
        route_counts[route_name] = len(rows)
        for r in rows:
            content_key = _normalize_content(_clean(r.get("content")))
            if not content_key:
                continue
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            item = dict(r)
            item["_route"] = route_name
            merged.append(item)
            if len(merged) >= max_merged:
                break
        if len(merged) >= max_merged:
            break

    return merged, {"route_input_counts": route_counts, "merged_count": len(merged)}


def _collect_samples(query_root: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for conv_dir in sorted(query_root.iterdir()):
        if not conv_dir.is_dir():
            continue
        for sample_dir in sorted(conv_dir.iterdir()):
            if (sample_dir / "input.json").exists():
                out.append((conv_dir.name, sample_dir.name))
    return out


def run_shard(
    shard_index: int,
    samples: List[Tuple[str, str]],
    query_root: Path,
    roots: List[Tuple[str, Path]],
    out_root: Path,
    base_url: str,
    api_key: str,
    model_name: str,
    top_n_each: int,
    max_merged: int,
    max_tokens: int,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    done = 0
    fail = 0
    started_all = time.time()

    for conv, sample_id in samples:
        qa_dir = out_root / conv / sample_id
        qa_dir.mkdir(parents=True, exist_ok=True)
        summary_path = qa_dir / "summary.json"
        if summary_path.exists():
            done += 1
            continue

        started = time.time()
        ok = False
        err = None
        qa_answer = ""
        try:
            input_json = query_root / conv / sample_id / "input.json"
            inp = _load_json(input_json)
            query = _clean(inp.get("question"))
            gold_answer = inp.get("answer")

            merged, merge_meta = _merge_three_routes(
                conv=conv,
                sample_id=sample_id,
                roots=roots,
                top_n_each=top_n_each,
                max_merged=max_merged,
            )

            payload = build_qa_payload(query=query, retrieved_records=merged)

            resp_json = None
            for attempt in range(1, max_retries + 1):
                try:
                    resp_json = _chat_with_thinking(
                        base_url=base_url,
                        api_key=api_key,
                        model_name=model_name,
                        prompt=payload["prompt"],
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
                    break
                except Exception as exc:
                    if attempt >= max_retries:
                        raise
                    time.sleep(min(3 * attempt, 10))

            qa_raw = _extract_message(resp_json)
            qa_parsed = _parse_thinking_answer(qa_raw)
            qa_answer = qa_parsed.get("answer", "")

            _write_json(qa_dir / "qa_raw_response.json", resp_json)
            _write_json(qa_dir / "qa_result.json", qa_parsed)
            _write_json(
                qa_dir / "summary.json",
                {
                    "conv_name": conv,
                    "sample_id": sample_id,
                    "query": query,
                    "gold_answer": gold_answer,
                    "qa_answer": qa_answer,
                    "thinking_length": len(qa_parsed.get("thinking", "")),
                    "ok": True,
                    "error": None,
                    "merge_meta": merge_meta,
                },
            )
            ok = True
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            _write_json(
                qa_dir / "summary.json",
                {
                    "conv_name": conv,
                    "sample_id": sample_id,
                    "qa_answer": "",
                    "ok": False,
                    "error": err,
                },
            )

        if ok:
            done += 1
        else:
            fail += 1

        if (done + fail) % 20 == 0:
            print(f"  [shard {shard_index}] {done+fail}/{len(samples)} done={done} fail={fail}")

    return {
        "shard_index": shard_index,
        "total": len(samples),
        "done": done,
        "fail": fail,
        "elapsed_seconds": time.time() - started_all,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel QA with Qwen3-4B thinking on SFT memories")
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--structured-root", type=Path, required=True)
    parser.add_argument("--minilm-root", type=Path, required=True)
    parser.add_argument("--bm25-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ports", default="7790,7791,7792,7793,7794,7795,7796,7797")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="/data/aios-weights/Qwen/Qwen3-4B")
    parser.add_argument("--top-n-each", type=int, default=15)
    parser.add_argument("--max-merged", type=int, default=45)
    parser.add_argument("--max-tokens", type=int, default=16384, help="Large enough for thinking + answer")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    ports = [p.strip() for p in args.ports.split(",") if p.strip()]
    num_shards = len(ports)

    all_samples = _collect_samples(args.query_root)
    print(f"Total samples: {len(all_samples)}, shards: {num_shards}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "run_manifest.json", {
        "created_at": datetime.now().isoformat(),
        "query_root": str(args.query_root),
        "structured_root": str(args.structured_root),
        "minilm_root": str(args.minilm_root),
        "bm25_root": str(args.bm25_root),
        "output_root": str(args.output_root),
        "model_name": args.model_name,
        "ports": ports,
        "num_shards": num_shards,
        "top_n_each": args.top_n_each,
        "max_merged": args.max_merged,
        "max_tokens": args.max_tokens,
        "thinking_enabled": True,
    })

    roots = [
        ("structured", args.structured_root),
        ("minilm", args.minilm_root),
        ("bm25", args.bm25_root),
    ]

    # Split samples into shards
    shards = [[] for _ in range(num_shards)]
    for i, s in enumerate(all_samples):
        shards[i % num_shards].append(s)

    # Run shards in parallel processes
    with ProcessPoolExecutor(max_workers=num_shards) as ex:
        futures = []
        for shard_idx in range(num_shards):
            base_url = f"http://127.0.0.1:{ports[shard_idx]}/v1"
            f = ex.submit(
                run_shard,
                shard_idx,
                shards[shard_idx],
                args.query_root,
                roots,
                args.output_root,
                base_url,
                args.api_key,
                args.model_name,
                args.top_n_each,
                args.max_merged,
                args.max_tokens,
                args.timeout,
                args.max_retries,
            )
            futures.append(f)

        results = []
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            print(f"  Shard {r['shard_index']} done: {r['done']}/{r['total']} in {r['elapsed_seconds']:.0f}s")

    total_done = sum(r["done"] for r in results)
    total_fail = sum(r["fail"] for r in results)
    print(f"\n=== ALL DONE ===")
    print(f"Total: {total_done + total_fail}, OK: {total_done}, FAIL: {total_fail}")
    print(f"Output: {args.output_root}")


if __name__ == "__main__":
    main()
