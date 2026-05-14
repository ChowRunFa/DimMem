#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
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


def _chat_with_timeout(*, base_url: str, api_key: str, model_name: str, prompt: str, timeout: int) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
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


def _parse_answer(raw_text: str) -> Dict[str, Any]:
    reasoning = ""
    answer = raw_text.strip()
    for line in raw_text.splitlines():
        t = line.strip()
        if t.lower().startswith("reasoning:"):
            reasoning = t.split(":", 1)[1].strip()
        elif t.lower().startswith("answer:"):
            answer = t.split(":", 1)[1].strip()
    return {"reasoning": reasoning, "answer": answer, "raw_text": raw_text}


def _collect_samples(query_root: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for conv_dir in sorted(query_root.iterdir()):
        if not conv_dir.is_dir():
            continue
        for sample_dir in sorted(conv_dir.iterdir()):
            if (sample_dir / "input.json").exists():
                out.append((conv_dir.name, sample_dir.name))
    return out


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
            if max_merged > 0 and len(merged) >= max_merged:
                break
        if max_merged > 0 and len(merged) >= max_merged:
            break

    return merged, {"route_input_counts": route_counts, "merged_count": len(merged)}


def run(args: argparse.Namespace) -> Path:
    out_root = args.output_base / (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_root.mkdir(parents=True, exist_ok=True)

    all_samples = _collect_samples(args.query_root)
    if args.num_shards > 1:
        samples = [s for i, s in enumerate(all_samples) if i % args.num_shards == args.shard_index]
    else:
        samples = all_samples

    roots = [
        ("structured", args.structured_root),
        ("minilm", args.minilm_root),
        ("bm25", args.bm25_root),
    ]

    shard_tag = f"shard_{args.shard_index:02d}"
    shard_status = out_root / "_shards" / f"{shard_tag}.status.json"
    shard_result = out_root / "_shards" / f"{shard_tag}.result.json"

    if args.shard_index == 0:
        _write_json(
            out_root / "run_manifest.json",
            {
                "created_at": datetime.now().isoformat(),
                "query_root": str(args.query_root),
                "structured_root": str(args.structured_root),
                "minilm_root": str(args.minilm_root),
                "bm25_root": str(args.bm25_root),
                "output_root": str(out_root),
                "top_n_each": args.top_n_each,
                "max_merged": args.max_merged,
                "dedup_key": "content",
                "model_name": args.model_name,
                "base_url": args.base_url,
                "num_shards": args.num_shards,
            },
        )

    done = 0
    fail = 0
    rows: List[Dict[str, Any]] = []
    started_all = time.time()

    for conv, sample_id in samples:
        qa_dir = out_root / conv / sample_id
        qa_dir.mkdir(parents=True, exist_ok=True)
        summary_path = qa_dir / "summary.json"
        if args.resume and summary_path.exists():
            done += 1
            continue

        started = time.time()
        ok = False
        err = None
        qa_answer = ""
        try:
            input_json = args.query_root / conv / sample_id / "input.json"
            inp = _load_json(input_json)
            query = _clean(inp.get("question"))
            gold_answer = inp.get("answer")

            merged, merge_meta = _merge_three_routes(
                conv=conv,
                sample_id=sample_id,
                roots=roots,
                top_n_each=args.top_n_each,
                max_merged=args.max_merged,
            )

            payload = build_qa_payload(query=query, retrieved_records=merged)
            (qa_dir / "qa_prompt.txt").write_text(payload["prompt"], encoding="utf-8")
            _write_json(
                qa_dir / "qa_request.json",
                {
                    "model_name": args.model_name,
                    "query": query,
                    "gold_answer": gold_answer,
                    "merge_meta": merge_meta,
                    "retrieved_records": merged,
                    "roots": {k: str(v) for k, v in roots},
                },
            )

            resp_json = None
            last_exc: Exception | None = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    resp_json = _chat_with_timeout(
                        base_url=args.base_url,
                        api_key=args.api_key,
                        model_name=args.model_name,
                        prompt=payload["prompt"],
                        timeout=args.timeout,
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt >= args.max_retries:
                        raise
                    time.sleep(min(3 * attempt, 10))

            qa_raw = _extract_message(resp_json)
            qa_parsed = _parse_answer(qa_raw)
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
        rows.append(
            {
                "conv_name": conv,
                "sample_id": sample_id,
                "ok": ok,
                "error": err,
                "qa_dir": str(qa_dir),
                "qa_answer": qa_answer,
                "elapsed_seconds": time.time() - started,
            }
        )
        _write_json(
            shard_status,
            {
                "worker_name": args.worker_name,
                "total": len(samples),
                "done": done,
                "fail": fail,
                "running": {"conv_name": conv, "sample_id": sample_id},
                "updated_at": time.time(),
            },
        )

    final = {
        "worker_name": args.worker_name,
        "total": len(samples),
        "done": done,
        "fail": fail,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "elapsed_seconds": time.time() - started_all,
        "rows": rows,
    }
    _write_json(shard_result, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QA from three retrieval routes (topN each, dedup by content).")
    p.add_argument("--query-root", type=Path, required=True)
    p.add_argument("--structured-root", type=Path, required=True)
    p.add_argument("--minilm-root", type=Path, required=True)
    p.add_argument("--bm25-root", type=Path, required=True)
    p.add_argument("--output-base", type=Path, required=True)
    p.add_argument("--run-name", default="")
    p.add_argument("--top-n-each", type=int, default=10)
    p.add_argument("--max-merged", type=int, default=0, help="Max merged records (0 = no limit)")
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model-name", default="qwen3-30b-a3b")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--worker-name", default="")
    p.add_argument("--resume", action="store_true", default=True)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
