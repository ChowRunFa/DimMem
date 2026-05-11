#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

THIS_FILE = Path(__file__).resolve()
LOCOMO_SRC_ROOT = THIS_FILE.parents[1]
if str(LOCOMO_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCOMO_SRC_ROOT))

from prompts.qa_prompts import build_qa_payload


def _clean(v: Any) -> str:
    return str(v or "").strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_text(resp_json: Dict[str, Any]) -> str:
    try:
        return _clean(resp_json["choices"][0]["message"]["content"])
    except Exception:
        return ""


def _parse_answer(raw_text: str) -> Dict[str, str]:
    reasoning = ""
    answer = _clean(raw_text)
    for line in raw_text.splitlines():
        t = line.strip()
        if t.lower().startswith("reasoning:"):
            reasoning = t.split(":", 1)[1].strip()
        elif t.lower().startswith("answer:"):
            answer = t.split(":", 1)[1].strip()
    return {"reasoning": reasoning, "answer": answer, "raw_text": raw_text}


def _call_chat(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    timeout: int,
    max_tokens: int,
) -> Dict[str, Any]:
    session = requests.Session()
    session.trust_env = False
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = session.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def run(args: argparse.Namespace) -> Path:
    out_root = args.output_base / (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_root.mkdir(parents=True, exist_ok=True)

    top_files = sorted(args.retrieval_root.glob("*/*/top_records.json"))
    if args.num_shards > 1:
        top_files = [p for i, p in enumerate(top_files) if i % args.num_shards == args.shard_index]
    done = 0
    fail = 0
    rows: List[Dict[str, Any]] = []

    if not args.no_global_files:
        _write_json(
            out_root / "run_manifest.json",
            {
                "created_at": datetime.now().isoformat(),
                "retrieval_root": str(args.retrieval_root),
                "query_root": str(args.query_root),
                "output_root": str(out_root),
                "model_name": args.model_name,
                "base_url": args.base_url,
                "timeout": args.timeout,
                "max_retries": args.max_retries,
                "max_tokens": args.max_tokens,
                "top_files": len(top_files),
                "prompt_file": "locomo/src/prompts/qa_prompts.py",
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            },
        )

    for top_file in top_files:
        conv_name = top_file.parent.parent.name
        sample_id = top_file.parent.name
        out_dir = out_root / conv_name / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.resume and (out_dir / "summary.json").exists():
            done += 1
            continue

        query_input = args.query_root / conv_name / sample_id / "input.json"
        if not query_input.exists():
            rec = {
                "conv_name": conv_name,
                "sample_id": sample_id,
                "ok": False,
                "error": f"missing_query_input: {query_input}",
                "output_dir": str(out_dir),
            }
            _write_json(out_dir / "result.json", rec)
            rows.append(rec)
            fail += 1
            continue

        row_input = _load_json(query_input)
        query = _clean(row_input.get("question"))
        retrieved = _load_json(top_file)
        if not isinstance(retrieved, list):
            retrieved = []
        qa_payload = build_qa_payload(query=query, retrieved_records=retrieved)
        (out_dir / "qa_prompt.txt").write_text(qa_payload["prompt"], encoding="utf-8")
        _write_json(out_dir / "qa_request.json", {"query": query, "retrieval_file": str(top_file), "retrieved_records": retrieved})

        ok = False
        err = None
        response_json: Dict[str, Any] | None = None
        qa_parsed: Dict[str, Any] | None = None
        started = time.time()
        for attempt in range(1, max(1, args.max_retries) + 1):
            try:
                response_json = _call_chat(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model_name=args.model_name,
                    prompt=qa_payload["prompt"],
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                )
                qa_raw = _extract_text(response_json)
                qa_parsed = _parse_answer(qa_raw)
                ok = True
                err = None
                break
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_retries:
                    time.sleep(min(2 * attempt, 8))

        if response_json is not None:
            _write_json(out_dir / "qa_raw_response.json", response_json)
        if qa_parsed is not None:
            _write_json(out_dir / "qa_result.json", qa_parsed)

        rec = {
            "conv_name": conv_name,
            "sample_id": sample_id,
            "ok": ok,
            "error": err,
            "elapsed_seconds": time.time() - started,
            "qa_answer": (qa_parsed or {}).get("answer", ""),
            "output_dir": str(out_dir),
        }
        _write_json(out_dir / "summary.json", rec)
        _write_json(out_dir / "result.json", rec)
        rows.append(rec)
        if ok:
            done += 1
        else:
            fail += 1

        if not args.no_global_files:
            _write_json(
                out_root / "status.json",
                {
                    "state": "running",
                    "total": len(top_files),
                    "done": done,
                    "fail": fail,
                    "running": {"conv_name": conv_name, "sample_id": sample_id},
                    "updated_at": datetime.now().isoformat(),
                },
            )

    final = {
        "state": "completed",
        "total": len(top_files),
        "done": done,
        "fail": fail,
        "rows": rows,
    }
    if not args.no_global_files:
        _write_json(out_root / "summary.json", final)
        _write_json(
            out_root / "status.json",
            {
                "state": "completed",
                "total": len(top_files),
                "done": done,
                "fail": fail,
                "updated_at": datetime.now().isoformat(),
            },
        )
    print(str(out_root))
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LoCoMo QA from retrieval results.")
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("./results/locomo_qa"),
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:7790/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="qwen3-30b-a3b")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-global-files", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
