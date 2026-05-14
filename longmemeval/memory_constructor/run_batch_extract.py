#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

THIS_FILE = Path(__file__).resolve()
LONGMEMEVAL_DIR = THIS_FILE.parents[1]
if str(LONGMEMEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LONGMEMEVAL_DIR))

from memory_constructor.extract_helpers import (
    _build_prompt,
    _build_session_map_from_window,
    _call_chat,
    _clean,
    _extract_text,
    _load_window,
    _normalize_memory_entry,
    _safe_json_fragment,
    _source_time_by_id_from_dialogue,
    _window_paths,
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _iter_record_dirs(segments_root: Path) -> List[Path]:
    # Support both one-level and two-level directory structures
    two_level = [p.parent for p in sorted(segments_root.glob("*/*/summary.json"))]
    one_level = [p.parent for p in sorted(segments_root.glob("*/summary.json"))]
    # Deduplicate and sort
    all_dirs = sorted(set(two_level + one_level))
    return all_dirs


def _output_rel(record_dir: Path, segments_root: Path) -> Path:
    rel = record_dir.relative_to(segments_root)
    # Keep the full relative path structure
    return rel


def _process_record(
    *,
    record_dir: Path,
    segments_root: Path,
    output_root: Path,
    base_url: str,
    api_key: str,
    model_name: str,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    overlap: int,
) -> Dict[str, Any]:
    rel = _output_rel(record_dir, segments_root)
    out_record_dir = output_root / rel
    _ensure_dir(out_record_dir)

    _write_json(
        out_record_dir / "experiment_config.json",
        {
            "source_record_dir": str(record_dir),
            "output_record_dir": str(out_record_dir),
            "base_url": base_url,
            "model_name": model_name,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
            "started_at": datetime.now().isoformat(),
        },
    )

    rows: List[Dict[str, Any]] = []
    all_memories: List[Dict[str, Any]] = []
    windows_dir = record_dir / "windows"

    for window_path in _window_paths(windows_dir):
        window = _load_window(window_path)
        window_idx = int(window.get("window_index", 0))
        overlap_count = overlap if window_idx > 0 else 0
        win_dir = out_record_dir / f"window_{window_idx:04d}"
        _ensure_dir(win_dir)

        conversation = _clean(window.get("text"))
        source_time_map = _source_time_by_id_from_dialogue(conversation)
        session_map = _build_session_map_from_window(window)
        prompt = _build_prompt(conversation, window_index=window_idx, overlap_count=overlap_count)

        _write_json(win_dir / "window_input.json", window)
        _write_text(win_dir / "dialogue_input.txt", conversation)
        _write_text(win_dir / "extract_prompt.txt", prompt)

        ok = False
        err = None
        parsed: Any = None
        raw_text = ""
        response_json: Dict[str, Any] | None = None
        memories: List[Dict[str, Any]] = []
        started = time.time()
        attempt = 0
        while attempt < max(1, max_retries):
            attempt += 1
            try:
                response_json = _call_chat(
                    base_url=base_url,
                    api_key=api_key,
                    model_name=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                raw_text = _extract_text(response_json)
                parsed = _safe_json_fragment(raw_text)
                mem_rows = parsed.get("memories") if isinstance(parsed, dict) else []
                if not isinstance(mem_rows, list):
                    raise ValueError("parsed_response_missing_memories_list")
                for m in mem_rows:
                    norm = _normalize_memory_entry(m, source_time_map=source_time_map, session_map=session_map)
                    if norm is not None:
                        memories.append(norm)
                ok = True
                break
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                if attempt >= max_retries:
                    break
                time.sleep(min(2 * attempt, 8))

        if response_json is not None:
            _write_json(win_dir / "raw_response.json", response_json)
        _write_text(win_dir / "raw_response.txt", raw_text)
        if parsed is not None:
            _write_json(win_dir / "parsed_payload.json", parsed)
        _write_json(win_dir / "normalized_memories.json", {"memories": memories})

        result = {
            "window_index": window_idx,
            "source_path": str(window_path),
            "ok": ok,
            "error": err,
            "attempt_count": attempt,
            "memory_count": len(memories),
            "elapsed_seconds": time.time() - started,
            "overlap_count": overlap_count,
            "usage": (response_json or {}).get("usage"),
        }
        _write_json(win_dir / "result.json", result)
        rows.append(result)

        for i, m in enumerate(memories):
            item = dict(m)
            item["window_index"] = window_idx
            item["memory_index"] = i
            all_memories.append(item)

    summary = {
        "source_record_dir": str(record_dir),
        "output_record_dir": str(out_record_dir),
        "count": len(rows),
        "ok_count": sum(1 for x in rows if x["ok"]),
        "error_count": sum(1 for x in rows if not x["ok"]),
        "total_memory_count": sum(int(x["memory_count"]) for x in rows),
        "rows": rows,
    }
    _write_json(out_record_dir / "summary.json", summary)
    _write_json(
        out_record_dir / "all_memories.json",
        {
            "source_record_dir": str(record_dir),
            "memory_count": len(all_memories),
            "memories": all_memories,
        },
    )
    return summary


def run(args: argparse.Namespace) -> Path:
    segments_root = args.segments_root.resolve()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.output_root.resolve() / run_name
    _ensure_dir(out_root)

    record_dirs = _iter_record_dirs(segments_root)
    if args.max_records > 0:
        record_dirs = record_dirs[: args.max_records]

    status_path = out_root / "status.json"
    failures_path = out_root / "failures.json"
    failures: List[Dict[str, Any]] = []
    if args.resume and failures_path.exists():
        try:
            failures = list(json.loads(failures_path.read_text(encoding="utf-8")).get("failures") or [])
        except Exception:
            failures = []

    done = 0
    failed = 0
    skipped = 0
    started_at = datetime.now().isoformat()
    _write_json(
        status_path,
        {
            "state": "running",
            "started_at": started_at,
            "updated_at": datetime.now().isoformat(),
            "segments_root": str(segments_root),
            "output_root": str(out_root),
            "records_total": len(record_dirs),
            "done": done,
            "failed": failed,
            "skipped_existing": skipped,
            "inflight_record": None,
            "resume": bool(args.resume),
        },
    )

    for record_dir in record_dirs:
        rel = _output_rel(record_dir, segments_root)
        summary_path = out_root / rel / "summary.json"
        if args.resume and summary_path.exists():
            skipped += 1
            done += 1
            continue

        _write_json(
            status_path,
            {
                "state": "running",
                "started_at": started_at,
                "updated_at": datetime.now().isoformat(),
                "segments_root": str(segments_root),
                "output_root": str(out_root),
                "records_total": len(record_dirs),
                "done": done,
                "failed": failed,
                "skipped_existing": skipped,
                "inflight_record": str(record_dir),
                "resume": bool(args.resume),
            },
        )
        try:
            _process_record(
                record_dir=record_dir,
                segments_root=segments_root,
                output_root=out_root,
                base_url=args.base_url,
                api_key=args.api_key,
                model_name=args.model_name,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                overlap=args.overlap,
            )
            done += 1
        except Exception as exc:
            failed += 1
            failures.append({"record_dir": str(record_dir), "error": f"{type(exc).__name__}: {exc}"})
            _write_json(failures_path, {"failures": failures})

        _write_json(
            status_path,
            {
                "state": "running",
                "started_at": started_at,
                "updated_at": datetime.now().isoformat(),
                "segments_root": str(segments_root),
                "output_root": str(out_root),
                "records_total": len(record_dirs),
                "done": done,
                "failed": failed,
                "skipped_existing": skipped,
                "inflight_record": None,
                "resume": bool(args.resume),
            },
        )

    _write_json(
        out_root / "experiment_config.json",
        {
            "created_at": datetime.now().isoformat(),
            "segments_root": str(segments_root),
            "output_root": str(out_root),
            "base_url": args.base_url,
            "model_name": args.model_name,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "records_total": len(record_dirs),
            "records_done": done,
            "records_failed": failed,
            "records_skipped_existing": skipped,
            "resume": bool(args.resume),
        },
    )
    if failures:
        _write_json(failures_path, {"failures": failures})
    _write_json(
        status_path,
        {
            "state": "completed",
            "started_at": started_at,
            "updated_at": datetime.now().isoformat(),
            "segments_root": str(segments_root),
            "output_root": str(out_root),
            "records_total": len(record_dirs),
            "done": done,
            "failed": failed,
            "skipped_existing": skipped,
            "inflight_record": None,
            "resume": bool(args.resume),
        },
    )
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build structured memories from LongMemEval segments.")
    parser.add_argument(
        "--segments-root",
        type=Path,
        required=True,
        help="Path to segments directory (e.g., results/segments/raw/<run_name>)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./results/memories"),
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--overlap", type=int, default=5, help="Number of overlapping messages between windows")
    parser.add_argument("--base-url", default="http://127.0.0.1:7790/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="gpt-4.1-mini")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    print(str(out))


if __name__ == "__main__":
    main()
