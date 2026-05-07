#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
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

from prompts.prompts import LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT, OverlappingContextRules


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_json_fragment(text: str) -> Any:
    raw = _clean(text)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass
    left = raw.find("{")
    right = raw.rfind("}")
    if left >= 0 and right > left:
        frag = raw[left : right + 1]
        try:
            return json.loads(frag)
        except Exception:
            return {}
    return {}


def _extract_message(resp_json: Dict[str, Any]) -> str:
    try:
        return _clean(resp_json["choices"][0]["message"]["content"])
    except Exception:
        return ""


def _build_prompt(conversation: str, *, overlapping_rules: str) -> str:
    prompt = LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT
    # Compatible with both "{name}" and "{{name}}" placeholder styles.
    prompt = prompt.replace("{OverlappingContextRules}", overlapping_rules)
    prompt = prompt.replace("{{OverlappingContextRules}}", overlapping_rules)
    prompt = prompt.replace("{conversation}", conversation)
    prompt = prompt.replace("{{conversation}}", conversation)
    return prompt.strip()


def _call_chat(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
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
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _source_time_by_id_from_dialogue(dialogue_text: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for line in dialogue_text.splitlines():
        m = re.match(r"^\[([^,\]]+)\s*,\s*[^\]]+\]\s*(\d+)\.", line.strip())
        if not m:
            continue
        ts = _clean(m.group(1))
        sid = int(m.group(2))
        out[sid] = ts
    return out


def _normalize_memory_entry(row: Any, *, source_time_map: Dict[int, str]) -> Dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    source_id_raw = row.get("source_id")
    try:
        source_id = int(source_id_raw)
    except Exception:
        source_id = None
    dimension = row.get("dimension") if isinstance(row.get("dimension"), dict) else {}

    keywords: List[str] = []
    for kw in list(dimension.get("keywords") or []):
        item = _clean(kw)
        if item and item not in keywords:
            keywords.append(item)

    memory_type = _clean(dimension.get("memory_type")).lower()
    if memory_type not in {"fact", "episodic", "profile"}:
        memory_type = ""

    normalized = {
        "source_id": source_id if source_id is not None else source_id_raw,
        "source_speaker": _clean(row.get("source_speaker")),
        "source_time": source_time_map.get(source_id or -1, ""),
        "content": _clean(row.get("content")),
        "dimension": {
            "memory_type": memory_type,
            "time": _clean(dimension.get("time")),
            "location": _clean(dimension.get("location")),
            "reason": _clean(dimension.get("reason")),
            "purpose": _clean(dimension.get("purpose")),
            "keywords": keywords,
        },
    }
    if not normalized["content"]:
        return None
    return normalized


def _iter_record_dirs(compressed_root: Path) -> List[Path]:
    return [p.parent for p in sorted(compressed_root.glob("*/*/summary.json"))]


def _window_paths(record_dir: Path) -> List[Path]:
    return sorted((record_dir / "windows").glob("window_*.json"))


def _output_rel(record_dir: Path, compressed_root: Path) -> Path:
    rel = record_dir.relative_to(compressed_root)
    # Flatten output by conv name only:
    # compressed: <conv>/<record>/...
    # output:     <conv>/...
    if len(rel.parts) >= 1:
        return Path(rel.parts[0])
    return rel


def _process_record(
    *,
    record_dir: Path,
    compressed_root: Path,
    output_root: Path,
    base_url: str,
    api_key: str,
    model_name: str,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    empty_overlap_for_first_window: bool,
) -> Dict[str, Any]:
    rel = _output_rel(record_dir, compressed_root)
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
            "prompt_file": "locomo/src/prompts/prompts.py",
            "prompt_constant": "LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT",
            "started_at": datetime.now().isoformat(),
        },
    )

    rows: List[Dict[str, Any]] = []
    all_memories: List[Dict[str, Any]] = []
    for window_path in _window_paths(record_dir):
        window = _load_json(window_path)
        window_idx = int(window.get("window_index", 0))
        win_dir = out_record_dir / f"window_{window_idx:04d}"
        _ensure_dir(win_dir)

        conversation = _clean(window.get("text"))
        source_time_map = _source_time_by_id_from_dialogue(conversation)
        overlap_rules = ""
        if not (empty_overlap_for_first_window and window_idx == 0):
            overlap_rules = OverlappingContextRules
        prompt = _build_prompt(conversation, overlapping_rules=overlap_rules)

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
                    timeout=timeout,
                )
                raw_text = _extract_message(response_json)
                parsed = _safe_json_fragment(raw_text)
                mem_rows = parsed.get("memories") if isinstance(parsed, dict) else []
                if not isinstance(mem_rows, list):
                    raise ValueError("parsed_response_missing_memories_list")
                for m in mem_rows:
                    norm = _normalize_memory_entry(m, source_time_map=source_time_map)
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
            "overlapping_rules_empty": bool(window_idx == 0 and empty_overlap_for_first_window),
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
    compressed_root = args.compressed_root.resolve()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.output_root.resolve() / run_name
    _ensure_dir(out_root)

    record_dirs = _iter_record_dirs(compressed_root)
    if args.max_records > 0:
        record_dirs = record_dirs[: args.max_records]

    status_path = out_root / "status.json"
    failures_path = out_root / "failures.json"
    failures: List[Dict[str, Any]] = []
    if args.resume and failures_path.exists():
        try:
            failures = list(_load_json(failures_path).get("failures") or [])
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
            "compressed_root": str(compressed_root),
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
        rel = _output_rel(record_dir, compressed_root)
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
                "compressed_root": str(compressed_root),
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
                compressed_root=compressed_root,
                output_root=out_root,
                base_url=args.base_url,
                api_key=args.api_key,
                model_name=args.model_name,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                empty_overlap_for_first_window=args.empty_overlap_for_first_window,
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
                "compressed_root": str(compressed_root),
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
            "compressed_root": str(compressed_root),
            "output_root": str(out_root),
            "prompt_file": str(Path(__file__).resolve().parents[1] / "prompts" / "prompts.py"),
            "prompt_constant": "LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT",
            "overlapping_rules_source": "OverlappingContextRules",
            "empty_overlap_for_first_window": bool(args.empty_overlap_for_first_window),
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
            "compressed_root": str(compressed_root),
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
    parser = argparse.ArgumentParser(description="Build structured memories from compressed LoCoMo windows.")
    parser.add_argument(
        "--compressed-root",
        type=Path,
        required=True,
        help="Path like .../results/segment_results/compressed/<run_name>",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/DimMem/evaluation/dimmem/locomo/results/memory_results"
        ),
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:7790/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="qwen3-30b-a3b")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--empty-overlap-for-first-window", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    print(str(out))


if __name__ == "__main__":
    main()
