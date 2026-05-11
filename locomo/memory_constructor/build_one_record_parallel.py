#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

THIS_FILE = Path(__file__).resolve()
LOCOMO_SRC_ROOT = THIS_FILE.parents[1]
if str(LOCOMO_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCOMO_SRC_ROOT))

from prompts.prompts import LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT, OverlappingContextRules
from models import DimensionMemory


def _clean(v: Any) -> str:
    return str(v or "").strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _build_prompt(conversation: str, overlapping_rules: str) -> str:
    prompt = LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT
    prompt = prompt.replace("{OverlappingContextRules}", overlapping_rules)
    prompt = prompt.replace("{{OverlappingContextRules}}", overlapping_rules)
    prompt = prompt.replace("{conversation}", conversation)
    prompt = prompt.replace("{{conversation}}", conversation)
    return prompt.strip()


def _load_prompt_module(prompt_file: Path) -> tuple[str, str]:
    spec = importlib.util.spec_from_file_location("locomo_dynamic_prompts", str(prompt_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load prompt file: {prompt_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prompt = _clean(getattr(mod, "LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT", ""))
    overlap = _clean(getattr(mod, "OverlappingContextRules", ""))
    if not prompt:
        raise RuntimeError(f"LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT missing in: {prompt_file}")
    return prompt, overlap


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
    session = requests.Session()
    session.trust_env = False
    resp = session.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _source_time_by_id_from_dialogue(dialogue_text: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for line in dialogue_text.splitlines():
        m = re.match(r"^\[([^,\]]+)\s*,\s*[^\]]+\]\s*(\d+)\.", line.strip())
        if not m:
            continue
        out[int(m.group(2))] = _clean(m.group(1))
    return out


def _normalize_memory_entry(row: Any, source_time_map: Dict[int, str]) -> Dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    source_id_raw = row.get("source_id")
    try:
        source_id = int(source_id_raw)
    except Exception:
        source_id = None

    normalized = {
        "source_id": source_id if source_id is not None else source_id_raw,
        "source_speaker": _clean(row.get("source_speaker")),
        "source_time": source_time_map.get(source_id or -1, ""),
        "content": _clean(row.get("content")),
        "dimension": DimensionMemory.from_dict(row.get("dimension")).to_dict(),
    }
    if not normalized["content"]:
        return None
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel memory extraction for one LoCoMo record.")
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--output-record-dir", type=Path, required=True)
    parser.add_argument("--ports", default="7790,7791,7792,7793")
    parser.add_argument(
        "--base-urls",
        default="",
        help="Comma-separated full OpenAI-compatible base urls, e.g. http://10.0.0.1:7790/v1",
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="qwen3-30b-a3b")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prompt-file", type=Path, default=None)
    args = parser.parse_args()

    prompt_template = LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT
    overlap_rules_template = OverlappingContextRules
    prompt_file_used = "locomo/src/prompts/prompts.py"
    if args.prompt_file is not None:
        prompt_template, overlap_rules_template = _load_prompt_module(args.prompt_file)
        prompt_file_used = str(args.prompt_file)

    windows = sorted((args.record_dir / "windows").glob("window_*.json"))
    args.output_record_dir.mkdir(parents=True, exist_ok=True)
    ports = [p.strip() for p in args.ports.split(",") if p.strip()]
    base_urls = [u.strip().rstrip("/") for u in str(args.base_urls or "").split(",") if u.strip()]
    lock = threading.Lock()
    done = 0
    fail = 0
    rows: List[Dict[str, Any]] = []

    _write_json(
        args.output_record_dir / "experiment_config.json",
        {
            "created_at": datetime.now().isoformat(),
            "record_dir": str(args.record_dir),
            "output_record_dir": str(args.output_record_dir),
            "ports": ports,
            "base_urls": base_urls,
            "workers": args.workers,
            "model_name": args.model_name,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "prompt_file": prompt_file_used,
            "prompt_constant": "LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT",
        },
    )

    def run_one(i: int, window_path: Path) -> Dict[str, Any]:
        nonlocal done, fail
        if base_urls:
            base_url = base_urls[i % len(base_urls)]
        else:
            base_url = f"http://127.0.0.1:{ports[i % len(ports)]}/v1"
        win = _load_json(window_path)
        window_idx = int(win.get("window_index", i))
        conversation = _clean(win.get("text"))
        source_time_map = _source_time_by_id_from_dialogue(conversation)
        overlap = "" if window_idx == 0 else overlap_rules_template
        # Use the selected prompt template while keeping existing placeholder replacement logic.
        global LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT
        _orig = LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT
        LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT = prompt_template
        try:
            prompt = _build_prompt(conversation, overlap)
        finally:
            LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT = _orig

        out_dir = args.output_record_dir / f"window_{window_idx:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "window_input.json", win)
        _write_text(out_dir / "dialogue_input.txt", conversation)
        _write_text(out_dir / "extract_prompt.txt", prompt)

        response_json: Dict[str, Any] | None = None
        raw_text = ""
        parsed: Any = None
        memories: List[Dict[str, Any]] = []
        err = None
        ok = False
        started = time.time()

        for attempt in range(1, max(1, args.max_retries) + 1):
            try:
                response_json = _call_chat(
                    base_url=base_url,
                    api_key=args.api_key,
                    model_name=args.model_name,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                raw_text = _extract_message(response_json)
                parsed = _safe_json_fragment(raw_text)
                mem_rows = parsed.get("memories") if isinstance(parsed, dict) else []
                if not isinstance(mem_rows, list):
                    raise ValueError("parsed_response_missing_memories_list")
                for m in mem_rows:
                    norm = _normalize_memory_entry(m, source_time_map)
                    if norm is not None:
                        memories.append(norm)
                ok = True
                err = None
                break
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_retries:
                    time.sleep(min(2 * attempt, 8))

        if response_json is not None:
            _write_json(out_dir / "raw_response.json", response_json)
        _write_text(out_dir / "raw_response.txt", raw_text)
        if parsed is not None:
            _write_json(out_dir / "parsed_payload.json", parsed)
        _write_json(out_dir / "normalized_memories.json", {"memories": memories})

        row = {
            "window_index": window_idx,
            "source_path": str(window_path),
            "ok": ok,
            "error": err,
            "memory_count": len(memories),
            "elapsed_seconds": time.time() - started,
            "usage": (response_json or {}).get("usage"),
            "port": base_url,
        }
        _write_json(out_dir / "result.json", row)
        with lock:
            rows.append(row)
            if ok:
                done += 1
            else:
                fail += 1
            _write_json(
                args.output_record_dir / "status.json",
                {
                    "state": "running",
                    "total_windows": len(windows),
                    "done": done,
                    "fail": fail,
                    "updated_at": datetime.now().isoformat(),
                },
            )
        return row

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(run_one, i, wp) for i, wp in enumerate(windows)]
        for _ in as_completed(futures):
            pass

    rows_sorted = sorted(rows, key=lambda x: x.get("window_index", 0))
    all_memories: List[Dict[str, Any]] = []
    for r in rows_sorted:
        mem_path = args.output_record_dir / f"window_{int(r['window_index']):04d}" / "normalized_memories.json"
        mems = (_load_json(mem_path).get("memories") or []) if mem_path.exists() else []
        for j, m in enumerate(mems):
            item = dict(m)
            item["window_index"] = int(r["window_index"])
            item["memory_index"] = j
            all_memories.append(item)

    _write_json(
        args.output_record_dir / "summary.json",
        {
            "source_record_dir": str(args.record_dir),
            "output_record_dir": str(args.output_record_dir),
            "count": len(rows_sorted),
            "ok_count": sum(1 for x in rows_sorted if x.get("ok")),
            "error_count": sum(1 for x in rows_sorted if not x.get("ok")),
            "total_memory_count": len(all_memories),
            "rows": rows_sorted,
        },
    )
    _write_json(
        args.output_record_dir / "all_memories.json",
        {
            "source_record_dir": str(args.record_dir),
            "memory_count": len(all_memories),
            "memories": all_memories,
        },
    )
    _write_json(
        args.output_record_dir / "status.json",
        {
            "state": "completed",
            "total_windows": len(windows),
            "done": done,
            "fail": fail,
            "updated_at": datetime.now().isoformat(),
        },
    )
    print(str(args.output_record_dir))


if __name__ == "__main__":
    main()
