#!/usr/bin/env python3
"""
Parallel memory extraction from compressed LoCoMo windows using vLLM OPD model.
Uses system+user message format (matching training) and disables thinking mode.
"""
from __future__ import annotations

import argparse
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

# Load the exact system prompt from training data for consistency
_OPD_TRAINING_DATA = Path("/mnt/workspace/zhiyue-L3-TerminalPerceptiveMemory/workspace/qwt/projects/roll/ROLL/examples/locomo-opd/data/locomo_opd_messages.jsonl")
if _OPD_TRAINING_DATA.exists():
    import json as _json_init
    with open(_OPD_TRAINING_DATA) as _f:
        _first = _json_init.loads(_f.readline())
    OPD_SYSTEM_PROMPT = _first["messages"][0]["content"]
else:
    # Fallback
    from prompts.prompts import LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT
    OPD_SYSTEM_PROMPT = LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT


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
    # Strip <think>...</think> if present
    if "<think>" in raw:
        idx = raw.find("</think>")
        if idx >= 0:
            raw = raw[idx + len("</think>"):].strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Try to find the outermost JSON object
    left = raw.find("{")
    right = raw.rfind("}")
    if left >= 0 and right > left:
        frag = raw[left: right + 1]
        try:
            return json.loads(frag)
        except Exception:
            pass
    # Handle truncated JSON: find all complete memory entries via regex
    # This handles cases where max_tokens cuts off the output mid-JSON
    memories = []
    pattern = r'\{\s*"source_id"\s*:.*?"keywords"\s*:\s*\[.*?\]\s*\}\s*\}'
    import re as _re
    for m in _re.finditer(pattern, raw, _re.DOTALL):
        try:
            entry = json.loads(m.group())
            memories.append(entry)
        except Exception:
            continue
    if memories:
        return {"memories": memories}
    return {}


def _extract_message(resp_json: Dict[str, Any]) -> str:
    try:
        return _clean(resp_json["choices"][0]["message"]["content"])
    except Exception:
        return ""


def _get_system_prompt() -> str:
    """Return the exact system prompt used during OPD training."""
    return OPD_SYSTEM_PROMPT


def _call_chat(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
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
    dim = row.get("dimension") if isinstance(row.get("dimension"), dict) else {}
    memory_type = _clean(dim.get("memory_type")).lower()
    if memory_type not in {"fact", "episodic", "profile"}:
        memory_type = ""
    keywords: List[str] = []
    for kw in list(dim.get("keywords") or []):
        s = _clean(kw)
        if s and s not in keywords:
            keywords.append(s)
    normalized = {
        "source_id": source_id if source_id is not None else source_id_raw,
        "source_speaker": _clean(row.get("source_speaker")),
        "source_time": source_time_map.get(source_id or -1, ""),
        "content": _clean(row.get("content")),
        "dimension": {
            "memory_type": memory_type,
            "time": _clean(dim.get("time")),
            "location": _clean(dim.get("location")),
            "reason": _clean(dim.get("reason")),
            "purpose": _clean(dim.get("purpose")),
            "keywords": keywords,
        },
    }
    if not normalized["content"]:
        return None
    return normalized


def process_all_records(args: argparse.Namespace) -> None:
    compressed_root = args.compressed_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ports = [p.strip() for p in args.ports.split(",") if p.strip()]
    base_urls = [f"http://127.0.0.1:{p}/v1" for p in ports]

    # Find all record dirs
    record_dirs = sorted([p.parent for p in compressed_root.glob("*/*/summary.json")])
    print(f"Found {len(record_dirs)} records, {len(base_urls)} vLLM endpoints")

    # Collect all (record_dir, window_path) pairs
    tasks: List[tuple] = []
    for record_dir in record_dirs:
        conv_name = record_dir.parent.name  # e.g. Audrey-conv44
        windows = sorted((record_dir / "windows").glob("window_*.json"))
        for wp in windows:
            tasks.append((record_dir, conv_name, wp))

    print(f"Total windows to process: {len(tasks)}")

    lock = threading.Lock()
    done = [0]
    fail = [0]
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    def run_one(idx: int, record_dir: Path, conv_name: str, window_path: Path) -> None:
        base_url = base_urls[idx % len(base_urls)]
        win = _load_json(window_path)
        window_idx = int(win.get("window_index", 0))
        conversation = _clean(win.get("text"))
        source_time_map = _source_time_by_id_from_dialogue(conversation)

        system_prompt = _get_system_prompt()

        out_dir = output_root / conv_name / f"window_{window_idx:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "window_input.json", win)
        _write_text(out_dir / "dialogue_input.txt", conversation)

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
                    system_prompt=system_prompt,
                    user_content=conversation,
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
            "conv_name": conv_name,
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
            if ok:
                done[0] += 1
            else:
                fail[0] += 1
            all_results.setdefault(conv_name, []).append(row)
            if (done[0] + fail[0]) % 10 == 0:
                print(f"  Progress: {done[0]+fail[0]}/{len(tasks)} (done={done[0]}, fail={fail[0]})")

    print(f"Starting with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(run_one, i, record_dir, conv_name, wp)
            for i, (record_dir, conv_name, wp) in enumerate(tasks)
        ]
        for f in as_completed(futures):
            exc = f.exception()
            if exc:
                print(f"  FATAL: {exc}")

    # Write per-conv summaries
    for conv_name, rows in all_results.items():
        rows_sorted = sorted(rows, key=lambda x: x.get("window_index", 0))
        conv_dir = output_root / conv_name

        # Merge all memories
        all_memories: List[Dict[str, Any]] = []
        for r in rows_sorted:
            mem_path = conv_dir / f"window_{int(r['window_index']):04d}" / "normalized_memories.json"
            mems = (_load_json(mem_path).get("memories") or []) if mem_path.exists() else []
            for j, m in enumerate(mems):
                item = dict(m)
                item["window_index"] = int(r["window_index"])
                item["memory_index"] = j
                all_memories.append(item)

        _write_json(conv_dir / "summary.json", {
            "conv_name": conv_name,
            "count": len(rows_sorted),
            "ok_count": sum(1 for x in rows_sorted if x.get("ok")),
            "error_count": sum(1 for x in rows_sorted if not x.get("ok")),
            "total_memory_count": len(all_memories),
            "rows": rows_sorted,
        })
        _write_json(conv_dir / "all_memories.json", {
            "conv_name": conv_name,
            "memory_count": len(all_memories),
            "memories": all_memories,
        })

    # Write global summary
    _write_json(output_root / "experiment_config.json", {
        "created_at": datetime.now().isoformat(),
        "compressed_root": str(compressed_root),
        "output_root": str(output_root),
        "model_name": args.model_name,
        "ports": ports,
        "workers": args.workers,
        "max_tokens": args.max_tokens,
        "total_records": len(record_dirs),
        "total_windows": len(tasks),
        "done": done[0],
        "failed": fail[0],
    })

    print(f"\n=== DONE ===")
    print(f"Total: {len(tasks)} windows, done={done[0]}, fail={fail[0]}")
    print(f"Output: {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build memories from compressed windows using OPD model")
    parser.add_argument("--compressed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ports", default="7790,7791,7792,7793,7794,7795,7796,7797")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    process_all_records(args)


if __name__ == "__main__":
    main()
