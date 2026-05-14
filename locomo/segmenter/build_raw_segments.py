#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", _clean(value)).strip("_")
    return text or "unknown"


def _parse_locomo_session_time(raw: str) -> datetime:
    # Example: "8:56 pm on 20 July, 2023"
    normalized = _clean(raw).replace("am", "AM").replace("pm", "PM")
    return datetime.strptime(normalized, "%I:%M %p on %d %B, %Y")


def _extract_question_type(file_path: Path, is_single_file_input: bool = False) -> str:
    # When the input is a single flat JSON file, no question type grouping
    if is_single_file_input:
        return ""
    # /.../locomo10_by_type/1Multi-hop/locomo10.json -> 1Multi-hop
    return file_path.parent.name


def _truncate_text(text: str, *, threshold: int, head: int, middle: int, tail: int) -> str:
    if len(text) <= threshold:
        return text
    start = text[: max(0, head)]
    mid_start = max(0, (len(text) - middle) // 2)
    mid = text[mid_start : mid_start + max(0, middle)] if middle > 0 else ""
    end = text[-max(0, tail) :] if tail > 0 else ""
    parts = [p for p in [start, mid, end] if p]
    return "\n...\n".join(parts)


@dataclass
class Turn:
    global_turn_index: int
    session_id: str
    session_local_turn_index: int
    timestamp: str
    weekday: str
    speaker: str
    text: str


def _iter_session_ids(conversation: Dict[str, Any]) -> List[int]:
    out: List[int] = []
    for key in conversation.keys():
        m = re.fullmatch(r"session_(\d+)$", key)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _collect_turns(
    conversation: Dict[str, Any],
    *,
    threshold: int,
    head: int,
    middle: int,
    tail: int,
) -> List[Turn]:
    turns: List[Turn] = []
    global_idx = 0
    for sid_num in _iter_session_ids(conversation):
        date_key = f"session_{sid_num}_date_time"
        sess_key = f"session_{sid_num}"
        if sess_key not in conversation:
            continue
        base_time_raw = _clean(conversation.get(date_key))
        if not base_time_raw:
            continue
        base_dt = _parse_locomo_session_time(base_time_raw)
        session_turns = list(conversation.get(sess_key) or [])
        for i, turn in enumerate(session_turns, start=1):
            global_idx += 1
            ts = base_dt + timedelta(seconds=0.5 * (i - 1))
            speaker = _clean(turn.get("speaker"))
            text = _truncate_text(
                _clean(turn.get("text")),
                threshold=threshold,
                head=head,
                middle=middle,
                tail=tail,
            )
            turns.append(
                Turn(
                    global_turn_index=global_idx,
                    session_id=sess_key,
                    session_local_turn_index=i,
                    timestamp=ts.isoformat(timespec="microseconds" if ts.microsecond else "seconds"),
                    weekday=ts.strftime("%a"),
                    speaker=speaker,
                    text=text,
                )
            )
    return turns


def _window_text(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, m in enumerate(messages, start=1):
        lines.append(f"[{m['timestamp']}, {m['weekday']}] {idx}.{m['speaker']}: {m['text']}")
    return "\n".join(lines)


def _build_windows(turns: List[Turn], *, window_size: int, overlap: int) -> List[Dict[str, Any]]:
    rows = [t.__dict__ for t in turns]
    if not rows:
        return []
    step = max(1, window_size - overlap)
    windows: List[Dict[str, Any]] = []
    for start in range(0, len(rows), step):
        chunk = rows[start : start + window_size]
        if not chunk:
            break
        # Keep the last window even if it's shorter than window_size
        pass
        window_idx = len(windows)
        overlap_count = overlap if window_idx > 0 else 0
        extract_start_index = overlap_count + 1
        windows.append(
            {
                "window_index": window_idx,
                "message_count": len(chunk),
                "overlap_count": overlap_count,
                "extract_start_message_index": extract_start_index,
                "start_global_turn_index": chunk[0]["global_turn_index"],
                "end_global_turn_index": chunk[-1]["global_turn_index"],
                "start_timestamp": chunk[0]["timestamp"],
                "end_timestamp": chunk[-1]["timestamp"],
                "start_session_id": chunk[0]["session_id"],
                "end_session_id": chunk[-1]["session_id"],
                "text": _window_text(chunk),
                "messages": chunk,
            }
        )
        if start + window_size >= len(rows):
            break
    return windows


def _iter_input_files(input_root: Path) -> List[Path]:
    if input_root.is_file():
        return [input_root]
    return sorted(input_root.glob("*/*.json"))


def run(args: argparse.Namespace) -> Path:
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.output_root / run_name
    _ensure_dir(out_root)

    files = _iter_input_files(args.input_root)
    is_single_file = args.input_root.is_file()
    total_samples = 0
    for file_path in files:
        file_qtype = _extract_question_type(file_path, is_single_file_input=is_single_file)
        data = list(_load_json(file_path) or [])
        if args.max_items > 0:
            data = data[: args.max_items]
        for idx, item in enumerate(data):
            # Group key: use question type from directory, or sample_id for flat file
            group_key = file_qtype or _slugify(_clean(item.get("sample_id", f"conv_{idx}")))
            sample_slug = _slugify(_clean(item.get("sample_id", f"row_{idx}")))
            rec_id = f"{idx:04d}_{sample_slug}"
            rec_dir = out_root / group_key / rec_id if file_qtype else out_root / group_key
            win_dir = rec_dir / "windows"
            _ensure_dir(win_dir)

            conv = item.get("conversation") if isinstance(item.get("conversation"), dict) else {}
            turns = _collect_turns(
                conv,
                threshold=args.truncate_threshold,
                head=args.truncate_head,
                middle=args.truncate_middle,
                tail=args.truncate_tail,
            )
            windows = _build_windows(turns, window_size=args.window_size, overlap=args.overlap)

            _write_json(rec_dir / "input_item.json", item)
            (rec_dir / "all_turns.txt").write_text(_window_text([t.__dict__ for t in turns]), encoding="utf-8")
            for w in windows:
                stem = f"window_{int(w['window_index']):04d}"
                _write_json(win_dir / f"{stem}.json", w)
                (win_dir / f"{stem}.txt").write_text(w["text"], encoding="utf-8")

            summary = {
                "source_file": str(file_path),
                "row_index": idx,
                "sample_id": item.get("sample_id"),
                "question_type": group_key,
                "output_dir": str(rec_dir),
                "window_size": args.window_size,
                "overlap": args.overlap,
                "window_step": max(1, args.window_size - args.overlap),
                "turn_count": len(turns),
                "window_count": len(windows),
                "windows": windows,
            }
            _write_json(rec_dir / "summary.json", summary)
            total_samples += 1

    manifest = {
        "created_at": datetime.now().isoformat(),
        "input_root": str(args.input_root),
        "output_root": str(out_root),
        "window_size": args.window_size,
        "overlap": args.overlap,
        "window_step": max(1, args.window_size - args.overlap),
        "truncate_threshold": args.truncate_threshold,
        "truncate_head": args.truncate_head,
        "truncate_middle": args.truncate_middle,
        "truncate_tail": args.truncate_tail,
        "input_files": [str(x) for x in files],
        "sample_count": total_samples,
    }
    _write_json(out_root / "run_manifest.json", manifest)
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build LoCoMo raw segment windows.")
    parser.add_argument("--input-root", type=Path, default=Path("data/locomo10.json"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./results/locomo_segments/raw"),
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=0, help="0 means all")
    parser.add_argument("--truncate-threshold", type=int, default=8000)
    parser.add_argument("--truncate-head", type=int, default=500)
    parser.add_argument("--truncate-middle", type=int, default=200)
    parser.add_argument("--truncate-tail", type=int, default=300)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    print(str(out))


if __name__ == "__main__":
    main()
