#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", _clean(value))
    text = text.strip("_")
    return text or "unknown"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_session_time(raw: str) -> datetime:
    # Example: "2023/06/25 (Sun) 13:22"
    return datetime.strptime(_clean(raw), "%Y/%m/%d (%a) %H:%M")


def _iso_compact(ts: datetime) -> str:
    if ts.microsecond:
        return ts.isoformat(timespec="microseconds")
    return ts.isoformat(timespec="seconds")


def _truncate_user_content(
    text: str,
    *,
    threshold: int,
    head: int,
    middle: int,
    tail: int,
) -> str:
    if len(text) <= threshold:
        return text
    if head + middle + tail <= 0:
        return text
    start = text[: max(0, head)]
    mid_start = max(0, (len(text) - middle) // 2)
    mid = text[mid_start : mid_start + max(0, middle)] if middle > 0 else ""
    end = text[-max(0, tail) :] if tail > 0 else ""
    parts = [p for p in [start, mid, end] if p]
    return "\n...\n".join(parts)


@dataclass
class UserMsg:
    global_user_index: int
    session_id: str
    session_local_user_index: int
    timestamp: str
    weekday: str
    content: str


def _collect_user_messages(
    item: Dict[str, Any],
    *,
    use_user_only: bool,
    truncate_threshold: int,
    truncate_head: int,
    truncate_middle: int,
    truncate_tail: int,
) -> List[UserMsg]:
    sessions = list(item.get("haystack_sessions") or [])
    dates = list(item.get("haystack_dates") or [])
    session_ids = list(item.get("haystack_session_ids") or [])

    out: List[UserMsg] = []
    global_idx = 0
    for i, sess in enumerate(sessions):
        if i >= len(dates):
            continue
        base_ts = _parse_session_time(dates[i])
        sid = _clean(session_ids[i]) if i < len(session_ids) else f"session_{i:04d}"
        local_idx = 0
        for turn_idx, row in enumerate(list(sess or [])):
            role = _clean(row.get("role", row.get("speaker", ""))).lower()
            if use_user_only and role != "user":
                continue
            if not use_user_only and role not in {"user", "assistant"}:
                continue
            local_idx += 1
            global_idx += 1
            ts = base_ts + timedelta(seconds=0.5 * (local_idx - 1))
            content = _clean(row.get("content"))
            content = _truncate_user_content(
                content,
                threshold=truncate_threshold,
                head=truncate_head,
                middle=truncate_middle,
                tail=truncate_tail,
            )
            out.append(
                UserMsg(
                    global_user_index=global_idx,
                    session_id=sid,
                    session_local_user_index=local_idx,
                    timestamp=_iso_compact(ts),
                    weekday=ts.strftime("%a"),
                    content=content,
                )
            )
    return out


def _window_text(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, m in enumerate(messages, start=1):
        lines.append(f"[{m['timestamp']}, {m['weekday']}] {idx}.User: {m['content']}")
    return "\n".join(lines)


def _get_assistant_reply(session_msgs: List[Dict[str, str]], user_index: int) -> str:
    """Get assistant reply for the N-th user message (1-based) in a session."""
    user_count = 0
    for i, msg in enumerate(session_msgs):
        if msg.get("role") == "user":
            user_count += 1
            if user_count == user_index:
                if i + 1 < len(session_msgs) and session_msgs[i + 1].get("role") == "assistant":
                    return _clean(session_msgs[i + 1].get("content"))
                return ""
    return ""


def _build_window_assistant_replies(
    window: Dict[str, Any],
    session_map: Dict[str, List[Dict[str, str]]],
) -> Dict[str, Dict[str, Any]]:
    """Build uid -> assistant reply mapping for one window.

    uid format: w{window_index:04d}u{local_source_id:02d}
    local_source_id is 1-based, matching dialogue text numbering (1.User, 2.User, ...).
    """
    window_index = int(window.get("window_index", 0))
    replies: Dict[str, Dict[str, Any]] = {}
    messages = window.get("messages", [])
    for local_idx_0, msg in enumerate(messages):
        local_source_id = local_idx_0 + 1
        uid = f"w{window_index:04d}u{local_source_id:02d}"
        gidx = msg.get("global_user_index")
        session_id = _clean(msg.get("session_id"))
        session_local_idx = int(msg.get("session_local_user_index", 0))
        session_msgs = session_map.get(session_id, [])
        assistant_reply = _get_assistant_reply(session_msgs, session_local_idx) if session_msgs and session_local_idx > 0 else ""
        replies[uid] = {
            "uid": uid,
            "source_id": local_source_id,
            "global_user_index": int(gidx) if gidx is not None else None,
            "session_id": session_id,
            "session_local_user_index": session_local_idx,
            "assistant_reply": assistant_reply,
        }
    return replies


def _build_session_map(item: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """Build session_id -> [messages] lookup from item's haystack data."""
    session_map: Dict[str, List[Dict[str, str]]] = {}
    sessions = list(item.get("haystack_sessions") or [])
    session_ids = list(item.get("haystack_session_ids") or [])
    for sid, msgs in zip(session_ids, sessions):
        session_map[str(sid)] = msgs
    return session_map


def _build_windows(messages: List[UserMsg], *, window_size: int, overlap: int) -> List[Dict[str, Any]]:
    if not messages:
        return []
    step = max(1, window_size - overlap)
    rows = [m.__dict__ for m in messages]
    windows: List[Dict[str, Any]] = []
    for start in range(0, len(rows), step):
        chunk = rows[start : start + window_size]
        if not chunk:
            break
        if len(chunk) < window_size and start != 0:
            break
        overlap_count = 0 if len(windows) == 0 else min(overlap, len(chunk))
        window = {
            "window_index": len(windows),
            "message_count": len(chunk),
            "overlap_count": overlap_count,
            "extract_start_message_index": overlap_count + 1,
            "start_global_user_index": chunk[0]["global_user_index"],
            "end_global_user_index": chunk[-1]["global_user_index"],
            "start_timestamp": chunk[0]["timestamp"],
            "end_timestamp": chunk[-1]["timestamp"],
            "start_session_id": chunk[0]["session_id"],
            "end_session_id": chunk[-1]["session_id"],
            "text": _window_text(chunk),
            "messages": chunk,
        }
        windows.append(window)
        if start + window_size >= len(rows):
            break
    return windows


def _resolve_overlap(*, window_size: int, overlap: int, overlap_ratio: float) -> int:
    if overlap >= 0:
        resolved = overlap
    else:
        resolved = int(round(float(window_size) * float(overlap_ratio)))
    resolved = max(0, resolved)
    resolved = min(resolved, max(0, window_size - 1))
    return resolved


def _question_type_from_path(path: Path) -> str:
    # e.g. longmemeval_s_cleaned__knowledge-update.json
    stem = path.stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return ""


def _iter_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("longmemeval_s_cleaned__*.json"))


def run(args: argparse.Namespace) -> Path:
    ts = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / ts
    _ensure_dir(run_root)

    files = _iter_input_files(args.input_path)
    resolved_overlap = _resolve_overlap(
        window_size=int(args.window_size),
        overlap=int(args.overlap),
        overlap_ratio=float(args.overlap_ratio),
    )

    manifest = {
        "created_at": datetime.now().isoformat(),
        "input_path": str(args.input_path),
        "output_root": str(run_root),
        "window_size": args.window_size,
        "overlap": resolved_overlap,
        "window_step": max(1, args.window_size - resolved_overlap),
        "overlap_ratio": float(args.overlap_ratio),
        "use_user_only": bool(args.use_user_only),
        "truncate_threshold": args.truncate_threshold,
        "truncate_head": args.truncate_head,
        "truncate_middle": args.truncate_middle,
        "truncate_tail": args.truncate_tail,
        "input_files": [str(x) for x in files],
        "record_count": 0,
    }

    total_records = 0
    for file_path in files:
        file_qtype = _question_type_from_path(file_path)
        data = list(_load_json(file_path) or [])
        if args.max_items > 0:
            data = data[: args.max_items]
        for row_idx, item in enumerate(data):
            qtype = file_qtype or _clean(item.get("question_type")) or "unknown"
            qid = _slugify(_clean(item.get("question_id", f"row_{row_idx}")))
            sample_id = f"{row_idx:04d}_{qid}"
            rec_dir = run_root / qtype / sample_id
            windows_dir = rec_dir / "windows"
            _ensure_dir(windows_dir)

            messages = _collect_user_messages(
                item,
                use_user_only=bool(args.use_user_only),
                truncate_threshold=args.truncate_threshold,
                truncate_head=args.truncate_head,
                truncate_middle=args.truncate_middle,
                truncate_tail=args.truncate_tail,
            )
            windows = _build_windows(messages, window_size=args.window_size, overlap=resolved_overlap)

            _write_json(rec_dir / "input_item.json", item)
            all_user_text = _window_text([m.__dict__ for m in messages]) if messages else ""
            (rec_dir / "all_user_messages.txt").write_text(all_user_text, encoding="utf-8")

            for win in windows:
                stem = f"window_{int(win['window_index']):04d}"
                _write_json(windows_dir / f"{stem}.json", win)
                (windows_dir / f"{stem}.txt").write_text(win["text"], encoding="utf-8")

            # Generate assistant_replies.json per window
            session_map = _build_session_map(item)
            for win in windows:
                replies = _build_window_assistant_replies(win, session_map)
                stem = f"window_{int(win['window_index']):04d}"
                _write_json(windows_dir / f"{stem}_assistant_replies.json", {
                    "window_index": win.get("window_index"),
                    "source_count": len(replies),
                    "replies": replies,
                })

            summary = {
                "source_file": str(file_path),
                "row_index": row_idx,
                "question_id": item.get("question_id"),
                "question_type": qtype,
                "question": item.get("question"),
                "question_date": item.get("question_date"),
                "output_dir": str(rec_dir),
                "window_size": args.window_size,
                "overlap": resolved_overlap,
                "window_step": max(1, args.window_size - resolved_overlap),
                "overlap_ratio": float(args.overlap_ratio),
                "user_message_count": len(messages),
                "window_count": len(windows),
                "windows": windows,
            }
            _write_json(rec_dir / "summary.json", summary)
            total_records += 1

    manifest["record_count"] = total_records
    _write_json(run_root / "run_manifest.json", manifest)
    return run_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build LongMemEval raw segment windows.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/longmemeval_s_cleaned.json"),
        help="Input json file (single file with question_type field) or directory containing longmemeval_s_cleaned__*.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./results/segments/raw"),
        help="Output root for run timestamp folder",
    )
    parser.add_argument("--run-name", default="", help="Optional custom run folder name")
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--overlap", type=int, default=-1, help="Absolute overlap count. -1 means auto from --overlap-ratio.")
    parser.add_argument("--overlap-ratio", type=float, default=0.2, help="Used only when --overlap is -1.")
    parser.add_argument("--max-items", type=int, default=0, help="0 means all")
    parser.add_argument("--use-user-only", action="store_true", default=True)
    parser.add_argument("--truncate-threshold", type=int, default=8000)
    parser.add_argument("--truncate-head", type=int, default=500)
    parser.add_argument("--truncate-middle", type=int, default=200)
    parser.add_argument("--truncate-tail", type=int, default=300)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output = run(args)
    print(str(output))


if __name__ == "__main__":
    main()
