"""Assistant context lookup via per-window assistant_replies.json (uid-based).

Mapping chain:
  memory source_boundary_id -> (window_index, source_id)
  -> uid = w{window_index:04d}u{source_id:02d}
  -> assistant_replies.json[uid] -> assistant_reply
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _clean(v: Any) -> str:
    return str(v or "").strip()


def build_boundary_to_window_source(
    memory_dir: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """Build source_boundary_id -> {window_index, source_id} from all_memories.json.

    Mirrors load_records() in retrieve_from_parsed_query.py which builds
    source_boundary_id as f"{window_name}_{idx:04d}" where idx is the global
    enumerate index over all memories.

    Returns:
        (boundary_index, source_record_dir) tuple.
    """
    index: Dict[str, Dict[str, Any]] = {}
    all_memories_path = memory_dir / "all_memories.json"
    if not all_memories_path.exists():
        return index, None
    payload = json.loads(all_memories_path.read_text(encoding="utf-8"))
    memories = payload.get("memories") or []
    source_record_dir: Optional[str] = None
    for idx, mem in enumerate(memories):
        if not isinstance(mem, dict):
            continue
        window_name = _clean(mem.get("window_dir")) or _clean(mem.get("window_index"))
        boundary_id = f"{window_name}_{idx:04d}"
        window_index = mem.get("window_index", 0)
        source_id = mem.get("source_id", 0)
        if source_record_dir is None:
            source_record_dir = _clean(mem.get("source_record_dir"))
        index[boundary_id] = {
            "window_index": int(window_index),
            "source_id": int(source_id),
        }
    return index, source_record_dir


def load_window_assistant_replies(windows_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all window_XXXX_assistant_replies.json into a uid -> reply_info mapping."""
    uid_map: Dict[str, Dict[str, Any]] = {}
    for path in sorted(windows_dir.glob("window_*_assistant_replies.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        replies = data.get("replies", {})
        uid_map.update(replies)
    return uid_map


def attach_assistant_context(
    records: List[Dict[str, Any]],
    boundary_index: Dict[str, Dict[str, Any]],
    uid_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach assistant_reply to each retrieved record.

    Lookup chain: source_boundary_id -> (window_index, source_id) -> uid -> reply.
    """
    for rec in records:
        boundary_id = _clean(rec.get("source_boundary_id"))
        info = boundary_index.get(boundary_id)
        if info:
            wi = info["window_index"]
            sid = info["source_id"]
            uid = f"w{wi:04d}u{sid:02d}"
            reply_info = uid_map.get(uid, {})
            rec["assistant_reply"] = reply_info.get("assistant_reply", "")
            rec["assistant_uid"] = uid
            rec["session_id"] = reply_info.get("session_id", "")
            rec["session_local_user_index"] = reply_info.get("session_local_user_index", 0)
        else:
            rec["assistant_reply"] = ""
            rec["assistant_uid"] = ""
            rec["session_id"] = ""
            rec["session_local_user_index"] = 0
    return records
