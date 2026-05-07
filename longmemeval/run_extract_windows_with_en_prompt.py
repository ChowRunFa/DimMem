#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests


THIS_FILE = Path(__file__).resolve()
LONGMEM_ROOT = THIS_FILE.parent
PROMPT_FILE = Path(
    os.environ.get(
        "LONGMEM_PROMPT_FILE",
        str(LONGMEM_ROOT / "prompts/prompts.py"),
    )
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _extract_prompt_constant(name: str) -> str:
    text = PROMPT_FILE.read_text(encoding="utf-8")
    pattern = rf"{name}\s*=\s*(?:\"\"\"(.*?)\"\"\"|'''(.*?)''')"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"unable to locate prompt constant: {name}")
    return (match.group(1) or match.group(2) or "").strip()


EXTRACTION_PROMPT_TEMPLATE = _extract_prompt_constant("LONGMEMEVAL_STRUCTURED_MEMORY_EXTRACTION_PROMPT")
try:
    OVERLAP_RULE_TEXT = _extract_prompt_constant("OVERLAP_RULE")
except Exception:
    OVERLAP_RULE_TEXT = ""


def _build_prompt(
    conversation: str,
    window_index: int | None = None,
    overlap_count: int = 0,
) -> str:
    prompt = EXTRACTION_PROMPT_TEMPLATE
    # First window in a dialogue should not receive overlap rule.
    if "{overlap_rule}" in prompt:
        use_overlap_rule = (window_index is None or int(window_index) > 0) and int(overlap_count) > 0
        overlap_text = ""
        if use_overlap_rule:
            overlap_text = OVERLAP_RULE_TEXT
            overlap_text = overlap_text.replace("{overlap_count}", str(int(overlap_count)))
            overlap_text = overlap_text.replace("{extract_start_index}", str(int(overlap_count) + 1))
        prompt = prompt.replace("{overlap_rule}", overlap_text)
    return prompt.replace("{conversation}", _clean(conversation))


def _safe_json_fragment(text: str) -> Any:
    payload = _clean(text)
    if not payload:
        raise ValueError("empty response")
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload, flags=re.IGNORECASE)
        payload = re.sub(r"\s*```$", "", payload).strip()
    try:
        return json.loads(payload)
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", payload):
        try:
            parsed, _ = decoder.raw_decode(payload[match.start() :])
            return parsed
        except Exception:
            continue
    raise ValueError("unable to parse JSON")


def _call_chat(*, base_url: str, api_key: str, model_name: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
    }
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def _extract_text(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    return _clean((choices[0].get("message") or {}).get("content"))


def _window_paths(windows_dir: Path) -> List[Path]:
    return sorted(windows_dir.glob("window_*.json"))


def _load_window(window_path: Path) -> Dict[str, Any]:
    return json.loads(window_path.read_text(encoding="utf-8"))


def _source_time_by_id_from_dialogue(conversation: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for line in _clean(conversation).splitlines():
        m = re.match(r"^\[(?P<ts>[^,\]]+)(?:,[^\]]*)?\]\s*(?P<sid>\d+)\.User:", line.strip())
        if not m:
            continue
        source_id = int(m.group("sid"))
        ts = _clean(m.group("ts"))
        if source_id > 0 and ts:
            mapping[source_id] = ts
    return mapping


def _normalize_keywords(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    result: List[str] = []
    for row in values:
        text = _clean(row)
        if not text:
            continue
        if text not in result:
            result.append(text)
    return result


def _normalize_memory_type(value: Any) -> str:
    v = _clean(value).lower()
    if v in {"fact", "episodic", "profile"}:
        return v
    return ""


def _to_int(value: Any) -> int | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        return None


def _normalize_memory_entry(row: Any, *, source_time_map: Dict[int, str], session_map: Dict[int, Dict[str, Any]] | None = None) -> Dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    source_id = _to_int(row.get("source_id"))
    content = _clean(row.get("content"))
    dimension = row.get("dimension") if isinstance(row.get("dimension"), dict) else {}

    if source_id is None or not content:
        return None

    session_info = (session_map or {}).get(source_id, {})

    normalized = {
        "source_id": source_id,
        "source_time": _clean(source_time_map.get(source_id)),
        "session_id": _clean(session_info.get("session_id")),
        "session_local_user_index": int(session_info.get("session_local_user_index", 0)),
        "content": content,
        "dimension": {
            "memory_type": _normalize_memory_type(dimension.get("memory_type")),
            "time": _clean(dimension.get("time")),
            "location": _clean(dimension.get("location")),
            "reason": _clean(dimension.get("reason")),
            "purpose": _clean(dimension.get("purpose")),
            "keywords": _normalize_keywords(dimension.get("keywords")),
        },
    }
    return normalized


def _build_session_map_from_window(window: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Build source_id -> {session_id, session_local_user_index} from window messages."""
    session_map: Dict[int, Dict[str, Any]] = {}
    for msg in window.get("messages", []):
        gidx = msg.get("global_user_index")
        if gidx is not None:
            session_map[int(gidx)] = {
                "session_id": _clean(msg.get("session_id")),
                "session_local_user_index": int(msg.get("session_local_user_index", 0)),
            }
    return session_map


__all__: Iterable[str] = [
    "PROMPT_FILE",
    "_build_prompt",
    "_build_session_map_from_window",
    "_call_chat",
    "_clean",
    "_extract_text",
    "_load_window",
    "_normalize_memory_entry",
    "_safe_json_fragment",
    "_source_time_by_id_from_dialogue",
    "_window_paths",
]
