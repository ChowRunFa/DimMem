#!/usr/bin/env python3
"""LLM-based memory consolidation decisions (V2).

For each candidate pair, calls gpt-4.1-mini to decide: MERGE / SUPERSEDE / KEEP_BOTH.
Then applies decisions to produce consolidated memory banks.

V2 changes:
  - Time-aware prompts: "Memory A (newer) ... Memory B (older)"
  - SUPERSEDE always keeps the newer memory's content
  - Per-pair token tracking for efficiency analysis
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

DEFAULT_BASE_URL = "https://models-proxy.stepfun-inc.com/v1"
DEFAULT_API_KEY = "ak-2hx3cai1l5y5bkqkt3o8s7wr3fc8roal"
DEFAULT_MODEL = "gpt-4.1-mini"


def _clean(v: Any) -> str:
    return str(v or "").strip()


# ─── LLM API ─────────────────────────────────────────────────────────────────

def _chat(*, base_url: str, api_key: str, model_name: str, prompt: str,
          max_retries: int = 3, timeout: int = 120) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1024,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    session = requests.Session()
    session.trust_env = False
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(min(2 * attempt, 10))
    return {}


def _extract_message(resp_json: Dict[str, Any]) -> str:
    try:
        return _clean(resp_json["choices"][0]["message"]["content"])
    except Exception:
        return ""


def _extract_usage(resp_json: Dict[str, Any]) -> Dict[str, int]:
    usage = resp_json.get("usage", {})
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


# ─── Prompts ──────────────────────────────────────────────────────────────────

LIGHTMEM_PROMPT = """\
You are a memory consolidation assistant. Given two potentially redundant or conflicting memories, decide how to update them.

Memory A (newer, recorded at {time_new}):
{content_new}

Memory B (older, recorded at {time_old}):
{content_old}

Instructions:
- If both memories describe the same fact/event/preference with overlapping information, choose MERGE and combine them into one comprehensive memory.
- If the newer memory (A) is a more recent/accurate version of the older one (B), choose SUPERSEDE and keep the newer version.
- If they describe genuinely different things, choose KEEP_BOTH.
- The newer memory (A) takes precedence when there are conflicts.

Output strictly valid JSON with no other text:
{{"decision": "MERGE|SUPERSEDE|KEEP_BOTH", "merged_content": "the consolidated memory text if MERGE or SUPERSEDE, empty string if KEEP_BOTH", "reasoning": "brief explanation"}}"""

DIMMEM_PROMPT = """\
You are a memory consolidation assistant with access to dimensional metadata. Given two potentially redundant or conflicting memories, decide how to update them using the type-specific rules.

Memory A (newer, recorded at {source_time_new}):
  Content: {content_new}
  Type: {type_new}
  Time: {time_new}
  Keywords: {keywords_new}
  Reason: {reason_new}
  Purpose: {purpose_new}

Memory B (older, recorded at {source_time_old}):
  Content: {content_old}
  Type: {type_old}
  Time: {time_old}
  Keywords: {keywords_old}
  Reason: {reason_old}
  Purpose: {purpose_old}

Type-specific consolidation rules:
- fact: If both describe the same fact but values differ, SUPERSEDE with the newer one (Memory A). If they complement each other, MERGE.
- episodic: If both describe the exact same event/action, MERGE into one description. If they are different events (different times/contexts), KEEP_BOTH.
- profile: If both describe the same preference/habit/style, MERGE into a comprehensive description. If different aspects, KEEP_BOTH.

Note: Memory A is newer and takes precedence when there are conflicts.

Output strictly valid JSON with no other text:
{{"decision": "MERGE|SUPERSEDE|KEEP_BOTH", "merged_content": "consolidated memory text if MERGE or SUPERSEDE, empty if KEEP_BOTH", "merged_dimension": {{"memory_type": "...", "time": "...", "keywords": [...], "reason": "...", "purpose": "..."}}, "reasoning": "brief explanation"}}"""


def build_lightmem_prompt(pair: Dict[str, Any]) -> str:
    return LIGHTMEM_PROMPT.format(
        content_new=pair.get("content_new", ""),
        content_old=pair.get("content_old", ""),
        time_new=pair.get("source_time_new", "unknown"),
        time_old=pair.get("source_time_old", "unknown"),
    )


def build_dimmem_prompt(pair: Dict[str, Any]) -> str:
    return DIMMEM_PROMPT.format(
        content_new=pair.get("content_new", ""),
        content_old=pair.get("content_old", ""),
        source_time_new=pair.get("source_time_new", "unknown"),
        source_time_old=pair.get("source_time_old", "unknown"),
        type_new=pair.get("memory_type", "unknown"),
        type_old=pair.get("memory_type", "unknown"),
        time_new=pair.get("time_new", "unknown"),
        time_old=pair.get("time_old", "unknown"),
        keywords_new=", ".join(pair.get("keywords_new", [])),
        keywords_old=", ".join(pair.get("keywords_old", [])),
        reason_new=pair.get("reason_new", ""),
        reason_old=pair.get("reason_old", ""),
        purpose_new=pair.get("purpose_new", ""),
        purpose_old=pair.get("purpose_old", ""),
    )


def _parse_decision(raw_text: str) -> Dict[str, Any]:
    """Parse LLM decision JSON from response."""
    # Try direct JSON parse
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict) and "decision" in obj:
            return obj
    except Exception:
        pass
    # Try extracting JSON from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict) and "decision" in obj:
                return obj
        except Exception:
            pass
    # Try finding JSON object in text
    match = re.search(r"\{[^{}]*\"decision\"[^{}]*\}", raw_text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # Fallback
    for label in ("MERGE", "SUPERSEDE", "KEEP_BOTH"):
        if label in raw_text.upper():
            return {"decision": label, "merged_content": "", "reasoning": raw_text}
    return {"decision": "KEEP_BOTH", "merged_content": "", "reasoning": f"[parse_failed] {raw_text[:200]}"}


# ─── Process one pair ────────────────────────────────────────────────────────

def process_pair(
    pair: Dict[str, Any],
    method: str,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    if method == "lightmem":
        prompt = build_lightmem_prompt(pair)
    else:
        prompt = build_dimmem_prompt(pair)

    resp = _chat(base_url=base_url, api_key=api_key, model_name=model_name, prompt=prompt)
    raw_text = _extract_message(resp)
    usage = _extract_usage(resp)
    decision = _parse_decision(raw_text)

    return {
        "idx_new": pair["idx_new"],
        "idx_old": pair["idx_old"],
        "content_new": pair.get("content_new", ""),
        "content_old": pair.get("content_old", ""),
        "similarity": pair.get("similarity", 0.0),
        "decision": decision.get("decision", "KEEP_BOTH"),
        "merged_content": _clean(decision.get("merged_content")),
        "merged_dimension": decision.get("merged_dimension", {}),
        "reasoning": _clean(decision.get("reasoning")),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


# ─── Apply decisions ─────────────────────────────────────────────────────────

def apply_decisions(
    memories: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    method: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Apply merge/supersede decisions to produce updated memory bank.

    V2: SUPERSEDE always keeps the newer memory (idx_new).
    """
    removed_indices = set()
    added_memories: List[Dict[str, Any]] = []
    stats = {"merge": 0, "supersede": 0, "keep_both": 0}

    for d in decisions:
        action = d.get("decision", "KEEP_BOTH").upper()
        idx_new = d["idx_new"]
        idx_old = d["idx_old"]

        if idx_new in removed_indices or idx_old in removed_indices:
            stats["keep_both"] += 1
            continue

        if action == "MERGE":
            removed_indices.add(idx_new)
            removed_indices.add(idx_old)
            # Base on newer memory, update content
            merged = dict(memories[idx_new])
            merged["content"] = d.get("merged_content") or merged["content"]
            if method == "dimmem" and d.get("merged_dimension"):
                md = d["merged_dimension"]
                if isinstance(md, dict):
                    existing_dim = merged.get("dimension", {})
                    if isinstance(existing_dim, dict):
                        existing_dim.update({k: v for k, v in md.items() if v})
                        merged["dimension"] = existing_dim
            merged["_update_action"] = "merged"
            merged["_merged_from"] = [idx_new, idx_old]
            added_memories.append(merged)
            stats["merge"] += 1

        elif action == "SUPERSEDE":
            removed_indices.add(idx_new)
            removed_indices.add(idx_old)
            # SUPERSEDE: always keep the newer memory
            superseded = dict(memories[idx_new])
            superseded["content"] = d.get("merged_content") or superseded["content"]
            if method == "dimmem" and d.get("merged_dimension"):
                md = d["merged_dimension"]
                if isinstance(md, dict):
                    existing_dim = superseded.get("dimension", {})
                    if isinstance(existing_dim, dict):
                        existing_dim.update({k: v for k, v in md.items() if v})
                        superseded["dimension"] = existing_dim
            superseded["_update_action"] = "superseded"
            superseded["_superseded_from"] = [idx_new, idx_old]
            added_memories.append(superseded)
            stats["supersede"] += 1

        else:
            stats["keep_both"] += 1

    updated = [m for i, m in enumerate(memories) if i not in removed_indices]
    updated.extend(added_memories)
    return updated, stats
