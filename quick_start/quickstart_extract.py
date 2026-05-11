#!/usr/bin/env python3
"""
Quickstart: Dimension Memory Extraction Demo

Demonstrates how to extract structured memories (with dimension fields)
from conversation windows on both LongMemEval and LoCoMo datasets.

Usage:
    # Run both demos (default)
    python quick_start/quickstart_extract.py \
        --base-url http://127.0.0.1:7790/v1 \
        --model-name qwen3-30b-a3b

    # Run only LongMemEval demo
    python quick_start/quickstart_extract.py --demo longmemeval

    # Run only LoCoMo demo
    python quick_start/quickstart_extract.py --demo locomo
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests

# ---------------------------------------------------------------------------
# Setup: add project roots to sys.path so we can import shared modules
# ---------------------------------------------------------------------------
SUBMIT_ROOT = Path(__file__).resolve().parents[1]
LONGMEMEVAL_ROOT = SUBMIT_ROOT / "longmemeval"
LOCOMO_ROOT = SUBMIT_ROOT / "locomo"

# Both pipelines share the same DimensionMemory class; import from longmemeval
if str(LONGMEMEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LONGMEMEVAL_ROOT))
from models import DimensionMemory

# Use importlib to load prompt modules explicitly (avoid namespace collision)
import importlib.util as _ilu


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, str(path))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_longmem_prompts = _load_module("longmem_prompts", LONGMEMEVAL_ROOT / "prompts" / "prompts.py")
LONGMEMEVAL_EXTRACTION_PROMPT = _longmem_prompts.LONGMEMEVAL_STRUCTURED_MEMORY_EXTRACTION_PROMPT

_locomo_prompts = _load_module("locomo_prompts", LOCOMO_ROOT / "prompts" / "prompts.py")
LOCOMO_EXTRACTION_PROMPT = _locomo_prompts.LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT
LOCOMO_OVERLAP_RULES = _locomo_prompts.OverlappingContextRules

# =========================================================================
# Inline test data
# =========================================================================

# --- LongMemEval: user-only conversation window (matches segmenter output) ---
LONGMEMEVAL_SAMPLE_WINDOW = """\
[2023-05-08T10:15:00, Mon] 1.User: I just started a new position as a data engineer at Spotify last week. The team uses Apache Spark and Airflow for our data pipelines.
[2023-05-08T10:15:00.500000, Mon] 2.User: My manager suggested I look into Delta Lake for our data lakehouse architecture. We are currently using Parquet files on S3.
[2023-05-08T10:15:01, Mon] 3.User: I also enrolled in an online course on distributed systems from MIT OpenCourseWare. I plan to finish it by the end of June.
[2023-05-15T14:30:00, Mon] 4.User: Yesterday I had a one-on-one with my manager and she approved my proposal to migrate our batch pipeline from Airflow to Dagster.
[2023-05-15T14:30:00.500000, Mon] 5.User: I've been reading the Dagster documentation this week and I really like its software-defined assets approach. It feels more intuitive than traditional DAG-based orchestration.
[2023-05-15T14:30:01, Mon] 6.User: By the way, my sister is visiting me in Stockholm next month. She lives in Tokyo and we haven't seen each other since last Christmas.
[2023-05-22T09:00:00, Mon] 7.User: I finished the first three modules of the MIT distributed systems course over the weekend. The Raft consensus algorithm section was especially interesting.
[2023-05-22T09:00:00.500000, Mon] 8.User: Our team is planning a hackathon in early June. I want to build a real-time feature store prototype using Apache Kafka and Redis."""

# --- LoCoMo: multi-person conversation window (matches segmenter output) ---
LOCOMO_SAMPLE_WINDOW = """\
[2023-07-20T20:56:00, Thu] 1.Emma: Hey Olivia! How was your weekend trip to Lake Tahoe?
[2023-07-20T20:56:00.500000, Thu] 2.Olivia: It was amazing! We went kayaking on Saturday morning and the water was so clear. My dog Max loved running on the beach.
[2023-07-20T20:56:01, Thu] 3.Emma: That sounds wonderful! I've been wanting to take my kids there. How long is the drive from San Francisco?
[2023-07-20T20:56:01.500000, Thu] 4.Olivia: About 3.5 hours. We stayed at a cabin near Emerald Bay. I actually took up photography recently and got some great shots of the sunrise over the lake.
[2023-07-20T20:56:02, Thu] 5.Emma: Oh nice! I remember you mentioned wanting to learn photography. What camera did you get?
[2023-07-20T20:56:02.500000, Thu] 6.Olivia: I bought a Sony A7 IV last month. It's a full-frame mirrorless camera. My colleague Jake recommended it since he's been doing landscape photography for years.
[2023-07-20T20:56:03, Thu] 7.Emma: That's a great choice! By the way, are you still doing yoga every Wednesday evening?
[2023-07-20T20:56:03.500000, Thu] 8.Olivia: Yes, I go to the studio on Valencia Street. It really helps me de-stress after work. I've been practicing for about two years now."""


# =========================================================================
# Shared helpers (reused from the extraction pipelines)
# =========================================================================

def _clean(value: Any) -> str:
    return str(value or "").strip()


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
            parsed, _ = decoder.raw_decode(payload[match.start():])
            return parsed
        except Exception:
            continue
    raise ValueError("unable to parse JSON from LLM response")


def _call_chat(*, base_url: str, api_key: str, model_name: str,
               prompt: str, max_tokens: int) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_text(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    return _clean((choices[0].get("message") or {}).get("content"))


def _source_time_by_id(dialogue: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for line in dialogue.splitlines():
        m = re.match(r"^\[(?P<ts>[^,\]]+)(?:,[^\]]*)?\]\s*(?P<sid>\d+)\.", line.strip())
        if m:
            mapping[int(m.group("sid"))] = _clean(m.group("ts"))
    return mapping


def _normalize_memory(row: Any, source_time_map: Dict[int, str]) -> Dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    try:
        source_id = int(row.get("source_id"))
    except (TypeError, ValueError):
        return None
    content = _clean(row.get("content"))
    if not content:
        return None
    return {
        "source_id": source_id,
        "source_time": source_time_map.get(source_id, ""),
        "content": content,
        "dimension": DimensionMemory.from_dict(row.get("dimension")).to_dict(),
    }


def _print_memories(memories: List[Dict[str, Any]]) -> None:
    for i, m in enumerate(memories, 1):
        print(f"\n  Memory #{i}")
        print(f"    source_id   : {m['source_id']}")
        print(f"    source_time : {m.get('source_time', '')}")
        print(f"    content     : {m['content']}")
        dim = m.get("dimension", {})
        print(f"    dimension:")
        print(f"      memory_type : {dim.get('memory_type', '')}")
        print(f"      time        : {dim.get('time', '')}")
        print(f"      location    : {dim.get('location', '')}")
        print(f"      reason      : {dim.get('reason', '')}")
        print(f"      purpose     : {dim.get('purpose', '')}")
        print(f"      keywords    : {dim.get('keywords', [])}")


# =========================================================================
# Demo: LongMemEval extraction
# =========================================================================

def demo_longmemeval(*, base_url: str, api_key: str, model_name: str,
                     max_tokens: int) -> None:
    print("=" * 70)
    print("Demo: LongMemEval Dimension Memory Extraction")
    print("=" * 70)

    conversation = LONGMEMEVAL_SAMPLE_WINDOW
    print("\n[Input conversation window]\n")
    print(conversation)

    # Build prompt (same logic as memory_constructor/run_extract_windows_with_en_prompt.py)
    prompt = LONGMEMEVAL_EXTRACTION_PROMPT.replace(
        "{conversation}", conversation
    )
    # For a single window (window_index=0), no overlap rule applies
    if "{overlap_rule}" in prompt:
        prompt = prompt.replace("{overlap_rule}", "")

    print(f"\n[Calling LLM: {model_name} at {base_url}]")
    resp = _call_chat(
        base_url=base_url, api_key=api_key,
        model_name=model_name, prompt=prompt, max_tokens=max_tokens,
    )
    raw_text = _extract_text(resp)
    parsed = _safe_json_fragment(raw_text)

    source_time_map = _source_time_by_id(conversation)
    memories: List[Dict[str, Any]] = []
    for row in (parsed.get("memories") or []):
        norm = _normalize_memory(row, source_time_map)
        if norm is not None:
            memories.append(norm)

    print(f"\n[Extracted {len(memories)} structured memories]")
    _print_memories(memories)

    print("\n[Raw JSON output]")
    print(json.dumps({"memories": memories}, ensure_ascii=False, indent=2))


# =========================================================================
# Demo: LoCoMo extraction
# =========================================================================

def demo_locomo(*, base_url: str, api_key: str, model_name: str,
                max_tokens: int) -> None:
    print("=" * 70)
    print("Demo: LoCoMo Dimension Memory Extraction")
    print("=" * 70)

    conversation = LOCOMO_SAMPLE_WINDOW
    print("\n[Input conversation window]\n")
    print(conversation)

    # Build prompt (same logic as locomo/memory_constructor/build_one_record_parallel.py)
    # Window index 0 -> no overlapping rules
    prompt = LOCOMO_EXTRACTION_PROMPT
    prompt = prompt.replace("{OverlappingContextRules}", "")
    prompt = prompt.replace("{{OverlappingContextRules}}", "")
    prompt = prompt.replace("{conversation}", conversation)
    prompt = prompt.replace("{{conversation}}", conversation)
    prompt = prompt.strip()

    print(f"\n[Calling LLM: {model_name} at {base_url}]")
    resp = _call_chat(
        base_url=base_url, api_key=api_key,
        model_name=model_name, prompt=prompt, max_tokens=max_tokens,
    )
    raw_text = _extract_text(resp)
    parsed = _safe_json_fragment(raw_text)

    source_time_map = _source_time_by_id(conversation)
    memories: List[Dict[str, Any]] = []
    for row in (parsed.get("memories") or []):
        norm = _normalize_memory(row, source_time_map)
        if norm is not None:
            memories.append(norm)

    print(f"\n[Extracted {len(memories)} structured memories]")
    _print_memories(memories)

    print("\n[Raw JSON output]")
    print(json.dumps({"memories": memories}, ensure_ascii=False, indent=2))


# =========================================================================
# CLI entry point
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quickstart demo for dimension memory extraction on LongMemEval and LoCoMo.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:7790/v1",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="qwen3-30b-a3b")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--demo", choices=["longmemeval", "locomo", "both"],
                        default="both",
                        help="Which dataset demo to run (default: both)")
    args = parser.parse_args()

    kwargs = dict(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
    )

    if args.demo in ("longmemeval", "both"):
        demo_longmemeval(**kwargs)
        if args.demo == "both":
            print("\n")

    if args.demo in ("locomo", "both"):
        demo_locomo(**kwargs)


if __name__ == "__main__":
    main()
