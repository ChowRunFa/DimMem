from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


LOCOMO_QA_PROMPT = """You are answering a question about personal long-term memory.

You will receive:
1. The personal original question.
2. A small set of retrieved memory records.

Your job:
- Use only the retrieved memories as evidence.
- Prefer explicit facts over guesses.
- If the evidence is insufficient, say "I don't know".
- First write a short reasoning paragraph.
- Then give the final answer.

Reasoning rules:
- Keep the reasoning brief and evidence-grounded.
- Resolve temporal questions by comparing the timestamps in the memories when possible.
- If multiple records conflict, prefer the one with the latest source_time.
- If multiple records conflict and source_time is unavailable or tied, prefer the more explicit and more directly relevant one.
- Do not invent missing numbers, dates, places, or entities.

Answer rules:
- The final answer should be concise.
- If the answer is a count, return the count clearly.
- If the answer is a date or time difference, state the unit.
- If multiple answers are acceptable from the evidence, provide the most direct one.

Output format:
Reasoning: <brief reasoning>
Answer: <final answer>

User Question:
{{query}}

Retrieved Memories:
{{retrieved_memories}}
"""


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _memory_lines(record: Dict[str, Any], rank: int) -> List[str]:
    lines: List[str] = [f"[{rank}]"]
    source_time = _clean(record.get("source_time"))
    content = _clean(record.get("content"))

    if source_time:
        lines.append(f"source_time: {source_time}")
    if content:
        lines.append(f"content: {content}")

    dimension = record.get("dimension")
    if isinstance(dimension, dict):
        reason = _clean(dimension.get("reason"))
        purpose = _clean(dimension.get("purpose"))
    else:
        reason = _clean(record.get("reason"))
        purpose = _clean(record.get("purpose"))

    if reason:
        lines.append(f"reason: {reason}")
    if purpose:
        lines.append(f"purpose: {purpose}")

    return lines


def format_retrieved_memories(records: Iterable[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for idx, record in enumerate(records, start=1):
        blocks.append("\n".join(_memory_lines(record, idx)))
    return "\n\n".join(blocks) if blocks else "[No retrieved memories]"


def build_qa_prompt(*, query: str, retrieved_records: Iterable[Dict[str, Any]]) -> str:
    return (
        LOCOMO_QA_PROMPT.replace("{{query}}", _clean(query))
        .replace("{{retrieved_memories}}", format_retrieved_memories(retrieved_records))
    )


def build_qa_payload(*, query: str, retrieved_records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(retrieved_records)
    return {
        "query": _clean(query),
        "retrieved_records": records,
        "prompt": build_qa_prompt(query=query, retrieved_records=records),
    }


__all__ = [
    "LOCOMO_QA_PROMPT",
    "format_retrieved_memories",
    "build_qa_prompt",
    "build_qa_payload",
]
