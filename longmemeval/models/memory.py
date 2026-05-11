from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


VALID_MEMORY_TYPES = {"fact", "episodic", "profile"}


def clean(value: Any) -> str:
    return str(value or "").strip()


def unique_string_list(values: Any, *, lower_dedupe: bool = False) -> List[str]:
    """Normalize a scalar/list into a de-duplicated list of non-empty strings."""
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    result: List[str] = []
    seen = set()
    for value in values:
        text = clean(value)
        if not text:
            continue
        marker = text.lower() if lower_dedupe else text
        if marker in seen:
            continue
        seen.add(marker)
        result.append(text)
    return result


@dataclass
class DimensionMemory:
    """Structured dimension fields attached to a memory record."""

    memory_type: str = ""
    time: str = ""
    location: str = ""
    reason: str = ""
    purpose: str = ""
    keywords: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Any) -> "DimensionMemory":
        data = payload if isinstance(payload, dict) else {}
        memory_type = clean(data.get("memory_type")).lower()
        if memory_type not in VALID_MEMORY_TYPES:
            memory_type = ""
        return cls(
            memory_type=memory_type,
            time=clean(data.get("time")),
            location=clean(data.get("location")),
            reason=clean(data.get("reason")),
            purpose=clean(data.get("purpose")),
            keywords=unique_string_list(data.get("keywords")),
        )

    def to_dict(self, *, include_memory_type: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if include_memory_type:
            data["memory_type"] = self.memory_type
        data.update(
            {
                "time": self.time,
                "location": self.location,
                "reason": self.reason,
                "purpose": self.purpose,
                "keywords": list(self.keywords),
            }
        )
        return data

    def searchable_text(self, *, include_content: str = "") -> str:
        parts = [
            clean(include_content),
            self.reason,
            self.purpose,
            " ".join(self.keywords),
        ]
        return " | ".join(part for part in parts if clean(part))


@dataclass
class ParsedQuery:
    """Normalized result from the query parser."""

    parse_mode: str = ""
    query_anchor: str = ""
    target_memory_type: List[str] = field(default_factory=list)
    time: str = ""
    location: str = ""
    keywords: List[str] = field(default_factory=list)
    answer_dim: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Any) -> "ParsedQuery":
        data = payload if isinstance(payload, dict) else {}
        dimension = data.get("dimension") if isinstance(data.get("dimension"), dict) else {}
        return cls(
            parse_mode=clean(data.get("parse_mode")),
            query_anchor=clean(data.get("query_anchor")),
            target_memory_type=unique_string_list(dimension.get("target_memory_type"), lower_dedupe=True),
            time=clean(dimension.get("time")),
            location=clean(dimension.get("location")),
            keywords=unique_string_list(dimension.get("keywords"), lower_dedupe=True),
            answer_dim=clean(data.get("answer_dim")).lower(),
            raw=dict(data),
        )

    @property
    def dimension(self) -> Dict[str, Any]:
        return {
            "target_memory_type": list(self.target_memory_type),
            "time": self.time,
            "location": self.location,
            "keywords": list(self.keywords),
        }

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.raw)
        data.update(
            {
                "parse_mode": self.parse_mode,
                "query_anchor": self.query_anchor,
                "dimension": self.dimension,
                "answer_dim": self.answer_dim,
            }
        )
        return data

    def bm25_text(self) -> str:
        return " ".join([self.query_anchor] + self.keywords).strip()

    def constraints(self) -> Dict[str, List[str]]:
        constraints: Dict[str, List[str]] = {}
        if self.time:
            constraints["time"] = [self.time]
        if self.location:
            constraints["location"] = [self.location]
        return constraints

    def to_search_analysis(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_anchor,
            "rewrite": self.query_anchor,
            "intent": "lookup",
            "answer_field": self.answer_dim or "content",
            "content_query": self.query_anchor,
            "target_memory_types": list(self.target_memory_type),
            "entities": [],
            "constraints": self.constraints(),
            "keywords": list(self.keywords),
            "canonical_text": self.query_anchor,
        }


__all__ = ["DimensionMemory", "ParsedQuery", "VALID_MEMORY_TYPES", "clean", "unique_string_list"]
