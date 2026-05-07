from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .structured_memory_v2 import V2_COMPAT_MEMORY_TYPES, V2_DIMENSION_KEYS, V2_MEMORY_TYPES


V2_QUERY_INTENTS = (
    "lookup",
    "compare",
    "aggregate",
    "yes_no",
    "counterfactual",
)

V2_ANSWER_FIELDS = (
    "content",
    "time",
    "location",
    "reason",
    "purpose",
    "entity",
    "boolean",
)

V2_QUERY_CONSTRAINT_KEYS = ("time", "location")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _dedupe(values: List[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = _clean(value)
        if not text:
            continue
        marker = text.lower()
        if marker in seen:
            continue
        seen.add(marker)
        result.append(text)
    return result


def _string_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return _dedupe(values)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def normalize_v2_intent(value: Any) -> str:
    text = _clean(value).lower()
    return text if text in V2_QUERY_INTENTS else "lookup"


def normalize_v2_answer_field(value: Any) -> str:
    text = _clean(value).lower()
    return text if text in V2_ANSWER_FIELDS else "content"


def normalize_v2_memory_types(values: Any) -> List[str]:
    result: List[str] = []
    for value in _string_list(values):
        text = value.lower()
        if text in V2_COMPAT_MEMORY_TYPES and text not in result:
            result.append(text)
    return result


@dataclass
class QueryConstraintsV2:
    values: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, List[str]]:
        return {key: list(values) for key, values in self.values.items() if values}

    @classmethod
    def from_dict(cls, data: Any) -> "QueryConstraintsV2":
        if not isinstance(data, dict):
            return cls()
        values: Dict[str, List[str]] = {}
        for raw_key, raw_value in data.items():
            key = _clean(raw_key).lower()
            if key not in V2_QUERY_CONSTRAINT_KEYS:
                continue
            cleaned = _string_list(raw_value)
            if cleaned:
                values[key] = cleaned
        return cls(values=values)

    def get(self, key: str) -> List[str]:
        return list(self.values.get(key, []))

    def all_values(self) -> List[str]:
        return _dedupe([value for values in self.values.values() for value in values])


@dataclass
class QueryAnalysisV2:
    query_text: str
    rewrite: str = ""
    intent: str = "lookup"
    answer_field: str = "content"
    content_query: str = ""
    target_memory_types: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    constraints: QueryConstraintsV2 = field(default_factory=QueryConstraintsV2)

    expected_answer_type: str = ""
    is_multi_entity: bool = False
    keywords: List[str] = field(default_factory=list)
    canonical_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_text,
            "rewrite": self.rewrite,
            "intent": self.intent,
            "answer_field": self.answer_field,
            "content_query": self.content_query,
            "target_memory_types": list(self.target_memory_types),
            "entities": list(self.entities),
            "constraints": self.constraints.to_dict(),
            "expected_answer_type": self.expected_answer_type,
            "is_multi_entity": self.is_multi_entity,
            "keywords": list(self.keywords),
            "canonical_text": self.canonical_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryAnalysisV2":
        return cls(
            query_text=_clean(data.get("query_text") or data.get("query")),
            rewrite=_clean(data.get("rewrite")),
            intent=normalize_v2_intent(data.get("intent")),
            answer_field=normalize_v2_answer_field(data.get("answer_field")),
            content_query=_clean(data.get("content_query")),
            target_memory_types=normalize_v2_memory_types(data.get("target_memory_types")),
            entities=_string_list(data.get("entities")),
            constraints=QueryConstraintsV2.from_dict(data.get("constraints")),
            expected_answer_type=_clean(data.get("expected_answer_type") or data.get("answer_type")).lower(),
            is_multi_entity=_bool_value(data.get("is_multi_entity")),
            keywords=_string_list(data.get("keywords")),
            canonical_text=_clean(data.get("canonical_text")),
        )


__all__ = [
    "V2_ANSWER_FIELDS",
    "V2_QUERY_INTENTS",
    "V2_QUERY_CONSTRAINT_KEYS",
    "QueryAnalysisV2",
    "QueryConstraintsV2",
    "normalize_v2_answer_field",
    "normalize_v2_intent",
    "normalize_v2_memory_types",
]
