from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


V2_MEMORY_TYPES = (
    "event",
    "background",
    "relationship",
    "motivation",
    "value",
    "other",
)

LONGMEMEVAL_MEMORY_TYPES = (
    "preference",
    "interest",
    "consumption",
    "plan",
    "exploration",
    "habit",
    "experience",
)

V2_COMPAT_MEMORY_TYPES = V2_MEMORY_TYPES + tuple(
    memory_type for memory_type in LONGMEMEVAL_MEMORY_TYPES if memory_type not in V2_MEMORY_TYPES
)

V2_DIMENSION_KEYS = (
    "time",
    "location",
    "reason",
    "purpose",
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _dedupe_strings(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    result: List[str] = []
    seen = set()
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


def normalize_v2_memory_type(value: Any) -> str:
    text = _clean(value)
    return text if text in V2_COMPAT_MEMORY_TYPES else "other"


def normalize_v2_dimension(data: Any) -> Dict[str, str]:
    if not isinstance(data, dict):
        return {}
    result: Dict[str, str] = {}
    for raw_key, raw_value in data.items():
        key = _clean(raw_key)
        if key not in V2_DIMENSION_KEYS:
            continue
        value = _clean(raw_value)
        if value:
            result[key] = value
    return result


@dataclass
class StructuredMemoryV2:
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""

    memory_type: str = "other"
    content: str = ""
    dimension: Dict[str, str] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)

    embedding: Optional[List[float]] = None
    embedding_text: str = ""

    source_message_ids: List[str] = field(default_factory=list)
    source_boundary_id: str = ""
    source_time: Optional[datetime] = None
    record_time: datetime = field(default_factory=datetime.now)
    speaker: str = ""

    @property
    def memory_text(self) -> str:
        return self.content

    def to_dict(self, *, include_embedding: bool = False) -> Dict[str, Any]:
        payload = {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "dimension": dict(self.dimension),
            "entities": list(self.entities),
            "embedding_text": self.embedding_text,
            "source_message_ids": list(self.source_message_ids),
            "source_boundary_id": self.source_boundary_id,
            "source_time": self.source_time.isoformat() if self.source_time else None,
            "record_time": self.record_time.isoformat(),
            "speaker": self.speaker,
        }
        if include_embedding:
            payload["embedding"] = self.embedding
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredMemoryV2":
        source_time = _clean(data.get("source_time"))
        record_time = _clean(data.get("record_time"))
        return cls(
            memory_id=_clean(data.get("memory_id")) or str(uuid.uuid4()),
            user_id=_clean(data.get("user_id")),
            memory_type=normalize_v2_memory_type(data.get("memory_type")),
            content=_clean(data.get("content")),
            dimension=normalize_v2_dimension(data.get("dimension")),
            entities=_dedupe_strings(data.get("entities")),
            embedding=data.get("embedding"),
            embedding_text=_clean(data.get("embedding_text")),
            source_message_ids=_dedupe_strings(data.get("source_message_ids")),
            source_boundary_id=_clean(data.get("source_boundary_id")),
            source_time=datetime.fromisoformat(source_time) if source_time else None,
            record_time=datetime.fromisoformat(record_time) if record_time else datetime.now(),
            speaker=_clean(data.get("speaker")),
        )


__all__ = [
    "LONGMEMEVAL_MEMORY_TYPES",
    "V2_COMPAT_MEMORY_TYPES",
    "V2_MEMORY_TYPES",
    "V2_DIMENSION_KEYS",
    "StructuredMemoryV2",
    "normalize_v2_memory_type",
    "normalize_v2_dimension",
]
