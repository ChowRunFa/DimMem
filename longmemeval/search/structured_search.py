from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from shared.models.query_analysis_v2 import QueryAnalysisV2, QueryConstraintsV2
from shared.models.structured_memory_v2 import StructuredMemoryV2

from .time_constraints import time_constraint_match_score


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _string_list(values: Any) -> List[str]:
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


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(_clean(text).lower())


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _embed_text(embedding_client: Any, text: str) -> List[float]:
    if not _clean(text):
        return []
    try:
        return list(embedding_client.embed_text(text))
    except Exception:
        return []


def _similarity(embedding_client: Any, left: List[float], right: List[float]) -> float:
    if not left or not right:
        return 0.0
    try:
        return float(embedding_client.cosine_similarity(left, right))
    except Exception:
        return 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def _record_text(record: StructuredMemoryV2) -> str:
    parts = [record.content]
    parts.extend(record.entities)
    parts.extend(_clean(record.dimension.get(key)) for key in ("time", "location", "reason", "purpose"))
    return " | ".join(part for part in parts if part)


def _record_text_lower(record: StructuredMemoryV2) -> str:
    return _record_text(record).lower()


def _answer_field_score(answer_field: str, record: StructuredMemoryV2) -> float:
    field_name = _clean(answer_field).lower() or "content"
    if field_name == "content":
        return 1.0 if _clean(record.content) else 0.0
    if field_name == "time":
        return 1.0 if (_clean(record.dimension.get("time")) or record.source_time) else 0.0
    if field_name in {"location", "reason", "purpose"}:
        return 1.0 if _clean(record.dimension.get(field_name)) else 0.0
    return 0.0


def _memory_type_score(target_memory_types: List[str], record: StructuredMemoryV2) -> float:
    targets = {_clean(value).lower() for value in target_memory_types if _clean(value)}
    if not targets:
        return 0.0
    return 1.0 if _clean(record.memory_type).lower() in targets else 0.0


def _location_constraint_score(location_constraints: List[str], record: StructuredMemoryV2) -> float:
    constraints = [_clean(value).lower() for value in location_constraints if _clean(value)]
    if not constraints:
        return 0.0
    record_text = _record_text_lower(record)
    matched = sum(1 for constraint in constraints if constraint in record_text)
    return _safe_ratio(matched, len(constraints))


def _time_constraint_score(time_constraints: List[str], record: StructuredMemoryV2) -> float:
    return float(time_constraint_match_score(time_constraints, record)["score"])


def _keyword_phrase_score(keywords: List[str], record: StructuredMemoryV2) -> float:
    phrases = [_clean(value).lower() for value in keywords if _clean(value)]
    if not phrases:
        return 0.0
    record_text = _record_text_lower(record)
    matched = sum(1 for phrase in phrases if phrase in record_text)
    return _safe_ratio(matched, len(phrases))


def _keyword_token_overlap_score(keywords: List[str], record: StructuredMemoryV2) -> float:
    if not keywords:
        return 0.0
    query_tokens = set()
    for keyword in keywords:
        query_tokens.update(_tokens(keyword))
    if not query_tokens:
        return 0.0
    record_tokens = _token_set(_record_text(record))
    matched = len(query_tokens & record_tokens)
    return _safe_ratio(matched, len(query_tokens))


def _constraint_scores(analysis: QueryAnalysisV2, record: StructuredMemoryV2) -> Tuple[float, float]:
    location_score = _location_constraint_score(analysis.constraints.get("location"), record)
    time_score = _time_constraint_score(analysis.constraints.get("time"), record)
    return time_score, location_score


def _structural_score(analysis: QueryAnalysisV2, record: StructuredMemoryV2) -> Dict[str, Any]:
    answer_field_score = _answer_field_score(analysis.answer_field, record)
    memory_type_score = _memory_type_score(analysis.target_memory_types, record)
    time_match = time_constraint_match_score(analysis.constraints.get("time"), record)
    time_constraint_score = float(time_match["score"])
    location_constraint_score = _location_constraint_score(analysis.constraints.get("location"), record)
    keyword_phrase_score = _keyword_phrase_score(analysis.keywords, record)
    keyword_token_overlap_score = _keyword_token_overlap_score(analysis.keywords, record)

    base_weights = {
        "memory_type": 0.15,
        "time_constraint": 0.30,
        "location_constraint": 0.20,
        "keyword_phrase_match": 0.15,
        "keyword_token_overlap": 0.15,
    }

    active_components: Dict[str, float] = {}
    if analysis.target_memory_types:
        active_components["memory_type"] = memory_type_score
    if analysis.constraints.get("time"):
        active_components["time_constraint"] = time_constraint_score
    if analysis.constraints.get("location"):
        active_components["location_constraint"] = location_constraint_score
    if analysis.keywords:
        active_components["keyword_phrase_match"] = keyword_phrase_score
        active_components["keyword_token_overlap"] = keyword_token_overlap_score

    total_weight = sum(base_weights[name] for name in active_components)
    normalized_weights: Dict[str, float] = {}
    if total_weight > 0:
        normalized_weights = {
            name: base_weights[name] / total_weight
            for name in active_components
        }
    score = sum(
        normalized_weights.get(name, 0.0) * value
        for name, value in active_components.items()
    )
    return {
        "score": score,
        "answer_field_score": answer_field_score,
        "memory_type_score": memory_type_score,
        "time_constraint_score": time_constraint_score,
        "time_constraint_match": time_match,
        "location_constraint_score": location_constraint_score,
        "keyword_phrase_match_score": keyword_phrase_score,
        "keyword_token_overlap_score": keyword_token_overlap_score,
        "weights": normalized_weights,
        "active_components": active_components,
        "base_weights": base_weights,
    }


def map_structured_query(parsed_query: Dict[str, Any]) -> QueryAnalysisV2:
    dimension = parsed_query.get("dimension") if isinstance(parsed_query.get("dimension"), dict) else {}
    constraints: Dict[str, List[str]] = {}
    if _clean(dimension.get("time")):
        constraints["time"] = [_clean(dimension.get("time"))]
    if _clean(dimension.get("location")):
        constraints["location"] = [_clean(dimension.get("location"))]

    analysis = QueryAnalysisV2(
        query_text=_clean(parsed_query.get("query_anchor")),
        rewrite=_clean(parsed_query.get("query_anchor")),
        intent="lookup",
        answer_field=_clean(parsed_query.get("answer_dim")).lower() or "content",
        content_query=_clean(parsed_query.get("query_anchor")),
        target_memory_types=[],
        entities=[],
        constraints=QueryConstraintsV2.from_dict(constraints),
        keywords=_string_list(dimension.get("keywords")),
        canonical_text=_clean(parsed_query.get("query_anchor")),
    )
    analysis.target_memory_types = _string_list(dimension.get("target_memory_type"))
    analysis.entities = []
    return analysis


def search_structured(
    *,
    parsed_query: Dict[str, Any],
    records: List[StructuredMemoryV2],
    embedding_client: Any,
    top_k: int,
) -> Dict[str, Any]:
    analysis = map_structured_query(parsed_query)
    ranked: List[Dict[str, Any]] = []
    for record in records:
        structural_components = _structural_score(analysis, record)
        structural_score = float(structural_components["score"])

        row = record.to_dict()
        row.update(
            {
                "score": structural_score,
                "structural_score": structural_score,
                "score_components": {
                    **structural_components,
                    "query_fields": ["dimension.time", "dimension.location", "dimension.target_memory_type", "dimension.keywords", "query_anchor"],
                    "memory_fields": ["dimension.memory_type", "dimension.time", "dimension.location", "dimension.keywords", "content"],
                    "record_text": _record_text(record),
                },
            }
        )
        ranked.append(row)

    ranked.sort(
        key=lambda row: (
            row.get("score", 0.0),
            row.get("structural_score", 0.0),
            row.get("dense_score", 0.0),
        ),
        reverse=True,
    )
    return {
        "search_mode": "structured",
        "mapped_query_analysis": analysis.to_dict(),
        "all_ranked_records": ranked,
        "top_records": ranked[:top_k],
    }


def dumps_structured_debug(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
