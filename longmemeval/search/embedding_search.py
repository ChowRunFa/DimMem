from __future__ import annotations

from typing import Any, Dict, List

from shared.models.structured_memory_v2 import StructuredMemoryV2


def _clean(v: Any) -> str:
    return str(v or "").strip()


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


def map_embedding_query(parsed_query: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query_text": _clean(parsed_query.get("query_anchor")),
    }


def search_embedding(
    *,
    parsed_query: Dict[str, Any],
    records: List[StructuredMemoryV2],
    embedding_client: Any,
    top_k: int,
) -> Dict[str, Any]:
    mapped = map_embedding_query(parsed_query)
    query_embedding = _embed_text(embedding_client, mapped["query_text"])
    record_texts = [
        " ".join(
            [
                _clean(record.content),
                _clean(record.dimension.get("reason")),
                _clean(record.dimension.get("purpose")),
            ]
        ).strip()
        for record in records
    ]
    try:
        record_embeddings = embedding_client.embed_texts(record_texts).embeddings
    except Exception:
        record_embeddings = [_embed_text(embedding_client, text) for text in record_texts]
    try:
        dense_scores = embedding_client.batch_cosine_similarity(query_embedding, record_embeddings)
    except Exception:
        dense_scores = [
            _similarity(embedding_client, query_embedding, record_embedding)
            for record_embedding in record_embeddings
        ]

    ranked: List[Dict[str, Any]] = []
    for record, dense_score in zip(records, dense_scores):
        row = record.to_dict()
        row["score"] = float(dense_score)
        row["score_components"] = {
            "dense_cosine_score": float(dense_score),
            "query_field": "query_anchor",
            "memory_fields": ["content", "dimension.reason", "dimension.purpose"],
        }
        ranked.append(row)

    ranked.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
    return {
        "search_mode": "minilm",
        "mapped_query_analysis": mapped,
        "all_ranked_records": ranked,
        "top_records": ranked[:top_k],
    }
