from __future__ import annotations

from typing import Any, Dict, List

from models import DimensionMemory, ParsedQuery


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
        "query_text": ParsedQuery.from_dict(parsed_query).query_anchor,
    }


def search_embedding(
    *,
    parsed_query: Dict[str, Any],
    records: List[Dict[str, Any]],
    embedding_client: Any,
    top_k: int,
) -> Dict[str, Any]:
    mapped = map_embedding_query(parsed_query)
    query_embedding = _embed_text(embedding_client, mapped["query_text"])
    record_texts = []
    for record in records:
        dimension = DimensionMemory.from_dict(record.get("dimension"))
        text = " ".join(
            [
                _clean(record.get("content")),
                dimension.reason,
                dimension.purpose,
            ]
        ).strip()
        record_texts.append(text)

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
        row = dict(record)
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
