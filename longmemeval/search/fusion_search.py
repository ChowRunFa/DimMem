from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from models import ParsedQuery

from .bm25_search import search_bm25
from .embedding_search import search_embedding
from .structured_search import search_structured


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _content_key(row: Dict[str, Any]) -> str:
    content = _clean(row.get("content"))
    if content:
        return "content:" + hashlib.md5(content.lower().encode("utf-8")).hexdigest()
    marker = "|".join(
        [
            _clean(row.get("source_boundary_id")),
            _clean(row.get("source_time")),
            str(row.get("dimension") or {}),
        ]
    )
    return "fallback:" + hashlib.md5(marker.encode("utf-8")).hexdigest()


def _annotate(row: Dict[str, Any], *, route: str, rank: int) -> Dict[str, Any]:
    item = dict(row)
    item["retrieval_method"] = route
    item["retrieval_rank"] = rank
    item["retrieval_score"] = float(row.get("score", 0.0) or 0.0)
    item["fusion_sources"] = [
        {
            "method": route,
            "rank": rank,
            "score": float(row.get("score", 0.0) or 0.0),
        }
    ]
    return item


def search_top15_content_dedup(
    *,
    parsed_query: Dict[str, Any],
    records: List[Dict[str, Any]],
    embedding_client: Any,
    top_k: int = 15,
) -> Dict[str, Any]:
    """
    Run three independent retrieval routes and merge their top-k results.

    Routes:
    - bm25: query_anchor + dimension.keywords against content/reason/purpose/keywords
    - structured: time/location/memory_type/keywords against structured memory fields
    - minilm: query_anchor embedding against content + reason + purpose embedding

    The final list preserves route order bm25 -> structured -> minilm and de-duplicates
    records by normalized content.
    """

    route_outputs = {
        "bm25": search_bm25(parsed_query=parsed_query, records=records, top_k=top_k),
        "structured": search_structured(
            parsed_query=parsed_query,
            records=records,
            embedding_client=embedding_client,
            top_k=top_k,
        ),
        "minilm": search_embedding(
            parsed_query=parsed_query,
            records=records,
            embedding_client=embedding_client,
            top_k=top_k,
        ),
    }

    seen: Dict[str, Dict[str, Any]] = {}
    ranked: List[Dict[str, Any]] = []
    for route in ("bm25", "structured", "minilm"):
        for rank, row in enumerate(route_outputs[route].get("top_records", []), start=1):
            key = _content_key(row)
            if key in seen:
                seen[key]["fusion_sources"].append(
                    {
                        "method": route,
                        "rank": rank,
                        "score": float(row.get("score", 0.0) or 0.0),
                    }
                )
                continue
            item = _annotate(row, route=route, rank=rank)
            seen[key] = item
            ranked.append(item)

    query = ParsedQuery.from_dict(parsed_query)
    mapped_query = {
        "query_text": query.query_anchor,
        "query_anchor": query.query_anchor,
        "dimension": query.dimension,
        "routes": {
            route: payload.get("mapped_query_analysis", {})
            for route, payload in route_outputs.items()
        },
    }

    return {
        "search_mode": "tri_top15_content_dedup",
        "mapped_query_analysis": mapped_query,
        "all_ranked_records": ranked,
        "top_records": ranked,
        "sub_results": {
            route: {
                "mapped_query_analysis": payload.get("mapped_query_analysis", {}),
                "top_records": payload.get("top_records", []),
            }
            for route, payload in route_outputs.items()
        },
        "route_top_k": top_k,
        "dedup_key": "content",
    }


def search_fused(
    *,
    parsed_query: Dict[str, Any],
    records: List[Dict[str, Any]],
    embedding_client: Any,
    top_k: int = 15,
    **_: Any,
) -> Dict[str, Any]:
    return search_top15_content_dedup(
        parsed_query=parsed_query,
        records=records,
        embedding_client=embedding_client,
        top_k=top_k,
    )
