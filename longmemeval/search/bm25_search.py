from __future__ import annotations

import math
import re
from typing import Any, Dict, List

from models import DimensionMemory, ParsedQuery


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _clean(v: Any) -> str:
    return str(v or "").strip()


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(_clean(text).lower())


def _record_text(record: Dict[str, Any]) -> str:
    return DimensionMemory.from_dict(record.get("dimension")).searchable_text(include_content=_clean(record.get("content")))


def _build_query_terms(parsed_query: Dict[str, Any]) -> List[str]:
    query = ParsedQuery.from_dict(parsed_query)
    seen = set()
    terms: List[str] = []
    for t in _tokens(query.bm25_text()):
        if t in seen:
            continue
        seen.add(t)
        terms.append(t)
    return terms


def _build_bm25_index(doc_tokens: List[List[str]]) -> Dict[str, Any]:
    n_docs = len(doc_tokens)
    doc_lens = [len(toks) for toks in doc_tokens]
    avgdl = (sum(doc_lens) / n_docs) if n_docs > 0 else 0.0

    dfs: Dict[str, int] = {}
    tfs: List[Dict[str, int]] = []
    for toks in doc_tokens:
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        tfs.append(tf)
        for t in tf.keys():
            dfs[t] = dfs.get(t, 0) + 1

    return {
        "n_docs": n_docs,
        "doc_lens": doc_lens,
        "avgdl": avgdl,
        "dfs": dfs,
        "tfs": tfs,
    }


def _idf(n_docs: int, df: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def _bm25_scores(query_terms: List[str], index: Dict[str, Any], *, k1: float, b: float) -> List[float]:
    n_docs = index["n_docs"]
    doc_lens: List[int] = index["doc_lens"]
    avgdl: float = index["avgdl"]
    dfs: Dict[str, int] = index["dfs"]
    tfs: List[Dict[str, int]] = index["tfs"]

    scores = [0.0 for _ in range(n_docs)]
    if n_docs == 0:
        return scores

    for q in query_terms:
        df = dfs.get(q, 0)
        if df <= 0:
            continue
        idf = _idf(n_docs, df)
        for i in range(n_docs):
            tf = tfs[i].get(q, 0)
            if tf <= 0:
                continue
            dl = doc_lens[i]
            denom = tf + k1 * (1.0 - b + b * (dl / (avgdl + 1e-12)))
            scores[i] += idf * (tf * (k1 + 1.0)) / (denom + 1e-12)
    return scores


def map_bm25_query(parsed_query: Dict[str, Any]) -> Dict[str, Any]:
    query = ParsedQuery.from_dict(parsed_query)
    return {
        "query_text": query.query_anchor,
        "keywords": list(query.keywords),
        "query_terms": _build_query_terms(parsed_query),
    }


def search_bm25(
    *,
    parsed_query: Dict[str, Any],
    records: List[Dict[str, Any]],
    top_k: int,
    k1: float = 1.2,
    b: float = 0.75,
) -> Dict[str, Any]:
    mapped = map_bm25_query(parsed_query)
    doc_tokens = [_tokens(_record_text(r)) for r in records]
    index = _build_bm25_index(doc_tokens)
    scores = _bm25_scores(mapped["query_terms"], index, k1=k1, b=b)

    ranked: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        row = dict(record)
        row["score"] = float(scores[i])
        row["score_components"] = {
            "bm25_score": float(scores[i]),
            "bm25_k1": float(k1),
            "bm25_b": float(b),
            "query_fields": ["query_anchor", "dimension.keywords"],
            "memory_fields": ["content", "dimension.reason", "dimension.purpose", "dimension.keywords"],
        }
        ranked.append(row)

    ranked.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
    return {
        "search_mode": "bm25",
        "mapped_query_analysis": mapped,
        "all_ranked_records": ranked,
        "top_records": ranked[:top_k],
    }
