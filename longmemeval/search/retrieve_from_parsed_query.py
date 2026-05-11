#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


THIS_FILE = Path(__file__).resolve()
LONGMEMEVAL_DIR = THIS_FILE.parents[1]
SUBMIT_ROOT = THIS_FILE.parents[2]
if str(SUBMIT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMIT_ROOT))
if str(LONGMEMEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LONGMEMEVAL_DIR))

from models import DimensionMemory
from utils.local_embedding_client import LocalEmbeddingClient

from search import search_bm25, search_embedding, search_fused, search_structured, search_top15_content_dedup


DEFAULT_QUERY_PARSED = SUBMIT_ROOT / "results/query_analysis/parsed.json"
DEFAULT_MEMORY_DIR = SUBMIT_ROOT / "results/memories"
DEFAULT_OUTPUT_ROOT = SUBMIT_ROOT / "results/retrieval"
DEFAULT_EMBEDDING_MODEL = "/data/aios-weights/embeddings/all-MiniLM-L6-v2"


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


def load_parsed_query(parsed_path: Path) -> Dict[str, Any]:
    return json.loads(parsed_path.read_text(encoding="utf-8"))


def load_records(memory_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    all_memories_path = memory_dir / "all_memories.json"
    if all_memories_path.exists():
        payload = json.loads(all_memories_path.read_text(encoding="utf-8"))
        memory_sources = [("all_memories", payload.get("memories") if isinstance(payload, dict) else [])]
    else:
        memory_sources = []
        for path in sorted(memory_dir.glob("window_*/normalized_memories.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            memory_sources.append((path.parent.name, payload.get("memories") if isinstance(payload, dict) else []))

    for source_name, memories in memory_sources:
        for idx, row in enumerate(memories or []):
            if not isinstance(row, dict):
                continue
            window_name = _clean(row.get("window_dir")) or _clean(row.get("window_index")) or source_name
            dimension_model = DimensionMemory.from_dict(row.get("dimension"))
            keywords = dimension_model.keywords
            source_time_str = _clean(row.get("source_time"))
            source_time = None
            if source_time_str:
                try:
                    source_time = datetime.fromisoformat(source_time_str)
                except ValueError:
                    pass
            record = {
                "user_id": memory_dir.name,
                "memory_type": dimension_model.memory_type or "other",
                "content": _clean(row.get("content")),
                "dimension": dimension_model.to_dict(include_memory_type=False),
                "entities": keywords,
                "embedding_text": _clean(row.get("content")),
                "source_message_ids": [window_name],
                "source_boundary_id": f"{window_name}_{idx:04d}",
                "source_time": source_time,
                "record_time": datetime.now().isoformat(),
            }
            records.append(record)
    return records


def run_retrieval(
    *,
    query_parsed: Path,
    memory_dir: Path,
    output_root: Path,
    top_k: int,
    embedding_model: str,
    embedding_device: str,
) -> Path:
    parsed_query = load_parsed_query(query_parsed)
    records = load_records(memory_dir)

    question_type = query_parsed.parent.parent.name

    embedding_client = LocalEmbeddingClient(
        model=embedding_model,
        device=embedding_device,
        batch_size=32,
    )
    parse_mode = _clean(parsed_query.get("parse_mode")).lower() or "structured"
    if parse_mode in {"rrf_hybrid", "rrf", "hybrid", "fused"}:
        search_result = search_top15_content_dedup(
            parsed_query=parsed_query,
            records=records,
            embedding_client=embedding_client,
            top_k=top_k,
        )
    elif parse_mode == "hybrid_legacy":
        search_result = search_embedding(
            parsed_query=parsed_query,
            records=records,
            embedding_client=embedding_client,
            top_k=top_k,
        )
    elif parse_mode == "raw":
        search_result = search_bm25(
            parsed_query=parsed_query,
            records=records,
            top_k=top_k,
        )
    else:
        search_result = search_structured(
            parsed_query=parsed_query,
            records=records,
            embedding_client=embedding_client,
            top_k=top_k,
        )
    search_mode = search_result["search_mode"]
    run_dir = output_root / question_type / search_mode / query_parsed.parent.name
    run_dir.mkdir(parents=True, exist_ok=True)
    mapped_query = search_result["mapped_query_analysis"]
    ranked = search_result["all_ranked_records"]
    top_records = search_result["top_records"]

    experiment = {
        "query_parsed": str(query_parsed),
        "memory_dir": str(memory_dir),
        "output_dir": str(run_dir),
        "question_type": question_type,
        "top_k": top_k,
        "embedding_model": embedding_model,
        "embedding_device": embedding_device,
        "record_count": len(records),
        "retrieval_method": search_mode,
        "started_at": datetime.now().isoformat(),
    }
    (run_dir / "experiment_config.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "mapped_query_analysis.json").write_text(
        json.dumps(mapped_query, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "all_ranked_records.json").write_text(
        json.dumps(ranked, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (run_dir / "top_records.json").write_text(
        json.dumps(top_records, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    summary = {
        "parse_mode": parse_mode,
        "retrieval_method": search_mode,
        "query_text": mapped_query.get("query_text") or mapped_query.get("canonical_text") or "",
        "answer_field": mapped_query.get("answer_field", ""),
        "target_memory_types": list(mapped_query.get("target_memory_types") or []),
        "keywords": list(mapped_query.get("keywords") or []),
        "record_count": len(records),
        "top_k": top_k,
        "output_dir": str(run_dir),
        "top_records": [
            {
                "rank": i + 1,
                "score": row.get("score"),
                "memory_type": row.get("memory_type"),
                "content": row.get("content"),
                "source_boundary_id": row.get("source_boundary_id"),
                "score_components": row.get("score_components"),
            }
            for i, row in enumerate(top_records)
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-parsed", type=Path, default=DEFAULT_QUERY_PARSED)
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", default="cuda")
    args = parser.parse_args()

    run_dir = run_retrieval(
        query_parsed=args.query_parsed,
        memory_dir=args.memory_dir,
        output_root=args.output_root,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
    )
    print((run_dir / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
