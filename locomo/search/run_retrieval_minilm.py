#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


THIS_FILE = Path(__file__).resolve()
LOCOMO_SRC_ROOT = THIS_FILE.parents[1]
if str(LOCOMO_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCOMO_SRC_ROOT))

from models import DimensionMemory, ParsedQuery

DEFAULT_QUERY_ROOT = Path("./results/locomo_query_analysis")
DEFAULT_MEMORY_ROOT = Path("./results/locomo_memory")
DEFAULT_OUTPUT_BASE = Path("./results/locomo_retrieval")
DEFAULT_EMBEDDING_MODEL = "/data/aios-weights/embeddings/all-MiniLM-L6-v2"


def _clean(v: Any) -> str:
    return str(v or "").strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _memory_text(row: Dict[str, Any]) -> str:
    dimension = DimensionMemory.from_dict(row.get("dimension"))
    parts = [
        _clean(row.get("content")),
        dimension.reason,
        dimension.purpose,
    ]
    return " | ".join([part for part in parts if part])


def _query_text(parsed: Dict[str, Any]) -> str:
    return ParsedQuery.from_dict(parsed).query_anchor


def _find_memory_file(memory_root: Path, conv_name: str) -> Path | None:
    direct = memory_root / conv_name / "all_memories.json"
    if direct.exists():
        return direct
    conv_nested = sorted((memory_root / conv_name).glob("*/all_memories.json"))
    if conv_nested:
        return conv_nested[-1]
    run_level = sorted(memory_root.glob(f"*/{conv_name}/all_memories.json"))
    if run_level:
        return run_level[-1]
    nested = sorted(memory_root.glob(f"*/{conv_name}/*/all_memories.json"))
    if nested:
        return nested[-1]
    return None


def _collect_query_items(query_run_root: Path) -> List[Tuple[str, Path]]:
    rows: List[Tuple[str, Path]] = []
    for conv_dir in sorted(query_run_root.iterdir()):
        if not conv_dir.is_dir():
            continue
        for sample_dir in sorted(conv_dir.iterdir()):
            parsed = sample_dir / "parsed.json"
            if parsed.exists():
                rows.append((conv_dir.name, parsed))
    return rows


def _normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def run(args: argparse.Namespace) -> Path:
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.output_base / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.embedding_model, device=device)

    query_items = _collect_query_items(args.query_run_root)
    done = 0
    fail = 0
    skipped = 0
    summary: List[Dict[str, Any]] = []

    conv_cache: Dict[str, Dict[str, Any]] = {}

    _write_json(
        out_root / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "query_run_root": str(args.query_run_root),
            "memory_root": str(args.memory_root),
            "output_root": str(out_root),
            "top_k": args.top_k,
            "total_queries": len(query_items),
            "search_mode": "dense_minilm",
            "embedding_model": args.embedding_model,
            "device": device,
        },
    )

    for conv_name, parsed_path in query_items:
        sample_id = parsed_path.parent.name
        out_dir = out_root / conv_name / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)

        parsed = _load_json(parsed_path)

        if conv_name not in conv_cache:
            mem_file = _find_memory_file(args.memory_root, conv_name)
            if mem_file is None:
                conv_cache[conv_name] = {"error": "missing_memory"}
            else:
                payload = _load_json(mem_file)
                if isinstance(payload, dict) and isinstance(payload.get("memories"), list):
                    memories = payload.get("memories") or []
                elif isinstance(payload, list):
                    memories = payload
                else:
                    memories = []
                mem_texts = [_memory_text(m) for m in memories]
                mem_emb = model.encode(mem_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=128) if mem_texts else np.zeros((0, 384), dtype=np.float32)
                conv_cache[conv_name] = {
                    "memory_file": str(mem_file),
                    "memories": memories,
                    "mem_emb": mem_emb,
                }

        cached = conv_cache[conv_name]
        if "error" in cached:
            skipped += 1
            rec = {
                "conv_name": conv_name,
                "sample_id": sample_id,
                "ok": False,
                "error": cached["error"],
                "query_parsed": str(parsed_path),
                "memory_file": None,
                "output_dir": str(out_dir),
            }
            _write_json(out_dir / "result.json", rec)
            summary.append(rec)
            continue

        try:
            query_text = _query_text(parsed)
            q_emb = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
            mem_emb = cached["mem_emb"]
            memories = cached["memories"]
            if mem_emb.shape[0] == 0:
                ranked: List[Dict[str, Any]] = []
            else:
                scores = np.dot(mem_emb, q_emb)
                order = np.argsort(-scores)
                ranked = []
                for idx in order.tolist():
                    row = dict(memories[idx])
                    row["score"] = float(scores[idx])
                    row["score_components"] = {"dense_cosine_score": float(scores[idx])}
                    ranked.append(row)
            top_records = ranked[: args.top_k]

            _write_json(out_dir / "parsed.json", parsed)
            _write_json(
                out_dir / "retrieval_summary.json",
                {
                    "search_mode": "dense_minilm",
                    "conv_name": conv_name,
                    "sample_id": sample_id,
                    "query_parsed": str(parsed_path),
                    "memory_file": cached["memory_file"],
                    "memory_count": len(memories),
                    "top_k": args.top_k,
                    "embedding_model": args.embedding_model,
                },
            )
            _write_json(out_dir / "top_records.json", top_records)
            _write_json(out_dir / "all_ranked_records.json", ranked)

            rec = {
                "conv_name": conv_name,
                "sample_id": sample_id,
                "ok": True,
                "error": None,
                "query_parsed": str(parsed_path),
                "memory_file": cached["memory_file"],
                "memory_count": len(memories),
                "output_dir": str(out_dir),
            }
            _write_json(out_dir / "result.json", rec)
            summary.append(rec)
            done += 1
        except Exception as exc:
            rec = {
                "conv_name": conv_name,
                "sample_id": sample_id,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "query_parsed": str(parsed_path),
                "memory_file": cached.get("memory_file"),
                "output_dir": str(out_dir),
            }
            _write_json(out_dir / "result.json", rec)
            summary.append(rec)
            fail += 1

        _write_json(
            out_root / "status.json",
            {
                "total": len(query_items),
                "done": done,
                "fail": fail,
                "skipped_missing_memory": skipped,
            },
        )

    final = {
        "total": len(query_items),
        "done": done,
        "fail": fail,
        "skipped_missing_memory": skipped,
        "rows": summary,
    }
    _write_json(out_root / "summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LoCoMo dense retrieval with MiniLM.")
    parser.add_argument("--query-run-root", type=Path, default=DEFAULT_QUERY_ROOT)
    parser.add_argument("--memory-root", type=Path, default=DEFAULT_MEMORY_ROOT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
