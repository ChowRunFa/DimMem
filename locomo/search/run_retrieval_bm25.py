#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_QUERY_ROOT = Path("./results/locomo_query_analysis")
DEFAULT_MEMORY_ROOT = Path("./results/locomo_memory")
DEFAULT_OUTPUT_BASE = Path("./results/locomo_retrieval")

THIS_FILE = Path(__file__).resolve()
LOCOMO_SRC_ROOT = THIS_FILE.parents[1]
if str(LOCOMO_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCOMO_SRC_ROOT))

from models import DimensionMemory, ParsedQuery

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _clean(v: Any) -> str:
    return str(v or "").strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(_clean(text).lower())


def _memory_text(row: Dict[str, Any]) -> str:
    return DimensionMemory.from_dict(row.get("dimension")).searchable_text(include_content=_clean(row.get("content")))


def _build_query_tokens(parsed: Dict[str, Any]) -> List[str]:
    toks = _tokens(ParsedQuery.from_dict(parsed).bm25_text())
    # BM25 query term set to reduce repeated term overweight from anchor wording
    uniq = []
    seen = set()
    for t in toks:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


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
    # BM25+ style stable positive idf
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def _bm25_score(
    query_terms: List[str],
    index: Dict[str, Any],
    *,
    k1: float,
    b: float,
) -> List[float]:
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


def run(args: argparse.Namespace) -> Path:
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.output_base / run_name
    out_root.mkdir(parents=True, exist_ok=True)

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
            "search_mode": "bm25_anchor_keywords",
            "bm25_k1": args.k1,
            "bm25_b": args.b,
            "query_fields": ["query_anchor", "dimension.keywords"],
            "memory_fields": ["content", "dimension.reason", "dimension.purpose", "dimension.keywords"],
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
                doc_tokens = [_tokens(_memory_text(m)) for m in memories]
                index = _build_bm25_index(doc_tokens)
                conv_cache[conv_name] = {
                    "memory_file": str(mem_file),
                    "memories": memories,
                    "index": index,
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
            memories = cached["memories"]
            index = cached["index"]
            query_terms = _build_query_tokens(parsed)
            scores = _bm25_score(query_terms, index, k1=args.k1, b=args.b)
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            ranked: List[Dict[str, Any]] = []
            for idx in order:
                row = dict(memories[idx])
                row["score"] = float(scores[idx])
                row["score_components"] = {"bm25_score": float(scores[idx])}
                ranked.append(row)
            top_records = ranked[: args.top_k]

            _write_json(out_dir / "parsed.json", parsed)
            _write_json(
                out_dir / "retrieval_summary.json",
                {
                    "search_mode": "bm25_anchor_keywords",
                    "conv_name": conv_name,
                    "sample_id": sample_id,
                    "query_parsed": str(parsed_path),
                    "memory_file": cached["memory_file"],
                    "memory_count": len(memories),
                    "top_k": args.top_k,
                    "bm25_k1": args.k1,
                    "bm25_b": args.b,
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
    parser = argparse.ArgumentParser(description="Run LoCoMo BM25 retrieval.")
    parser.add_argument("--query-run-root", type=Path, default=DEFAULT_QUERY_ROOT)
    parser.add_argument("--memory-root", type=Path, default=DEFAULT_MEMORY_ROOT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
