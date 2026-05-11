#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(_clean(text).lower())


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _memory_text(row: Dict[str, Any]) -> str:
    dimension = DimensionMemory.from_dict(row.get("dimension"))
    parts = [
        _clean(row.get("content")),
        _clean(row.get("source_time")),
        dimension.time,
        dimension.location,
        dimension.reason,
        dimension.purpose,
        " ".join(dimension.keywords),
    ]
    return " | ".join([part for part in parts if part])


def _time_score(query_time: str, mem: Dict[str, Any]) -> float:
    if not _clean(query_time):
        return 0.0
    text = _memory_text(mem).lower()
    qt = _clean(query_time).lower()
    return 1.0 if qt in text else 0.0


def _location_score(query_location: str, mem: Dict[str, Any]) -> float:
    if not _clean(query_location):
        return 0.0
    return 1.0 if _clean(query_location).lower() in _memory_text(mem).lower() else 0.0


def _memory_type_score(target_types: List[str], mem: Dict[str, Any]) -> float:
    targets = {_clean(x).lower() for x in target_types if _clean(x)}
    if not targets:
        return 0.0
    mt = _clean(mem.get("memory_type") or (mem.get("dimension") or {}).get("memory_type")).lower()
    return 1.0 if mt in targets else 0.0


def _keyword_phrase_score(keywords: List[str], mem: Dict[str, Any]) -> float:
    ks = [_clean(x).lower() for x in keywords if _clean(x)]
    if not ks:
        return 0.0
    txt = _memory_text(mem).lower()
    hit = sum(1 for k in ks if k in txt)
    return hit / len(ks)


def _keyword_token_overlap_score(keywords: List[str], mem: Dict[str, Any]) -> float:
    ks = set()
    for k in keywords:
        ks.update(_tokens(k))
    if not ks:
        return 0.0
    ms = _token_set(_memory_text(mem))
    return len(ks & ms) / max(1, len(ks))


def _anchor_overlap_score(anchor: str, mem: Dict[str, Any]) -> float:
    q = _token_set(anchor)
    if not q:
        return 0.0
    m = _token_set(_memory_text(mem))
    return len(q & m) / max(1, len(q))


def _score_memory(parsed: Dict[str, Any], mem: Dict[str, Any]) -> Dict[str, Any]:
    query = ParsedQuery.from_dict(parsed)

    s_memory_type = _memory_type_score(query.target_memory_type, mem)
    s_time = _time_score(query.time, mem)
    s_location = _location_score(query.location, mem)
    s_kw_phrase = _keyword_phrase_score(query.keywords, mem)
    s_kw_tok = _keyword_token_overlap_score(query.keywords, mem)
    s_anchor = _anchor_overlap_score(query.query_anchor, mem)

    score = (
        0.15 * s_memory_type
        + 0.20 * s_time
        + 0.10 * s_location
        + 0.20 * s_kw_phrase
        + 0.15 * s_kw_tok
        + 0.20 * s_anchor
    )
    return {
        "score": float(score),
        "score_components": {
            "memory_type_score": s_memory_type,
            "time_constraint_score": s_time,
            "location_constraint_score": s_location,
            "keyword_phrase_match_score": s_kw_phrase,
            "keyword_token_overlap_score": s_kw_tok,
            "anchor_overlap_score": s_anchor,
        },
    }


def _find_memory_file(memory_root: Path, conv_name: str) -> Path | None:
    # 1) Direct layout: <memory_root>/<conv_name>/all_memories.json
    direct = memory_root / conv_name / "all_memories.json"
    if direct.exists():
        return direct
    # 1.5) Run-dir nested layout: <memory_root>/<conv_name>/<record>/all_memories.json
    # (e.g., memory_root already points to one concrete run directory)
    direct_nested = sorted((memory_root / conv_name).glob("*/all_memories.json"))
    if direct_nested:
        return direct_nested[-1]
    # 2) Run-root layout: <memory_root>/<run_name>/<conv_name>/all_memories.json
    run_level = sorted(memory_root.glob(f"*/{conv_name}/all_memories.json"))
    if run_level:
        return run_level[-1]
    # 3) Legacy nested layout: <memory_root>/<run_name>/<conv_name>/<record>/all_memories.json
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


def run(args: argparse.Namespace) -> Path:
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.output_base / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    query_items = _collect_query_items(args.query_run_root)
    done = 0
    fail = 0
    skipped = 0
    summary: List[Dict[str, Any]] = []

    _write_json(
        out_root / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "query_run_root": str(args.query_run_root),
            "memory_root": str(args.memory_root),
            "output_root": str(out_root),
            "top_k": args.top_k,
            "total_queries": len(query_items),
        },
    )

    for conv_name, parsed_path in query_items:
        sample_id = parsed_path.parent.name
        out_dir = out_root / conv_name / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)
        parsed = _load_json(parsed_path)

        mem_file = _find_memory_file(args.memory_root, conv_name)
        if mem_file is None:
            skipped += 1
            rec = {
                "conv_name": conv_name,
                "sample_id": sample_id,
                "ok": False,
                "error": "missing_memory",
                "query_parsed": str(parsed_path),
                "memory_file": None,
                "output_dir": str(out_dir),
            }
            _write_json(out_dir / "result.json", rec)
            summary.append(rec)
            continue

        try:
            memories_payload = _load_json(mem_file)
            if isinstance(memories_payload, dict) and isinstance(memories_payload.get("memories"), list):
                memories = memories_payload.get("memories") or []
            elif isinstance(memories_payload, list):
                memories = memories_payload
            else:
                raise ValueError("all_memories_not_list")
            ranked = []
            for mem in memories:
                row = dict(mem)
                scored = _score_memory(parsed, mem)
                row.update(scored)
                ranked.append(row)
            ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            top_records = ranked[: args.top_k]

            _write_json(out_dir / "parsed.json", parsed)
            _write_json(
                out_dir / "retrieval_summary.json",
                {
                    "search_mode": "structured_lexical",
                    "conv_name": conv_name,
                    "sample_id": sample_id,
                    "query_parsed": str(parsed_path),
                    "memory_file": str(mem_file),
                    "memory_count": len(memories),
                    "top_k": args.top_k,
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
                "memory_file": str(mem_file),
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
                "memory_file": str(mem_file),
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
    parser = argparse.ArgumentParser(description="Run LoCoMo retrieval from parsed query results.")
    parser.add_argument("--query-run-root", type=Path, required=True)
    parser.add_argument("--memory-root", type=Path, default=DEFAULT_MEMORY_ROOT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--top-k", type=int, default=40)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
