#!/usr/bin/env python3
"""Conflict detection for memory update ablation (V2).

Two methods:
  - LightMem: embed ALL N memories → full N×N pairwise sim → filter > 0.85 → time filter
  - DimMem:   type grouping (free) → keyword Jaccard blocking (>0.3) → embed ONLY candidate
              subset → sim verify (>0.7) → time filter

V2 changes vs V1:
  1. DimMem only embeds memories involved in keyword-matched candidates (K << N)
  2. Time constraint: only newer source_time can update older source_time
  3. Detailed efficiency stats: embedding_count, pairs_checked, pairs_to_llm, time_filtered
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


EMBEDDING_MODEL_DIR = "/data/aios-weights/embeddings/all-MiniLM-L6-v2"


# ─── Utilities ────────────────────────────────────────────────────────────────

def _clean(v: Any) -> str:
    return str(v or "").strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_memories(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("memories"), list):
        return data["memories"]
    return data if isinstance(data, list) else []


def _get_keywords(mem: Dict[str, Any]) -> Set[str]:
    dim = mem.get("dimension") if isinstance(mem.get("dimension"), dict) else {}
    kws = dim.get("keywords") if isinstance(dim.get("keywords"), list) else []
    return {_clean(k).lower() for k in kws if _clean(k)}


def _get_memory_type(mem: Dict[str, Any]) -> str:
    dim = mem.get("dimension") if isinstance(mem.get("dimension"), dict) else {}
    mt = _clean(mem.get("memory_type")) or _clean(dim.get("memory_type"))
    return mt.lower() if mt else "other"


def _parse_source_time(mem: Dict[str, Any]) -> Optional[datetime]:
    """Parse source_time field to datetime. Returns None if unparseable."""
    raw = _clean(mem.get("source_time"))
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _time_order_pair(
    memories: List[Dict[str, Any]], idx_a: int, idx_b: int
) -> Tuple[int, int, bool]:
    """Return (idx_new, idx_old, time_filtered).

    Rules:
      - If both have valid source_time and they differ: newer first, older second.
      - If times are identical: allow update (not filtered), keep original order.
      - If either time is missing: allow update (not filtered), keep original order.

    Returns: (idx_new, idx_old, was_time_filtered)
      was_time_filtered=True means the pair should be SKIPPED (older cannot update newer)
    """
    t_a = _parse_source_time(memories[idx_a])
    t_b = _parse_source_time(memories[idx_b])

    if t_a is not None and t_b is not None:
        if t_a > t_b:
            return idx_a, idx_b, False  # A is newer → A updates B
        elif t_b > t_a:
            return idx_b, idx_a, False  # B is newer → B updates A
        else:
            return idx_a, idx_b, False  # Same time, allow
    # Missing time — allow, keep original order
    return idx_a, idx_b, False


# ─── ONNX Embedding Model ────────────────────────────────────────────────────

def load_embedding_model(model_dir: str = EMBEDDING_MODEL_DIR):
    model_path = Path(model_dir)
    import onnxruntime as ort
    from tokenizers import Tokenizer

    class OnnxEmbedder:
        def __init__(self, root, max_length=256, batch_size=512):
            self.batch_size = batch_size
            self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
            self.tokenizer.enable_truncation(max_length=max_length)
            self.tokenizer.enable_padding(length=max_length)
            self.session = ort.InferenceSession(
                str(root / "onnx" / "model.onnx"),
                providers=["CPUExecutionProvider"],
            )
            self.input_names = {item.name for item in self.session.get_inputs()}

        def encode(self, texts: List[str]) -> np.ndarray:
            all_emb = []
            for start in range(0, len(texts), self.batch_size):
                batch = [str(x or "") for x in texts[start : start + self.batch_size]]
                encodings = self.tokenizer.encode_batch(batch)
                input_ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
                attention_mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
                feed = {"input_ids": input_ids, "attention_mask": attention_mask}
                if "token_type_ids" in self.input_names:
                    feed["token_type_ids"] = np.asarray(
                        [e.type_ids for e in encodings], dtype=np.int64
                    )
                outputs = self.session.run(None, feed)
                token_emb = outputs[0]
                mask = attention_mask[..., None].astype(np.float32)
                summed = (token_emb * mask).sum(axis=1)
                counts = np.clip(mask.sum(axis=1), 1e-9, None)
                pooled = summed / counts
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                pooled = np.divide(pooled, norms, out=np.zeros_like(pooled), where=norms != 0)
                all_emb.append(pooled.astype(np.float32))
            return np.concatenate(all_emb, axis=0) if all_emb else np.zeros((0, 384), dtype=np.float32)

    return OnnxEmbedder(model_path)


# ─── LightMem Detection ──────────────────────────────────────────────────────

def detect_lightmem(
    memories: List[Dict[str, Any]],
    embedder,
    threshold: float = 0.85,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """LightMem: embed ALL → full N×N pairwise → filter → time order.

    Cost: O(N) embeddings + O(N²) similarity computation.

    Returns: (pairs, stats)
    """
    n = len(memories)
    if n < 2:
        return [], {"total_memories": n, "embedding_count": 0, "pairs_checked": 0,
                    "pairs_before_time_filter": 0, "time_filtered": 0, "pairs_to_llm": 0}

    # Step 1: Embed ALL memories
    contents = [_clean(m.get("content")) for m in memories]
    embs = embedder.encode(contents)
    embedding_count = n

    # Step 2: Full N×N pairwise similarity
    sim = embs @ embs.T
    pairs_checked = n * (n - 1) // 2

    # Step 3: Filter by threshold
    rows, cols = np.where(np.triu(sim, k=1) >= threshold)
    pairs_before_time = len(rows)

    # Step 4: Time ordering + filtering
    pairs = []
    time_filtered = 0
    for r, c in zip(rows.tolist(), cols.tolist()):
        idx_new, idx_old, filtered = _time_order_pair(memories, r, c)
        if filtered:
            time_filtered += 1
            continue
        pairs.append({
            "idx_new": idx_new,
            "idx_old": idx_old,
            "content_new": _clean(memories[idx_new].get("content")),
            "content_old": _clean(memories[idx_old].get("content")),
            "source_time_new": _clean(memories[idx_new].get("source_time")),
            "source_time_old": _clean(memories[idx_old].get("source_time")),
            "similarity": round(float(sim[r, c]), 4),
            "detection_method": "embedding_only",
        })

    stats = {
        "total_memories": n,
        "embedding_count": embedding_count,
        "pairs_checked": pairs_checked,
        "pairs_before_time_filter": pairs_before_time,
        "time_filtered": time_filtered,
        "pairs_to_llm": len(pairs),
    }
    return pairs, stats


# ─── DimMem Detection ────────────────────────────────────────────────────────

def detect_dimmem(
    memories: List[Dict[str, Any]],
    embedder,
    jaccard_threshold: float = 0.3,
    sim_threshold: float = 0.7,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """DimMem: type grouping → keyword blocking → subset embedding → sim verify → time filter.

    Cost: O(1) grouping + O(M²) keyword ops per group + O(K) embeddings where K << N.

    Returns: (pairs, stats)
    """
    n = len(memories)
    if n < 2:
        return [], {"total_memories": n, "embedding_count": 0,
                    "type_groups": {}, "keyword_pairs_checked": 0,
                    "keyword_candidates": 0, "sim_passed": 0,
                    "time_filtered": 0, "pairs_to_llm": 0}

    # Step 1: Group by memory_type (FREE — no computation)
    all_types = [_get_memory_type(m) for m in memories]
    all_keywords = [_get_keywords(m) for m in memories]

    type_groups: Dict[str, List[int]] = {}
    for i, mt in enumerate(all_types):
        type_groups.setdefault(mt, []).append(i)

    type_group_stats = {mt: len(indices) for mt, indices in type_groups.items()}

    # Step 2: Within each type group, keyword Jaccard blocking
    keyword_pairs_checked = 0
    keyword_candidates: List[Tuple[int, int, float]] = []

    for mt, indices in type_groups.items():
        m = len(indices)
        for ai in range(m):
            for bi in range(ai + 1, m):
                idx_a, idx_b = indices[ai], indices[bi]
                kw_a, kw_b = all_keywords[idx_a], all_keywords[idx_b]
                keyword_pairs_checked += 1
                if not kw_a or not kw_b:
                    continue
                inter = len(kw_a & kw_b)
                union = len(kw_a | kw_b)
                jaccard = inter / union if union > 0 else 0.0
                if jaccard >= jaccard_threshold:
                    keyword_candidates.append((idx_a, idx_b, jaccard))

    # Step 3: Embed ONLY the subset of memories involved in candidates
    unique_indices = set()
    for idx_a, idx_b, _ in keyword_candidates:
        unique_indices.add(idx_a)
        unique_indices.add(idx_b)

    embedding_count = len(unique_indices)

    if embedding_count == 0:
        stats = {
            "total_memories": n,
            "embedding_count": 0,
            "type_groups": type_group_stats,
            "keyword_pairs_checked": keyword_pairs_checked,
            "keyword_candidates": 0,
            "sim_passed": 0,
            "time_filtered": 0,
            "pairs_to_llm": 0,
        }
        return [], stats

    sorted_indices = sorted(unique_indices)
    idx_to_local = {idx: local for local, idx in enumerate(sorted_indices)}
    contents = [_clean(memories[i].get("content")) for i in sorted_indices]
    embs = embedder.encode(contents)

    # Step 4: Embedding similarity verification
    sim_passed_candidates = []
    for idx_a, idx_b, jaccard in keyword_candidates:
        local_a = idx_to_local[idx_a]
        local_b = idx_to_local[idx_b]
        sim_val = float(embs[local_a] @ embs[local_b])
        if sim_val >= sim_threshold:
            sim_passed_candidates.append((idx_a, idx_b, jaccard, sim_val))

    sim_passed = len(sim_passed_candidates)

    # Step 5: Time ordering + filtering
    pairs = []
    time_filtered = 0
    for idx_a, idx_b, jaccard, sim_val in sim_passed_candidates:
        idx_new, idx_old, filtered = _time_order_pair(memories, idx_a, idx_b)
        if filtered:
            time_filtered += 1
            continue

        dim_new = memories[idx_new].get("dimension") if isinstance(memories[idx_new].get("dimension"), dict) else {}
        dim_old = memories[idx_old].get("dimension") if isinstance(memories[idx_old].get("dimension"), dict) else {}

        pairs.append({
            "idx_new": idx_new,
            "idx_old": idx_old,
            "content_new": _clean(memories[idx_new].get("content")),
            "content_old": _clean(memories[idx_old].get("content")),
            "source_time_new": _clean(memories[idx_new].get("source_time")),
            "source_time_old": _clean(memories[idx_old].get("source_time")),
            "memory_type": all_types[idx_new],
            "keywords_new": sorted(all_keywords[idx_new]),
            "keywords_old": sorted(all_keywords[idx_old]),
            "keywords_jaccard": round(jaccard, 4),
            "time_new": _clean(dim_new.get("time")) or _clean(memories[idx_new].get("source_time")),
            "time_old": _clean(dim_old.get("time")) or _clean(memories[idx_old].get("source_time")),
            "reason_new": _clean(dim_new.get("reason")),
            "reason_old": _clean(dim_old.get("reason")),
            "purpose_new": _clean(dim_new.get("purpose")),
            "purpose_old": _clean(dim_old.get("purpose")),
            "similarity": round(sim_val, 4),
            "detection_method": "dimension_aware",
        })

    stats = {
        "total_memories": n,
        "embedding_count": embedding_count,
        "type_groups": type_group_stats,
        "keyword_pairs_checked": keyword_pairs_checked,
        "keyword_candidates": len(keyword_candidates),
        "sim_passed": sim_passed,
        "time_filtered": time_filtered,
        "pairs_to_llm": len(pairs),
    }
    return pairs, stats
