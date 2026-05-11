"""
Local sentence-transformers embedding client.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResponse:
    embeddings: List[List[float]]
    usage: Dict[str, Any]
    model: str
    response_time: float


class LocalEmbeddingClient:
    """
    Uses sentence-transformers to load a local or HuggingFace embedding model.
    """

    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        if not (model or "").strip():
            raise ValueError("Local embedding model path or HuggingFace id is required")

        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Local embedding requires sentence-transformers and torch. "
                "Install via: pip install sentence-transformers torch"
            ) from e

        self.model = model.strip()
        self.batch_size = max(1, int(batch_size))

        if device and device.strip():
            dev = device.strip()
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"[LocalEmbeddingClient] Loading model={self.model!r} device={dev}")
        t0 = time.time()
        self._model = SentenceTransformer(self.model, device=dev)
        self.embedding_dim = int(self._model.get_sentence_embedding_dimension())
        logger.info(
            f"[LocalEmbeddingClient] Ready in {time.time() - t0:.2f}s, dim={self.embedding_dim}"
        )

    def embed_texts(self, texts: List[str]) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse(embeddings=[], usage={}, model=self.model, response_time=0.0)

        fixed: List[str] = []
        for t in texts:
            if t is None:
                fixed.append("")
            elif isinstance(t, str):
                fixed.append(t)
            else:
                fixed.append(json.dumps(t, ensure_ascii=False))

        start_time = time.time()
        arr = self._model.encode(
            fixed,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        if arr.ndim == 1:
            all_embeddings = [arr.astype(float).tolist()]
        else:
            all_embeddings = [row.astype(float).tolist() for row in arr]

        response_time = time.time() - start_time
        return EmbeddingResponse(
            embeddings=all_embeddings,
            usage={"prompt_tokens": 0, "total_tokens": 0},
            model=self.model,
            response_time=response_time,
        )

    def embed_text(self, text: str) -> List[float]:
        response = self.embed_texts([text])
        return response.embeddings[0] if response.embeddings else []

    def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        if len(vector1) != len(vector2):
            raise ValueError("Vector dimensions do not match")
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    def batch_cosine_similarity(
        self, query_vector: List[float], vectors: List[List[float]]
    ) -> List[float]:
        if not vectors:
            return []
        query = np.array(query_vector)
        matrix = np.array(vectors)
        dot_products = np.dot(matrix, query)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        similarities = np.divide(
            dot_products, norms, out=np.zeros_like(dot_products), where=norms != 0
        )
        return similarities.tolist()
