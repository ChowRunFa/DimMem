"""
本地 / HuggingFace sentence-transformers 嵌入客户端，与 EmbeddingClient 接口对齐。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .embedding_client import EmbeddingResponse

logger = logging.getLogger(__name__)


class LocalEmbeddingClient:
    """
    使用 sentence-transformers 在本地或缓存目录加载模型。
    ``embedding_model`` 可为本地目录、或 HuggingFace 模型 id（如 sentence-transformers/all-MiniLM-L6-v2）。
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
            import importlib.metadata

            # In this workspace, transformers import can spend minutes scanning
            # every installed dist-info file through packages_distributions().
            # SentenceTransformer does not need that reverse mapping for local
            # model loading, so keep startup bounded.
            importlib.metadata.packages_distributions = lambda: {}
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "本地 embedding 需要安装 sentence-transformers 与 torch，请检查 requirements.txt"
            ) from e

        self.model = model.strip()
        self.batch_size = max(1, int(batch_size))

        if device and device.strip():
            dev = device.strip()
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"[LocalEmbeddingClient] Loading SentenceTransformer model={self.model!r} device={dev}")
        t0 = time.time()
        self._model = SentenceTransformer(self.model, device=dev)
        self.embedding_dim = int(self._model.get_sentence_embedding_dimension())
        logger.info(
            f"[LocalEmbeddingClient] Ready in {time.time() - t0:.2f}s, embedding_dim={self.embedding_dim}"
        )

        self.max_retries = 1
        self.retry_delay = 1.0
        self.timeout = 30.0

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
        # (N, dim) float32
        if arr.ndim == 1:
            all_embeddings = [arr.astype(float).tolist()]
        else:
            all_embeddings = [row.astype(float).tolist() for row in arr]

        response_time = time.time() - start_time
        usage = {"prompt_tokens": 0, "total_tokens": 0}
        return EmbeddingResponse(
            embeddings=all_embeddings,
            usage=usage,
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

    def normalize_vector(self, vector: List[float]) -> List[float]:
        v = np.array(vector)
        norm = np.linalg.norm(v)
        if norm == 0:
            return vector
        return (v / norm).tolist()

    def vector_distance(
        self, vector1: List[float], vector2: List[float], metric: str = "cosine"
    ) -> float:
        if len(vector1) != len(vector2):
            raise ValueError("Vector dimensions do not match")
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        if metric == "cosine":
            return 1 - self.cosine_similarity(vector1, vector2)
        if metric == "euclidean":
            return float(np.linalg.norm(v1 - v2))
        if metric == "manhattan":
            return float(np.sum(np.abs(v1 - v2)))
        raise ValueError(f"Unsupported distance metric: {metric}")

    def count_tokens(self, text: str) -> int:
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        english_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + english_chars / 4)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "backend": "local_sentence_transformers",
            "batch_size": self.batch_size,
        }
