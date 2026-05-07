"""Utility modules with lazy imports."""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "EmbeddingClient",
    "LocalEmbeddingClient",
    "EmbeddingResponse",
]


def __getattr__(name: str) -> Any:
    if name == "EmbeddingClient":
        return import_module(".embedding_client", __name__).EmbeddingClient
    if name in ("EmbeddingResponse",):
        return import_module(".embedding_client", __name__).EmbeddingResponse
    if name == "LocalEmbeddingClient":
        return import_module(".local_embedding_client", __name__).LocalEmbeddingClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
