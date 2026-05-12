"""Shared embedding gateway for PT IQ.

Loads the FastEmbed text model once and exposes a single embed(text) function
that returns a 384-dimension vector for use across tenders and business DNA.
"""

from __future__ import annotations

import os
from typing import Final

from fastembed import TextEmbedding


DEFAULT_EMBEDDING_MODEL: Final[str] = "BAAI/bge-small-en-v1.5"
EXPECTED_VECTOR_SIZE: Final[int] = 384

MODEL_NAME: Final[str] = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

# Load once at module import so every agent shares the same model instance.
_EMBEDDING_MODEL = TextEmbedding(model_name=MODEL_NAME)


def embed(text: str) -> list[float]:
    """Embed a single text string into a dense 384-dimension vector."""
    if not isinstance(text, str):
        raise TypeError("embed(text) expects a string input.")

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("embed(text) received empty text.")

    embedding = next(_EMBEDDING_MODEL.embed([cleaned]))
    vector = embedding.tolist()

    if len(vector) != EXPECTED_VECTOR_SIZE:
        raise ValueError(
            f"Unexpected embedding size from model '{MODEL_NAME}': "
            f"expected {EXPECTED_VECTOR_SIZE}, got {len(vector)}."
        )

    return vector
