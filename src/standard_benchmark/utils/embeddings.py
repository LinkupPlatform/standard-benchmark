"""Embedding generation utilities."""

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# Global model cache
_model_cache: Optional[SentenceTransformer] = None


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SentenceTransformer:
    """Get or load the embedding model (cached).

    Args:
        model_name: Name of the sentence-transformers model

    Returns:
        SentenceTransformer model
    """
    global _model_cache

    if _model_cache is None:
        _model_cache = SentenceTransformer(model_name)

    return _model_cache


def embed_text(text: str, model: Optional[SentenceTransformer] = None) -> np.ndarray:
    """Generate embedding for a single text.

    Args:
        text: Text to embed
        model: Optional pre-loaded model

    Returns:
        Embedding vector
    """
    if model is None:
        model = get_embedding_model()

    return np.asarray(model.encode(text))


def batch_embed(
    texts: list[str],
    batch_size: int = 32,
    model: Optional[SentenceTransformer] = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        batch_size: Batch size for encoding
        model: Optional pre-loaded model
        show_progress: Whether to show progress bar

    Returns:
        Array of embeddings
    """
    if model is None:
        model = get_embedding_model()

    return np.asarray(
        model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
    )


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings.

    Args:
        emb1: First embedding
        emb2: Second embedding

    Returns:
        Cosine similarity score
    """
    # Handle 1D arrays
    if emb1.ndim == 1:
        emb1 = emb1.reshape(1, -1)
    if emb2.ndim == 1:
        emb2 = emb2.reshape(1, -1)

    return float(sklearn_cosine_similarity(emb1, emb2)[0, 0])
