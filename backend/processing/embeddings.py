from typing import Optional

import httpx
import numpy as np

from config import settings

HF_FEATURE_EXTRACTION_URL = (
    f"https://api-inference.huggingface.co/pipeline/feature-extraction/{settings.embedding_model}"
)


async def embed_texts(texts: list[str], client: Optional[httpx.AsyncClient] = None) -> np.ndarray:
    """Call the HuggingFace Inference API to embed a batch of texts.

    Returns a 2D numpy array of shape (len(texts), hidden_size). Returns an empty array on failure.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=30.0)

    headers = {"Authorization": f"Bearer {settings.huggingface_api_key}"}
    body = {"inputs": texts, "options": {"wait_for_model": True}}

    try:
        resp = await client.post(HF_FEATURE_EXTRACTION_URL, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return np.zeros((0, 0), dtype=np.float32)
    finally:
        if owns_client:
            await client.aclose()

    arr = np.asarray(data, dtype=np.float32)
    # sentence-transformers pipelines return (batch, hidden) directly.
    # If token-level (batch, tokens, hidden) is returned, mean-pool over tokens.
    if arr.ndim == 3:
        arr = arr.mean(axis=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = vectors / norms
    return normed @ normed.T


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0] if a.ndim > 1 else 0, b.shape[0] if b.ndim > 1 else 0), dtype=np.float32)
    a_norm = a / np.where(np.linalg.norm(a, axis=1, keepdims=True) == 0, 1.0, np.linalg.norm(a, axis=1, keepdims=True))
    b_norm = b / np.where(np.linalg.norm(b, axis=1, keepdims=True) == 0, 1.0, np.linalg.norm(b, axis=1, keepdims=True))
    return a_norm @ b_norm.T
