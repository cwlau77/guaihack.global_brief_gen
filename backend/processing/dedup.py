import numpy as np

from backend.config import settings
from backend.models import Article


def deduplicate(articles: list[Article], embeddings: np.ndarray) -> tuple[list[Article], np.ndarray]:
    """Greedy dedup: keep the first article, drop any later article whose embedding
    has cosine similarity >= threshold with an already-kept article.

    Returns (kept_articles, kept_embeddings) with aligned indices.
    """
    if not articles or embeddings.size == 0:
        return articles, embeddings

    threshold = settings.dedup_similarity_threshold
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / norms

    kept_indices: list[int] = []
    kept_vecs: list[np.ndarray] = []

    seen_urls: set[str] = set()

    for i, article in enumerate(articles):
        if article.url in seen_urls:
            continue

        if kept_vecs:
            stack = np.stack(kept_vecs)
            sims = stack @ normed[i]
            if float(sims.max()) >= threshold:
                continue

        kept_indices.append(i)
        kept_vecs.append(normed[i])
        seen_urls.add(article.url)

    kept_articles = [articles[i] for i in kept_indices]
    kept_embeddings = embeddings[kept_indices] if kept_indices else np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    return kept_articles, kept_embeddings
