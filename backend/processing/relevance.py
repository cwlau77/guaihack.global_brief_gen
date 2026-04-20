import numpy as np

from backend.config import settings
from backend.models import Article

from .embeddings import cosine_similarity, embed_texts

# Common stopwords we never want to treat as relevance signals.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "by", "with",
    "from", "at", "as", "is", "are", "was", "were", "be", "been", "news", "world",
    "update", "daily", "briefing", "focus", "about", "over", "into", "amid",
}


def _extract_keywords(focus: str) -> list[str]:
    """Keep short-but-meaningful tokens (EU, UK, US, AI, oil) — drop stopwords only."""
    raw = [w.strip(".,;:!?()[]\"'").lower() for w in focus.split()]
    return [w for w in raw if w and w not in _STOPWORDS and len(w) >= 2]


def _keyword_hit(article: Article, keywords: list[str]) -> bool:
    if not keywords:
        return False
    haystack = f"{article.title} {article.snippet}".lower()
    return any(kw in haystack for kw in keywords)


async def filter_by_relevance(
    focus: str,
    articles: list[Article],
    embeddings: np.ndarray,
) -> tuple[list[Article], np.ndarray]:
    """Filter articles by combined keyword + embedding similarity to the focus phrase.

    An article passes if EITHER:
      - a non-trivial keyword from the focus appears in title/snippet, OR
      - cosine similarity(focus, article) >= RELEVANCE_SIMILARITY_THRESHOLD

    Returns (possibly empty) filtered list. No permissive fallback — callers must
    handle the empty case (main.py returns 422). Returning everything when filters
    fail to match defeats the purpose of the focus filter and causes off-topic
    articles to leak into the synthesis prompt.
    """
    if not articles:
        return articles, embeddings

    keywords = _extract_keywords(focus)

    focus_emb = await embed_texts([focus])
    if focus_emb.size == 0 or embeddings.size == 0:
        # Embedding unavailable — keyword-only filter, strict (no fallback).
        kept = [(i, a) for i, a in enumerate(articles) if _keyword_hit(a, keywords)]
        if not kept:
            return [], np.zeros((0, embeddings.shape[1] if embeddings.ndim == 2 else 0), dtype=np.float32)
        idx = [i for i, _ in kept]
        return [a for _, a in kept], embeddings[idx] if embeddings.size else embeddings

    sims = cosine_similarity(focus_emb, embeddings)[0]
    threshold = settings.relevance_similarity_threshold

    kept_indices: list[int] = []
    for i, article in enumerate(articles):
        if _keyword_hit(article, keywords) or float(sims[i]) >= threshold:
            kept_indices.append(i)

    if not kept_indices:
        return [], np.zeros((0, embeddings.shape[1]), dtype=np.float32)

    return [articles[i] for i in kept_indices], embeddings[kept_indices]
