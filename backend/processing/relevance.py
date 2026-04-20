import logging

import numpy as np

from backend.config import settings
from backend.focus_terms import extract_focus_terms
from backend.models import Article

from .embeddings import cosine_similarity, embed_texts

logger = logging.getLogger("briefing.relevance")

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
        logger.info("relevance skip for focus=%r: no articles to filter", focus)
        return articles, embeddings

    keywords = extract_focus_terms(focus)
    logger.info(
        "relevance start for focus=%r: %d articles, %d keywords=%s, embeddings_shape=%s",
        focus,
        len(articles),
        len(keywords),
        keywords,
        tuple(embeddings.shape),
    )

    # Embed the expanded focus (e.g. "climate" -> "climate climate change carbon
    # emissions renewable energy transition...") so the focus vector spans topic
    # synonyms — this dramatically improves recall for single-word focuses and
    # makes bare tokens like "climate" match articles about "carbon" or "net zero".
    focus_text = " ".join(keywords) if keywords else focus
    focus_emb = await embed_texts([focus_text])
    if focus_emb.size == 0 or embeddings.size == 0:
        # Embedding unavailable — keyword-only filter, strict (no fallback).
        kept = [(i, a) for i, a in enumerate(articles) if _keyword_hit(a, keywords)]
        logger.warning(
            "relevance fallback for focus=%r: keyword-only mode kept %d/%d articles",
            focus,
            len(kept),
            len(articles),
        )
        if not kept:
            logger.warning("relevance reject for focus=%r: keyword-only mode matched zero articles", focus)
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
        logger.warning(
            "relevance reject for focus=%r: semantic+keyword filter matched zero articles at threshold %.2f",
            focus,
            threshold,
        )
        return [], np.zeros((0, embeddings.shape[1]), dtype=np.float32)

    logger.info(
        "relevance keep for focus=%r: semantic+keyword filter kept %d/%d articles at threshold %.2f",
        focus,
        len(kept_indices),
        len(articles),
        threshold,
    )
    return [articles[i] for i in kept_indices], embeddings[kept_indices]
