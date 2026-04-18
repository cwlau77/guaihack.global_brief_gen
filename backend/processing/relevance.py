import numpy as np

from config import settings
from models import Article

from .embeddings import cosine_similarity, embed_texts


def _keyword_hit(article: Article, keywords: list[str]) -> bool:
    haystack = f"{article.title} {article.snippet}".lower()
    return any(kw in haystack for kw in keywords)


async def filter_by_relevance(
    focus: str,
    articles: list[Article],
    embeddings: np.ndarray,
) -> tuple[list[Article], np.ndarray]:
    """Filter articles by combined keyword + embedding similarity to the focus phrase.

    An article passes if either:
      - a non-trivial keyword from the focus appears in title/snippet, OR
      - cosine similarity(focus, article) >= RELEVANCE_SIMILARITY_THRESHOLD
    """
    if not articles:
        return articles, embeddings

    keywords = [w.lower() for w in focus.split() if len(w) > 3]

    focus_emb = await embed_texts([focus])
    if focus_emb.size == 0 or embeddings.size == 0:
        # If we can't embed, fall back to keyword-only filter.
        kept = [(i, a) for i, a in enumerate(articles) if _keyword_hit(a, keywords)]
        if not kept:
            return articles, embeddings
        idx = [i for i, _ in kept]
        return [a for _, a in kept], embeddings[idx]

    sims = cosine_similarity(focus_emb, embeddings)[0]
    threshold = settings.relevance_similarity_threshold

    kept_indices: list[int] = []
    for i, article in enumerate(articles):
        if _keyword_hit(article, keywords) or float(sims[i]) >= threshold:
            kept_indices.append(i)

    if not kept_indices:
        return articles, embeddings

    return [articles[i] for i in kept_indices], embeddings[kept_indices]
