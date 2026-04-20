import asyncio
from datetime import datetime, timezone
from time import mktime

import feedparser

from backend.config import settings
from backend.models import Article

RSS_FEEDS: list[tuple[str, str]] = [
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("The Guardian World", "https://www.theguardian.com/world/rss"),
    ("NPR World", "https://feeds.npr.org/1004/rss.xml"),
]


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "by", "with",
    "from", "at", "as", "is", "are", "was", "were", "be", "been", "news", "world",
    "update", "daily", "briefing", "focus", "about", "over", "into", "amid",
}


def _focus_keywords(focus: str) -> list[str]:
    raw = [w.strip(".,;:!?()[]\"'").lower() for w in focus.split()]
    return [w for w in raw if w and w not in _STOPWORDS and len(w) >= 2]


def _parse_feed(outlet: str, url: str, keywords: list[str]) -> list[Article]:
    parsed = feedparser.parse(url)

    articles: list[Article] = []
    for entry in parsed.entries[: settings.max_articles_per_source * 3]:
        title = getattr(entry, "title", "") or ""
        link = getattr(entry, "link", "") or ""
        snippet = getattr(entry, "summary", "") or title
        if not title or not link:
            continue

        # Pre-filter by focus keywords so unfiltered world-news feeds don't flood
        # the pipeline with off-topic articles (this was the "only-Iran" bug).
        if keywords:
            haystack = f"{title} {snippet}".lower()
            if not any(kw in haystack for kw in keywords):
                continue

        published_at = None
        struct_time = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
        if struct_time:
            try:
                published_at = datetime.fromtimestamp(mktime(struct_time), tz=timezone.utc)
            except Exception:
                published_at = None

        articles.append(
            Article(
                title=title,
                snippet=snippet,
                url=link,
                source=outlet,
                published_at=published_at,
                country=None,
                raw_source_type="rss",
            )
        )

        if len(articles) >= settings.max_articles_per_source:
            break
    return articles


async def fetch_rss(focus: str) -> list[Article]:
    """Pull articles from curated RSS feeds, pre-filtered by focus keywords.

    Feedparser is blocking, so each feed is run in its own thread.
    """
    keywords = _focus_keywords(focus)
    tasks = [asyncio.to_thread(_parse_feed, outlet, url, keywords) for outlet, url in RSS_FEEDS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles: list[Article] = []
    for result in results:
        if isinstance(result, Exception):
            continue
        articles.extend(result)
    return articles
