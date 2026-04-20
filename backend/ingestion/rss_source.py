import asyncio
import logging
from datetime import datetime, timezone
from time import mktime

import feedparser

from backend.config import settings
from backend.models import Article

logger = logging.getLogger("briefing.rss")

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


def _entry_to_article(outlet: str, entry) -> Article | None:
    title = getattr(entry, "title", "") or ""
    link = getattr(entry, "link", "") or ""
    snippet = getattr(entry, "summary", "") or title
    if not title or not link:
        return None

    published_at = None
    struct_time = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if struct_time:
        try:
            published_at = datetime.fromtimestamp(mktime(struct_time), tz=timezone.utc)
        except Exception:
            published_at = None

    return Article(
        title=title,
        snippet=snippet,
        url=link,
        source=outlet,
        published_at=published_at,
        country=None,
        raw_source_type="rss",
    )


def _parse_feed(outlet: str, url: str, keywords: list[str]) -> list[Article]:
    try:
        parsed = feedparser.parse(url)
    except Exception as exc:
        logger.warning("feed %s (%s) failed to parse: %s", outlet, url, exc)
        return []

    entries = list(parsed.entries[: settings.max_articles_per_source * 3])
    filtered: list[Article] = []
    for entry in entries:
        article = _entry_to_article(outlet, entry)
        if article is None:
            continue
        if keywords:
            haystack = f"{article.title} {article.snippet}".lower()
            if not any(kw in haystack for kw in keywords):
                continue
        filtered.append(article)
        if len(filtered) >= settings.max_articles_per_source:
            break

    # No fallback on empty. An earlier version returned raw top-K when the
    # keyword filter matched zero — that regressed to the "only-Iran" bug,
    # because world-news RSS frontpages are dominated by whatever is currently
    # in the news cycle (right now: Iran/Israel/war). If the focus doesn't
    # match this feed, NewsAPI + GDELT will supply articles instead.
    return filtered


async def fetch_rss(focus: str) -> list[Article]:
    """Pull articles from curated RSS feeds, pre-filtered by focus keywords.

    Feedparser is blocking, so each feed is run in its own thread.
    """
    keywords = _focus_keywords(focus)
    tasks = [asyncio.to_thread(_parse_feed, outlet, url, keywords) for outlet, url in RSS_FEEDS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles: list[Article] = []
    for (outlet, _), result in zip(RSS_FEEDS, results):
        if isinstance(result, Exception):
            logger.warning("feed %s raised: %s", outlet, result)
            continue
        logger.info("feed %s -> %d articles", outlet, len(result))
        articles.extend(result)
    return articles
