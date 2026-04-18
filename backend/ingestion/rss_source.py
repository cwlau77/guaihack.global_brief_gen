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


def _parse_feed(outlet: str, url: str) -> list[Article]:
    parsed = feedparser.parse(url)

    articles: list[Article] = []
    for entry in parsed.entries[: settings.max_articles_per_source]:
        title = getattr(entry, "title", "") or ""
        link = getattr(entry, "link", "") or ""
        snippet = getattr(entry, "summary", "") or title
        if not title or not link:
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
    return articles


async def fetch_rss(focus: str) -> list[Article]:
    """Pull articles from curated RSS feeds. Feedparser is blocking, so run in a thread."""
    tasks = [asyncio.to_thread(_parse_feed, outlet, url) for outlet, url in RSS_FEEDS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles: list[Article] = []
    for result in results:
        if isinstance(result, Exception):
            continue
        articles.extend(result)
    return articles
