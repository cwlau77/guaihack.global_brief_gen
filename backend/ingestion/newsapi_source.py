from datetime import datetime, timezone
from typing import Optional

import httpx

from backend.config import settings
from backend.models import Article

NEWSAPI_EVERYTHING = "https://newsapi.org/v2/everything"


async def fetch_newsapi(focus: str, client: Optional[httpx.AsyncClient] = None) -> list[Article]:
    """Query NewsAPI /everything for the focus phrase across the lookback window."""
    if not settings.newsapi_key:
        return []

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=20.0)

    # NOTE: free-tier NewsAPI delays articles by 24h, so we do not apply a `from` filter here;
    # we rely on sortBy=publishedAt to get the newest articles available to our plan.
    params = {
        "q": focus,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": settings.max_articles_per_source,
        "apiKey": settings.newsapi_key,
    }

    try:
        resp = await client.get(NEWSAPI_EVERYTHING, params=params)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []
    finally:
        if owns_client:
            await client.aclose()

    articles: list[Article] = []
    for item in payload.get("articles", []):
        title = item.get("title") or ""
        snippet = item.get("description") or item.get("content") or ""
        url = item.get("url") or ""
        outlet = (item.get("source") or {}).get("name") or "NewsAPI"
        if not title or not url:
            continue

        published_at = None
        raw_ts = item.get("publishedAt")
        if raw_ts:
            try:
                published_at = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            except ValueError:
                published_at = None

        articles.append(
            Article(
                title=title,
                snippet=snippet,
                url=url,
                source=outlet,
                published_at=published_at,
                country=None,
                raw_source_type="newsapi",
            )
        )

    return articles
