from datetime import datetime
from typing import Optional

import httpx

from config import settings
from models import Article

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


async def fetch_gdelt(focus: str, client: Optional[httpx.AsyncClient] = None) -> list[Article]:
    """Query GDELT DOC 2.0 ArtList for articles matching the focus phrase."""
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=20.0)

    timespan = f"{max(settings.hours_lookback, 1)}h"
    params = {
        "query": f'"{focus}" sourcelang:english',
        "mode": "ArtList",
        "maxrecords": settings.max_articles_per_source,
        "format": "json",
        "timespan": timespan,
        "sort": "DateDesc",
    }

    try:
        resp = await client.get(GDELT_DOC_API, params=params)
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
        url = item.get("url") or ""
        outlet = item.get("domain") or "GDELT"
        country = item.get("sourcecountry") or None
        if not title or not url:
            continue

        published_at = None
        raw_ts = item.get("seendate")
        if raw_ts:
            try:
                # GDELT seendate format: YYYYMMDDTHHMMSSZ
                published_at = datetime.strptime(raw_ts, "%Y%m%dT%H%M%SZ")
            except ValueError:
                published_at = None

        articles.append(
            Article(
                title=title,
                snippet=title,
                url=url,
                source=outlet,
                published_at=published_at,
                country=country,
                raw_source_type="gdelt",
            )
        )

    return articles
