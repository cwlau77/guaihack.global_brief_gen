import logging
from datetime import datetime
from typing import Optional

import httpx

from backend.config import settings
from backend.models import Article

logger = logging.getLogger("briefing.gdelt")

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
    except httpx.HTTPStatusError as exc:
        logger.warning("GDELT HTTP %s for focus=%r: %s", exc.response.status_code, focus, exc.response.text[:200])
        return []
    except Exception as exc:
        logger.warning("GDELT request failed for focus=%r: %s", focus, exc)
        return []
    finally:
        if owns_client:
            await client.aclose()

    logger.info("GDELT returned %d raw articles for focus=%r", len(payload.get("articles", [])), focus)

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
