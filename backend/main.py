import asyncio
import logging
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.config import settings
from backend.ingestion import fetch_gdelt, fetch_newsapi, fetch_rss
from backend.models import Article, Briefing, BriefingRequest
from backend.processing import deduplicate, embed_texts, filter_by_relevance
from backend.synthesis import enrich_with_historical_context, synthesize_briefing

logger = logging.getLogger("briefing")
logging.basicConfig(level=logging.INFO)
FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "index.html"

app = FastAPI(title="Global Briefing Generator", version="0.1.0")

# Simple in-memory briefing cache: {normalized_focus: (briefing, unix_expiry)}.
# Keeps costs low and responses fast for repeated same-focus requests within the
# configured TTL. Not shared across worker processes — fine for a hackathon demo.
_briefing_cache: dict[str, tuple[Briefing, float]] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND_INDEX)


async def _ingest_all(focus: str) -> list[Article]:
    async with httpx.AsyncClient(timeout=20.0) as client:
        results = await asyncio.gather(
            fetch_newsapi(focus, client=client),
            fetch_gdelt(focus, client=client),
            fetch_rss(focus),
            return_exceptions=True,
        )

    labels = ("newsapi", "gdelt", "rss")
    articles: list[Article] = []
    per_source_counts: dict[str, int] = {}
    for label, r in zip(labels, results):
        if isinstance(r, Exception):
            logger.warning("ingestion source %s failed: %s", label, r)
            per_source_counts[label] = 0
            continue
        per_source_counts[label] = len(r)
        articles.extend(r)
    logger.info("ingestion summary for focus=%r: %s", focus, per_source_counts)
    return articles


@app.post("/briefing", response_model=Briefing)
async def briefing_endpoint(req: BriefingRequest) -> Briefing:
    focus = req.focus.strip()
    cache_key = focus.lower()

    # Check cache first — skip the whole pipeline if we have a fresh briefing
    # for this focus within the configured TTL.
    if settings.cache_ttl_minutes > 0:
        cached = _briefing_cache.get(cache_key)
        if cached and cached[1] > time.time():
            logger.info("cache hit for focus=%r", focus)
            return cached[0]

    # Layer 2: ingest (parallel fan-out)
    articles = await _ingest_all(focus)
    logger.info("ingested %d raw articles for focus=%r", len(articles), focus)
    if not articles:
        missing_keys = []
        if not settings.newsapi_key:
            missing_keys.append("NEWSAPI_KEY")
        hint = (
            f" (missing env vars: {', '.join(missing_keys)})"
            if missing_keys
            else " (all three upstream sources returned zero; check Render logs for HTTP errors from NewsAPI/GDELT, and try a broader focus)"
        )
        raise HTTPException(
            status_code=502,
            detail=f"No articles could be fetched from any source.{hint}",
        )

    # Layer 3: process (embed -> dedupe -> relevance filter)
    texts = [f"{a.title}. {a.snippet}" for a in articles]
    embeddings = await embed_texts(texts)

    if embeddings.size == 0:
        logger.warning("embedding call returned empty; skipping dedup/relevance")
    else:
        articles, embeddings = deduplicate(articles, embeddings)
        articles, embeddings = await filter_by_relevance(focus, articles, embeddings)

    logger.info("post-processing article count: %d", len(articles))
    if not articles:
        raise HTTPException(
            status_code=422,
            detail=f"No recent articles matched the focus '{focus}'. Try a broader phrase or a different topic.",
        )

    # Layer 4: synthesize structured briefing (Haiku by default, per settings.synthesis_model)
    briefing = await synthesize_briefing(focus, articles)

    # Layer 4b: enrich key developments with historical context (parallel Haiku calls,
    # capped by context_enrichment_limit for speed).
    briefing = await enrich_with_historical_context(briefing)

    # Populate cache.
    if settings.cache_ttl_minutes > 0:
        _briefing_cache[cache_key] = (briefing, time.time() + settings.cache_ttl_minutes * 60)

    # Layer 5: return
    return briefing
