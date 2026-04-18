import asyncio
import logging

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ingestion import fetch_gdelt, fetch_newsapi, fetch_rss
from models import Article, Briefing, BriefingRequest
from processing import deduplicate, embed_texts, filter_by_relevance
from synthesis import enrich_with_historical_context, synthesize_briefing

logger = logging.getLogger("briefing")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Global Briefing Generator", version="0.1.0")

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


async def _ingest_all(focus: str) -> list[Article]:
    async with httpx.AsyncClient(timeout=20.0) as client:
        results = await asyncio.gather(
            fetch_newsapi(focus, client=client),
            fetch_gdelt(focus, client=client),
            fetch_rss(focus),
            return_exceptions=True,
        )

    articles: list[Article] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning("ingestion source failed: %s", r)
            continue
        articles.extend(r)
    return articles


@app.post("/briefing", response_model=Briefing)
async def briefing_endpoint(req: BriefingRequest) -> Briefing:
    focus = req.focus.strip()

    # Layer 2: ingest
    articles = await _ingest_all(focus)
    logger.info("ingested %d raw articles for focus=%r", len(articles), focus)
    if not articles:
        raise HTTPException(status_code=502, detail="No articles could be fetched from any source.")

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
        raise HTTPException(status_code=422, detail="No articles matched the focus area after filtering.")

    # Layer 4: synthesize structured briefing via Claude Sonnet
    briefing = await synthesize_briefing(focus, articles)

    # Layer 4b: enrich key developments with historical context via Claude Haiku
    briefing = await enrich_with_historical_context(briefing)

    # Layer 5: return
    return briefing
