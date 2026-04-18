# Backend — Global Briefing Generator

FastAPI service that turns a single focus phrase (e.g. `"South Asian security"`) into a structured
daily intelligence briefing synthesized from NewsAPI, GDELT, and curated RSS feeds.

## Architecture (5 layers)

1. **Input** — `POST /briefing` with `{"focus": "..."}`.
2. **Ingestion** — `ingestion/` fans out to NewsAPI, GDELT DOC 2.0, and RSS (BBC World, Al Jazeera, Reuters) in parallel.
3. **Processing** — `processing/` embeds article text via HuggingFace Inference API
   (`sentence-transformers/all-MiniLM-L6-v2`), deduplicates via cosine similarity, and filters by
   keyword + semantic relevance to the focus phrase.
4. **Synthesis** — `synthesis/briefing.py` calls Claude Sonnet 4.6 once with all filtered articles and
   returns structured JSON: key developments, emerging tensions, cross-source contradictions,
   priority alerts, and recommended readings. `synthesis/context.py` then fans out per key
   development to Claude Haiku 4.5 for short historical-context paragraphs.
5. **Output** — a `Briefing` object (see `models.py`) returned as JSON to the frontend.

## Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in real keys
```

Required keys in `.env`:

- `ANTHROPIC_API_KEY` — Claude API
- `NEWSAPI_KEY` — https://newsapi.org
- `HUGGINGFACE_API_KEY` — https://huggingface.co/settings/tokens

## Run

```bash
uvicorn main:app --reload --port 8000
```

Then:

```bash
curl -s -X POST http://localhost:8000/briefing \
  -H "Content-Type: application/json" \
  -d '{"focus":"South Asian security"}' | jq
```

Health check: `GET /health`.

## Tuning

All knobs live in `.env` / `config.py`:

- `MAX_ARTICLES_PER_SOURCE` (default 25)
- `DEDUP_SIMILARITY_THRESHOLD` (default 0.85 cosine)
- `RELEVANCE_SIMILARITY_THRESHOLD` (default 0.35 cosine vs. focus embedding)
- `HOURS_LOOKBACK` (default 24)

## Response shape

See `models.Briefing` — the frontend should render:

- `key_developments[]` — headline + analytical summary + sources + `historical_context`
- `emerging_tensions[]`
- `contradictions[]` — two competing accounts with separate source lists
- `priority_alerts[]` — severity `critical | high | elevated` (render red for the first two)
- `recommended_readings[]`
- `article_count`, `source_breakdown` — telemetry for the footer
