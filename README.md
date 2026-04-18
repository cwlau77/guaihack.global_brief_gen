# guaihack.global_brief_gen
An AI-powered briefing generator that synthesizes daily news from multiple countries based on a user-inputted major or area of focus. The tool should aggregate content from diverse international sources, identify patterns and tensions across regions, and produce a concise, structured daily briefing that flags the most important developments.

## Deploy

This repo is prepared for both Render and Railway.

### Render

Render can deploy directly from this GitHub repo using the root `render.yaml`.

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

Required environment variables:

- `ANTHROPIC_API_KEY`
- `NEWSAPI_KEY`
- `HUGGINGFACE_API_KEY`

### Railway

Railway can deploy directly from this repo using `railway.json`.

- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

### Local run

```bash
source backend/.venv/bin/activate
cp backend/.env.example backend/.env
uvicorn backend.main:app --reload
```
