import asyncio

from anthropic import AsyncAnthropic

from backend.config import settings
from backend.models import Briefing, KeyDevelopment

CONTEXT_SYSTEM_PROMPT = """You are a concise historical briefer for international-affairs students.
Given a news headline and summary, write ONE short paragraph (2-4 sentences) of historical and
structural context that helps a student understand why this development matters.

Rules:
- Draw on widely-accepted historical knowledge only; do not invent specific statistics or quotes.
- Focus on WHY: precedents, long-running dynamics, prior related events.
- No preamble, no meta-commentary, no hedging disclaimers. Just the paragraph."""


async def _context_for_development(client: AsyncAnthropic, dev: KeyDevelopment) -> str:
    user_prompt = (
        f"Headline: {dev.headline}\n"
        f"Summary: {dev.summary}\n"
        f"Regions: {', '.join(dev.regions) if dev.regions else 'global'}"
    )
    try:
        response = await client.messages.create(
            model=settings.context_model,
            max_tokens=400,
            system=CONTEXT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
        return "".join(parts).strip()
    except Exception:
        return ""


async def enrich_with_historical_context(briefing: Briefing) -> Briefing:
    """Attach a short Haiku-generated historical-context paragraph to each key development."""
    if not briefing.key_developments or not settings.anthropic_api_key:
        return briefing

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    results = await asyncio.gather(
        *(_context_for_development(client, dev) for dev in briefing.key_developments),
        return_exceptions=True,
    )

    for dev, ctx in zip(briefing.key_developments, results):
        if isinstance(ctx, str) and ctx:
            dev.historical_context = ctx
    return briefing
