import json
import re
from collections import Counter
from datetime import datetime, timezone

from anthropic import AsyncAnthropic

from backend.config import settings
from backend.models import (
    Article,
    Briefing,
    Contradiction,
    KeyDevelopment,
    PriorityAlert,
    RecommendedReading,
    SourceCitation,
    Tension,
)

SYSTEM_PROMPT = """You are a senior international-affairs analyst producing a daily intelligence briefing.
You read raw news wire items from multiple countries and synthesize them into a structured JSON briefing.

Analytic standards:
- Every claim must be grounded in the supplied articles; never invent facts.
- Prefer primary-source phrasing; attribute every point to specific outlets.
- Actively search across articles for cross-source CONTRADICTIONS where outlets disagree on facts,
  framing, or causation (e.g. "Reuters reports X; state media reports Y"). Surface these explicitly.
- Identify EMERGING TENSIONS between state/non-state actors, even if no direct incident has occurred yet.
- Priority alerts are reserved for developments with clear escalation potential (military, humanitarian,
  financial-stability, election-integrity). Be selective.
- The student user has limited time: be concise, analytical, and forward-looking.

Return ONLY valid JSON matching the schema you are given. No prose outside the JSON."""


SCHEMA_INSTRUCTIONS = """Return a JSON object with this exact shape:

{
  "key_developments": [
    {
      "headline": "string, <= 15 words",
      "summary": "string, 2-4 sentences of analytical summary",
      "regions": ["ISO-like region/country names"],
      "sources": [{"outlet": "string", "url": "string", "published_at": "ISO8601 or null"}]
    }
  ],
  "emerging_tensions": [
    {
      "description": "string, 1-3 sentences describing the tension and its trajectory",
      "actors": ["named states, groups, or institutions"],
      "sources": [{"outlet": "string", "url": "string", "published_at": "ISO8601 or null"}]
    }
  ],
  "contradictions": [
    {
      "topic": "string naming what outlets disagree about",
      "account_a": "one side's factual/framing claim",
      "account_b": "the other side's factual/framing claim",
      "sources_a": [{"outlet": "string", "url": "string", "published_at": "ISO8601 or null"}],
      "sources_b": [{"outlet": "string", "url": "string", "published_at": "ISO8601 or null"}]
    }
  ],
  "priority_alerts": [
    {
      "severity": "critical | high | elevated",
      "headline": "string, <= 15 words",
      "rationale": "1-2 sentences on why this warrants priority attention",
      "sources": [{"outlet": "string", "url": "string", "published_at": "ISO8601 or null"}]
    }
  ],
  "recommended_readings": [
    {"title": "string", "outlet": "string", "url": "string", "why": "1 sentence on why this is worth reading"}
  ]
}

Constraints:
- 3-6 key_developments, 0-4 emerging_tensions, 0-4 contradictions, 0-3 priority_alerts, 3-5 recommended_readings.
- Every URL MUST be one that appears in the provided articles.
- Omit sections (empty array) if evidence is insufficient rather than fabricating."""


def _format_articles(articles: list[Article]) -> str:
    lines: list[str] = []
    for i, a in enumerate(articles, start=1):
        ts = a.published_at.isoformat() if a.published_at else "unknown"
        country = f" [{a.country}]" if a.country else ""
        lines.append(
            f"[{i}] ({a.source}{country}, {ts}) {a.title}\n    {a.snippet}\n    URL: {a.url}"
        )
    return "\n\n".join(lines)


def _build_user_prompt(focus: str, articles: list[Article]) -> str:
    return (
        f"AREA OF FOCUS: {focus}\n\n"
        f"ARTICLES ({len(articles)}):\n\n"
        f"{_format_articles(articles)}\n\n"
        f"{SCHEMA_INSTRUCTIONS}"
    )


def _source_breakdown(articles: list[Article]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for a in articles:
        counts[a.raw_source_type] = counts.get(a.raw_source_type, 0) + 1
    return counts


def _parse_citations(raw: list[dict]) -> list[SourceCitation]:
    out: list[SourceCitation] = []
    for item in raw or []:
        try:
            published_at = None
            raw_ts = item.get("published_at")
            if raw_ts and isinstance(raw_ts, str):
                try:
                    published_at = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                except ValueError:
                    published_at = None
            out.append(
                SourceCitation(
                    outlet=item.get("outlet") or "unknown",
                    url=item.get("url") or "",
                    published_at=published_at,
                )
            )
        except Exception:
            continue
    return out


def _fallback_citations(articles: list[Article], limit: int = 3) -> list[SourceCitation]:
    return [
        SourceCitation(outlet=article.source, url=article.url, published_at=article.published_at)
        for article in articles[:limit]
    ]


def _infer_regions(article: Article, focus: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", f"{focus} {article.title}")
    stopwords = {
        "the", "and", "for", "with", "from", "into", "amid", "after", "before", "global",
        "policy", "briefing", "daily", "major", "world", "update", "over", "news"
    }
    regions: list[str] = []
    for token in tokens:
        if token.lower() in stopwords:
            continue
        if token[0].isupper() and token not in regions:
            regions.append(token)
    if article.country and article.country not in regions:
        regions.append(article.country)
    return regions[:4] or [focus]


def _fallback_briefing(focus: str, articles: list[Article]) -> Briefing:
    sorted_articles = sorted(
        articles,
        key=lambda article: article.published_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    article_count = len(sorted_articles)
    top_articles = sorted_articles[: min(4, article_count)]

    key_developments = [
        KeyDevelopment(
            headline=article.title[:120],
            summary=(
                f"{article.snippet or article.title} "
                f"This item is included because it appears prominently across the latest reporting tied to {focus}."
            ).strip(),
            regions=_infer_regions(article, focus),
            sources=_fallback_citations([article]),
        )
        for article in top_articles
    ]

    tension_keywords = ("conflict", "tension", "sanction", "military", "strike", "protest", "dispute", "ceasefire")
    tension_articles = [
        article for article in sorted_articles
        if any(keyword in f"{article.title} {article.snippet}".lower() for keyword in tension_keywords)
    ][:3]
    emerging_tensions = [
        Tension(
            description=(
                f"{article.title}. {article.snippet or 'Recent coverage suggests a potentially fast-moving situation.'}"
            ).strip(),
            actors=_infer_regions(article, focus),
            sources=_fallback_citations([article]),
        )
        for article in tension_articles
    ]

    alert_keywords = ("war", "attack", "crisis", "sanction", "emergency", "tariff", "election", "ceasefire")
    priority_alerts = []
    for article in sorted_articles:
        if not any(keyword in f"{article.title} {article.snippet}".lower() for keyword in alert_keywords):
            continue
        severity = "critical" if any(word in article.title.lower() for word in ("war", "attack", "emergency")) else "high"
        priority_alerts.append(
            PriorityAlert(
                severity=severity,
                headline=article.title[:120],
                rationale=(
                    f"This development may alter the near-term outlook for {focus} and warrants closer monitoring."
                ),
                sources=_fallback_citations([article]),
            )
        )
        if len(priority_alerts) == 3:
            break

    recommended_readings = [
        RecommendedReading(
            title=article.title,
            outlet=article.source,
            url=article.url,
            why=f"Use this piece to verify the reporting directly from {article.source}.",
        )
        for article in sorted_articles[: min(5, article_count)]
    ]

    if not recommended_readings:
        recommended_readings = [
            RecommendedReading(
                title=article.title,
                outlet=article.source,
                url=article.url,
                why="Use this piece as a starting point for source verification.",
            )
            for article in top_articles
        ]

    return Briefing(
        focus=focus,
        generated_at=datetime.now(timezone.utc),
        key_developments=key_developments,
        emerging_tensions=emerging_tensions,
        contradictions=[],
        priority_alerts=priority_alerts,
        recommended_readings=recommended_readings,
        article_count=article_count,
        source_breakdown=dict(Counter(article.raw_source_type for article in articles)),
    )


async def synthesize_briefing(focus: str, articles: list[Article]) -> Briefing:
    """Run the main Claude Sonnet synthesis pass and return a Briefing."""
    if not settings.anthropic_api_key:
        return _fallback_briefing(focus, articles)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    response = await client.messages.create(
        model=settings.synthesis_model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(focus, articles)}],
    )

    text_parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
    raw_text = "".join(text_parts).strip()

    # Tolerate the model wrapping JSON in ```json fences or stray prose.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1:
        raw_text = raw_text[start : end + 1]

    data = json.loads(raw_text)

    key_developments = [
        KeyDevelopment(
            headline=kd["headline"],
            summary=kd["summary"],
            regions=kd.get("regions", []),
            sources=_parse_citations(kd.get("sources", [])),
        )
        for kd in data.get("key_developments", [])
    ]
    tensions = [
        Tension(
            description=t["description"],
            actors=t.get("actors", []),
            sources=_parse_citations(t.get("sources", [])),
        )
        for t in data.get("emerging_tensions", [])
    ]
    contradictions = [
        Contradiction(
            topic=c["topic"],
            account_a=c["account_a"],
            account_b=c["account_b"],
            sources_a=_parse_citations(c.get("sources_a", [])),
            sources_b=_parse_citations(c.get("sources_b", [])),
        )
        for c in data.get("contradictions", [])
    ]
    priority_alerts = [
        PriorityAlert(
            severity=p["severity"],
            headline=p["headline"],
            rationale=p["rationale"],
            sources=_parse_citations(p.get("sources", [])),
        )
        for p in data.get("priority_alerts", [])
    ]
    recommended = [
        RecommendedReading(title=r["title"], outlet=r["outlet"], url=r["url"], why=r["why"])
        for r in data.get("recommended_readings", [])
    ]

    return Briefing(
        focus=focus,
        generated_at=datetime.now(timezone.utc),
        key_developments=key_developments,
        emerging_tensions=tensions,
        contradictions=contradictions,
        priority_alerts=priority_alerts,
        recommended_readings=recommended,
        article_count=len(articles),
        source_breakdown=_source_breakdown(articles),
    )
