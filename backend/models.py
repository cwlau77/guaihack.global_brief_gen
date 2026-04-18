from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Article(BaseModel):
    title: str
    snippet: str
    url: str
    source: str
    published_at: Optional[datetime] = None
    country: Optional[str] = None
    raw_source_type: Literal["newsapi", "gdelt", "rss"]


class BriefingRequest(BaseModel):
    focus: str = Field(..., min_length=2, max_length=200, description="Area of focus, e.g. 'South Asian security'")


class SourceCitation(BaseModel):
    outlet: str
    url: str
    published_at: Optional[datetime] = None


class KeyDevelopment(BaseModel):
    headline: str
    summary: str
    regions: list[str]
    sources: list[SourceCitation]
    historical_context: Optional[str] = None


class Tension(BaseModel):
    description: str
    actors: list[str]
    sources: list[SourceCitation]


class Contradiction(BaseModel):
    topic: str
    account_a: str
    account_b: str
    sources_a: list[SourceCitation]
    sources_b: list[SourceCitation]


class PriorityAlert(BaseModel):
    severity: Literal["critical", "high", "elevated"]
    headline: str
    rationale: str
    sources: list[SourceCitation]


class RecommendedReading(BaseModel):
    title: str
    outlet: str
    url: str
    why: str


class Briefing(BaseModel):
    focus: str
    generated_at: datetime
    key_developments: list[KeyDevelopment]
    emerging_tensions: list[Tension]
    contradictions: list[Contradiction]
    priority_alerts: list[PriorityAlert]
    recommended_readings: list[RecommendedReading]
    article_count: int
    source_breakdown: dict[str, int]
