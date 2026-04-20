"""Shared helpers for broadening focus phrases into practical search/filter terms."""

from __future__ import annotations

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "by", "with",
    "from", "at", "as", "is", "are", "was", "were", "be", "been", "news", "world",
    "update", "daily", "briefing", "focus", "about", "over", "into", "amid",
}

_TERM_ALIASES: dict[str, list[str]] = {
    "climate": [
        "climate",
        "climate change",
        "emissions",
        "carbon",
        "net zero",
        "renewable",
        "renewables",
        "clean energy",
        "energy transition",
        "cop",
    ],
    "energy": [
        "energy",
        "power",
        "electricity",
        "oil",
        "gas",
        "renewable",
        "renewables",
        "grid",
    ],
    "security": [
        "security",
        "defense",
        "military",
        "conflict",
        "ceasefire",
        "troops",
        "sanctions",
        "missile",
    ],
    "trade": [
        "trade",
        "tariff",
        "tariffs",
        "exports",
        "imports",
        "supply chain",
        "customs",
    ],
    "migration": [
        "migration",
        "migrant",
        "migrants",
        "refugee",
        "refugees",
        "asylum",
        "border",
    ],
}


def _normalize_focus_tokens(focus: str) -> list[str]:
    raw = [w.strip(".,;:!?()[]\"'").lower() for w in focus.split()]
    return [w for w in raw if w and w not in _STOPWORDS and len(w) >= 2]


def extract_focus_terms(focus: str, *, include_phrase: bool = True) -> list[str]:
    """Return deduplicated focus terms, including pragmatic aliases for broad topics."""
    normalized_focus = " ".join(_normalize_focus_tokens(focus))
    tokens = _normalize_focus_tokens(focus)

    terms: list[str] = []
    if include_phrase and normalized_focus:
        terms.append(normalized_focus)

    for token in tokens:
        if token not in terms:
            terms.append(token)
        for alias in _TERM_ALIASES.get(token, []):
            if alias not in terms:
                terms.append(alias)

    # Prefer longer/more specific phrases first so substring matching is less noisy.
    return sorted(terms, key=lambda term: (-len(term), term))


def build_boolean_query(focus: str, *, max_terms: int = 6) -> str:
    """Build a compact OR query string for upstream search APIs."""
    terms = extract_focus_terms(focus)[:max_terms]
    if not terms:
        return focus.strip()
    if len(terms) == 1:
        return f'"{terms[0]}"'
    return "(" + " OR ".join(f'"{term}"' for term in terms) + ")"
