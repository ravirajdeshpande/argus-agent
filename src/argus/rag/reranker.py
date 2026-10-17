from datetime import date


def _parse_date(value) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def rerank(
    hits: list[dict],
    as_of: date,
    *,
    expired_penalty: float = 0.3,
    recency_weight: float = 0.2,
    half_life_days: float = 365,
    drop_expired: bool = False,
) -> list[dict]:
    """Re-score retrieved chunks so expired or old ones cannot dominate."""
    ranked = []
    for h in hits:
        meta = h["metadata"]
        expires = _parse_date(meta.get("expires"))
        expired = expires is not None and expires < as_of
        if expired and drop_expired:
            continue

        final = h["score"]
        if expired:
            final *= expired_penalty

        doc_date = _parse_date(meta.get("date"))
        if doc_date is not None:
            age_days = max((as_of - doc_date).days, 0)
            freshness = 0.5 ** (age_days / half_life_days)
            final *= (1 - recency_weight) + recency_weight * freshness

        ranked.append(
            {**h, "final_score": final, "metadata": {**meta, "expired": expired}}
        )
    return sorted(ranked, key=lambda h: h["final_score"], reverse=True)