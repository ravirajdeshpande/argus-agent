from argus.guardrails.access_control import Confidentiality, filter_allowed
from argus.guardrails.injection import sanitize_retrieved


def _matches(metadata: dict, filters: dict) -> bool:
    return all(
        str(metadata.get(key, "")).lower() == str(value).lower()
        for key, value in filters.items()
    )


def retrieve(
    store,
    query: str,
    clearance: Confidentiality,
    *,
    k: int = 5,
    oversample: int = 4,
    filters: dict | None = None,
) -> list[dict]:
    """Search, drop what the user may not see, apply filters, neutralise injections."""
    candidates = store.search(query, k=k * oversample)
    allowed = filter_allowed(candidates, clearance)
    if filters:
        allowed = [c for c in allowed if _matches(c["metadata"], filters)]

    results = []
    for c in allowed[:k]:
        cleaned = sanitize_retrieved(c["text"])
        results.append(
            {
                **c,
                "text": cleaned.text,
                "metadata": {**c["metadata"], "injection_flagged": cleaned.flagged},
            }
        )
    return results