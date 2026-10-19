import json

from argus.agents.verifier import _value_key, index_evidence


def grounding_rate(state: dict) -> float:
    """Share of verified claims and recommended actions that cite only real evidence."""
    valid = set(index_evidence(state))
    cited = [list(c.sources) for c in state["verified_claims"]]
    cited += [list(a.get("sources") or []) for a in state["recommended_actions"]]
    if not cited:
        return 1.0
    grounded = sum(1 for sources in cited if sources and all(s in valid for s in sources))
    return round(grounded / len(cited), 2)


def unsupported_entities(state: dict) -> list[str]:
    """Named suppliers or products that appear nowhere in the evidence."""
    evidence = json.dumps(
        [item for source_id, item in index_evidence(state).items()], default=str
    ).lower()
    names = [*state["impacted_suppliers"], *state["impacted_products"]]
    return [n for n in names if n.lower() not in evidence]


def retrieval_scores(state: dict, relevant: set[str]) -> dict:
    """Precision and recall of the internal documents retrieved, by source file name."""
    retrieved = {
        d.get("metadata", {}).get("source")
        for d in state["internal_documents"]
        if d.get("metadata", {}).get("source")
    }
    if not retrieved or not relevant:
        return {"precision": 0.0, "recall": 0.0}
    hits = retrieved & relevant
    return {
        "precision": round(len(hits) / len(retrieved), 2),
        "recall": round(len(hits) / len(relevant), 2),
    }


def contradiction_handled(state: dict, claim_type: str, winning_value: str) -> bool:
    """Resolved to the expected value, or never in conflict and the right value is a verified fact."""
    conflicts = [c for c in state["conflicting_claims"] if c["claim_type"] == claim_type]
    if not conflicts:
        return any(
            c.claim_type == claim_type and _value_key(c.value) == winning_value
            for c in state["verified_claims"]
        )
    rejected = {r for c in conflicts for r in c["rejected_claims"]}
    return all(c["resolved"] and c["winning_value"] == winning_value for c in conflicts) and not any(
        c.claim in rejected for c in state["verified_claims"]
    )


def unresolved_conflicts(state: dict) -> int:
    return sum(1 for c in state["conflicting_claims"] if not c["resolved"])


def run_stats(budget, elapsed_s: float) -> dict:
    return {
        "elapsed_s": round(elapsed_s, 2),
        "calls": getattr(budget, "calls", None),
        "tokens": getattr(budget, "tokens", None),
        "cost_usd": getattr(budget, "cost_usd", None),
    }


def text_checks(report: str, must_include: list[str], must_exclude: list[str]) -> list[str]:
    """Failures for required and forbidden phrases in the final report."""
    lowered = report.lower()
    failures = [f"missing: {t}" for t in must_include if t.lower() not in lowered]
    failures += [f"forbidden: {t}" for t in must_exclude if t.lower() in lowered]
    return failures