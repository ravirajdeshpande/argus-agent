import json
import re

from argus.agents.verifier import EVIDENCE_FIELDS, index_evidence

MIN_CITATION_COVERAGE = 0.9


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split())


def _corpus(state: dict) -> str:
    """Everything the investigation actually found, as one searchable string."""
    parts: list[str] = []
    for c in state["claims"]:
        parts += [c.claim, c.subject, c.value]
    for d in state["internal_documents"]:
        parts.append(d["text"])
        parts += [v for v in d["metadata"].values() if isinstance(v, str)]
    for field in EVIDENCE_FIELDS:
        if field == "internal_documents":
            continue
        parts += [json.dumps(item.get("result"), default=str) for item in state[field]]
    return _norm(" ".join(parts))


def _is_supported(name: str, corpus: str) -> bool:
    n = _norm(name)
    return bool(n) and f" {n} " in f" {corpus} "


def _split(names: list[str], corpus: str) -> tuple[list[str], list[str]]:
    kept = [n for n in names if _is_supported(n, corpus)]
    removed = [n for n in names if n not in kept]
    return kept, removed


def citation_coverage(source_lists: list[list[str]], known_ids: set[str]) -> float:
    """Share of items that cite at least one source that really exists."""
    if not source_lists:
        return 0.0
    cited = sum(1 for sources in source_lists if any(s in known_ids for s in sources))
    return cited / len(source_lists)


def make_evaluator_node(*, min_citation_coverage: float = MIN_CITATION_COVERAGE):
    """Rule-based final evaluation: citations, grounding, conflicts, completeness."""

    def evaluator(state: dict) -> dict:
        corpus = _corpus(state)
        known_ids = set(index_evidence(state))

        suppliers, removed_suppliers = _split(state["impacted_suppliers"], corpus)
        products, removed_products = _split(state["impacted_products"], corpus)
        repairs = [f"removed unsupported supplier: {n}" for n in removed_suppliers]
        repairs += [f"removed unsupported product: {n}" for n in removed_products]

        coverage = citation_coverage([c.sources for c in state["claims"]], known_ids)
        action_coverage = citation_coverage(
            [a.get("sources", []) for a in state["recommended_actions"]], known_ids
        )
        unresolved = sum(1 for c in state["conflicting_claims"] if not c.get("resolved"))

        missing = []
        if not state["claims"]:
            missing.append("claims")
        if state["risk_assessment"].get("status") != "assessed":
            missing.append("risk assessment")
        if not state["recommended_actions"]:
            missing.append("recommended actions")

        issues = []
        if coverage < min_citation_coverage:
            issues.append(f"citation coverage {coverage:.2f} below {min_citation_coverage:.2f}")
        if missing:
            issues.append("missing: " + ", ".join(missing))

        return {
            "impacted_suppliers": suppliers,
            "impacted_products": products,
            "evaluation": {
                "citation_coverage": coverage,
                "action_citation_coverage": action_coverage,
                "unresolved_conflicts": unresolved,
                "missing_sections": missing,
                "repairs": repairs,
                "issues": issues,
                "passed": not issues,
            },
        }

    return evaluator