import pytest

from argus.agents.reporter import build_report, make_reporter_node
from argus.graph.state import initial_state
from argus.guardrails.evidence import Claim

TITLES = [
    "Objective", "Risk and confidence", "Potentially affected", "Facts",
    "Reported but not independently verified", "Inference", "Uncertainty and limitations",
    "Conflicts", "Evidence reviewed", "Recommended actions", "Scenarios",
]


def claim(**kw):
    data = dict(
        claim="Restrictions begin September 1", claim_type="effective_date", subject="rule",
        value="September 1", sources=["gov_1"], primary_source=True, source_authority=0.95,
        recency_score=0.9, corroboration_score=0.8, contradiction=False, confidence=0.87,
    )
    data.update(kw)
    return Claim(**data)


def section(report, title):
    for part in report.split("\n## ")[1:]:
        if part.startswith(title):
            return part
    raise AssertionError(f"missing section: {title}")


def state_with(**fields):
    s = initial_state("Will it affect us?")
    verified = claim()
    unverified = claim(
        claim="Supplier says deliveries unaffected", sources=["news_1"], primary_source=False,
        source_authority=0.6, corroboration_score=0.5, confidence=0.5,
    )
    s.update(
        investigation_objective="Assess chip export rules",
        claims=[verified, unverified],
        verified_claims=[verified],
        conflicting_claims=[
            {"claim_type": "effective_date", "subject": "rule", "values": ["immediately", "september 1"],
             "sources": ["gov_1", "news_2"], "resolved": True, "winning_value": "september 1",
             "rejected_claims": ["Effective immediately"]}
        ],
        news_articles=[{"source_id": "news_1"}, {"source_id": "news_2"}],
        government_sources=[{"source_id": "gov_1"}],
        internal_documents=[{"source_id": "internal_1"}, {"source_id": "internal_2"}, {"source_id": "internal_3"}],
        impacted_suppliers=["ABC Semiconductor"],
        impacted_products=["Product Alpha"],
        risk_score=8.2,
        confidence_score=0.87,
        evidence_quality_score=0.72,
        risk_assessment={
            "status": "assessed", "risk_level": "HIGH", "risk_score": 8.2,
            "expected_impact": "Supply interruption within 45-60 days",
            "rationale": "Supplier ABC provides category X components",
            "uncertainty": ["1 claim(s) in unresolved conflict"],
            "scenarios": [
                {"name": "Best case", "description": "no interruption", "probability": 0.2},
                {"name": "Most likely", "description": "4-6 week delay", "probability": 0.55},
                {"name": "Worst case", "description": "supplier unavailable", "probability": 0.25},
            ],
        },
        recommended_actions=[
            {"priority": 1, "action": "Contact Supplier ABC", "category": "supplier_communication",
             "owner": "Procurement lead", "urgency": "immediate", "rationale": "Supplies category X",
             "sources": ["gov_1"], "requires_approval": True},
            {"priority": 2, "action": "Continue monitoring", "category": "monitoring",
             "owner": "Risk analyst", "urgency": "low", "rationale": "Rules may change",
             "sources": [], "requires_approval": False},
        ],
        evaluation={"passed": True, "issues": [], "repairs": ["removed unsupported supplier: Phantom Corp"]},
    )
    s.update(fields)
    return s


def test_report_header_risk_and_confidence():
    r = build_report(state_with())
    assert r.startswith("# GLOBAL RISK INTELLIGENCE REPORT")
    assert "Assess chip export rules" in section(r, "Objective")
    risk = section(r, "Risk and confidence")
    assert "Risk level: HIGH - 8.2 / 10" in risk
    assert "Confidence: 87%" in risk


def test_sections_appear_in_order():
    r = build_report(state_with())
    positions = [r.index(f"## {t}") for t in TITLES]
    assert positions == sorted(positions)


def test_affected_suppliers_and_products():
    affected = section(build_report(state_with()), "Potentially affected")
    assert "ABC Semiconductor" in affected
    assert "Product Alpha" in affected


def test_nothing_identified_is_stated():
    affected = section(build_report(state_with(impacted_suppliers=[], impacted_products=[])), "Potentially affected")
    assert affected.count("none identified") == 2


def test_facts_hold_only_verified_claims_with_sources():
    r = build_report(state_with())
    facts = section(r, "Facts")
    assert "Restrictions begin September 1" in facts
    assert "[gov_1]" in facts
    assert "primary source" in facts
    assert "Supplier says deliveries unaffected" not in facts
    assert "Supplier says deliveries unaffected" in section(r, "Reported but not")


def test_inference_is_separate_from_facts():
    r = build_report(state_with())
    inference = section(r, "Inference")
    assert "Supply interruption within 45-60 days" in inference
    assert "Supplier ABC provides category X components" in inference
    assert "Supplier ABC provides category X components" not in section(r, "Facts")


def test_uncertainty_lists_reasons_issues_and_repairs():
    state = state_with(
        evaluation={"passed": False, "issues": ["citation coverage 0.50 below 0.90"],
                    "repairs": ["removed unsupported supplier: Phantom Corp"]}
    )
    uncertainty = section(build_report(state), "Uncertainty and limitations")
    assert "1 claim(s) in unresolved conflict" in uncertainty
    assert "citation coverage 0.50 below 0.90" in uncertainty
    assert "removed unsupported supplier: Phantom Corp" in uncertainty


def test_resolved_and_unresolved_conflicts_are_both_surfaced():
    unresolved = {"claim_type": "scope", "subject": "inventory", "values": ["30 days", "45 days"],
                  "sources": ["internal_1", "news_1"], "resolved": False, "winning_value": None,
                  "rejected_claims": []}
    state = state_with()
    state["conflicting_claims"] = [*state["conflicting_claims"], unresolved]
    conflicts = section(build_report(state), "Conflicts")
    assert "Resolved: rule (effective_date)" in conflicts
    assert "september 1" in conflicts
    assert "Effective immediately" in conflicts
    assert "UNRESOLVED: inventory (scope)" in conflicts
    assert "30 days, 45 days" in conflicts


def test_evidence_counts():
    evidence = section(build_report(state_with()), "Evidence reviewed")
    assert "2 news items" in evidence
    assert "1 government sources" in evidence
    assert "0 market signals" in evidence
    assert "3 internal documents" in evidence


def test_actions_show_owner_urgency_approval_rationale_and_sources():
    actions = section(build_report(state_with()), "Recommended actions")
    assert "1. Contact Supplier ABC" in actions
    assert "owner: Procurement lead" in actions
    assert "urgency: immediate" in actions
    assert actions.count("REQUIRES HUMAN APPROVAL") == 1
    assert "Rationale: Supplies category X" in actions
    assert "(sources: gov_1)" in actions
    assert "2. Continue monitoring" in actions


def test_scenarios_show_percentages():
    scenarios = section(build_report(state_with()), "Scenarios")
    assert "Most likely: 4-6 week delay - 55%" in scenarios
    assert "Best case: no interruption - 20%" in scenarios


def test_unavailable_assessment_is_reported_honestly():
    state = state_with(risk_assessment={"status": "unavailable", "uncertainty": ["no primary source found"]},
                       risk_score=0.0, confidence_score=0.0)
    r = build_report(state)
    risk = section(r, "Risk and confidence")
    assert "Risk level: UNDETERMINED" in risk
    assert "could not be completed" in risk
    assert "no primary source found" in section(r, "Uncertainty and limitations")
    assert section(r, "Scenarios").strip().endswith("None.")


def test_empty_state_still_builds_a_report():
    r = build_report(initial_state("q"))
    assert "UNDETERMINED" in r
    assert section(r, "Facts").strip().endswith("None.")
    assert section(r, "Recommended actions").strip().endswith("None.")


def test_reporter_node_returns_final_report():
    state = state_with()
    assert make_reporter_node()(state) == {"final_report": build_report(state)}