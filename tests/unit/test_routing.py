import pytest

from argus.graph.routing import route_after_gate, route_after_recommendation
from argus.graph.state import initial_state
from argus.guardrails.budgets import ControlBudget
from argus.guardrails.evidence import Claim


def budget(max_calls=10):
    return ControlBudget(max_calls=max_calls, max_tokens=1_000_000, max_cost_usd=100.0)


def claim(contradiction=False):
    return Claim(
        claim="c", claim_type="effective_date", sources=["s1"], primary_source=True,
        source_authority=0.9, recency_score=0.9, corroboration_score=0.9,
        contradiction=contradiction, confidence=0.9,
    )


def state(**overrides):
    s = initial_state("q")
    s.update(confidence_score=0.9, investigation_iterations=1, risk_score=3.0)
    s.update(overrides)
    return s


# ---- research loop -------------------------------------------------------

def test_sufficient_evidence_goes_to_risk_assessment():
    assert route_after_gate(state(), budget()) == "risk_assessment"


def test_insufficient_evidence_continues_research():
    assert route_after_gate(state(confidence_score=0.5), budget()) == "continue_research"


def test_budget_exhausted_stops_research_with_uncertainty():
    s = state(confidence_score=0.5)
    assert route_after_gate(s, budget(max_calls=0)) == "risk_assessment_with_uncertainty"


def test_budget_exhausted_does_not_block_sufficient_evidence():
    assert route_after_gate(state(), budget(max_calls=0)) == "risk_assessment"


def test_max_iterations_stops_research():
    s = state(investigation_iterations=5, confidence_score=0.1)
    assert route_after_gate(s, budget(), max_iterations=5) == "risk_assessment_with_uncertainty"


def test_gate_thresholds_are_configurable():
    s = state(confidence_score=0.8)
    assert route_after_gate(s, budget(), confidence_threshold=0.9) == "continue_research"


# ---- human-in-the-loop ---------------------------------------------------

def test_low_risk_confident_report_is_autonomous():
    assert route_after_recommendation(state()) == "final_evaluation"


@pytest.mark.parametrize("category", ["supplier_communication", "procurement", "legal"])
def test_sensitive_actions_require_human_approval(category):
    s = state(recommended_actions=[{"action": "do it", "category": category}])
    assert route_after_recommendation(s) == "human_approval"


def test_actions_without_sensitive_category_are_ignored():
    s = state(recommended_actions=[{"action": "monitor", "category": "monitoring"}, "plain text"])
    assert route_after_recommendation(s) == "final_evaluation"


def test_unresolved_conflict_goes_to_analyst_review():
    assert route_after_recommendation(state(claims=[claim(contradiction=True)])) == "analyst_review"


def test_low_confidence_goes_to_analyst_review():
    assert route_after_recommendation(state(confidence_score=0.6)) == "analyst_review"


def test_high_risk_with_strong_evidence_notifies_reviewer():
    assert route_after_recommendation(state(risk_score=8.2)) == "notify_reviewer"


def test_risk_exactly_at_threshold_counts_as_high():
    assert route_after_recommendation(state(risk_score=7.0)) == "notify_reviewer"


def test_high_risk_with_weak_evidence_goes_to_analyst_review():
    assert route_after_recommendation(state(risk_score=9.0, confidence_score=0.5)) == "analyst_review"


def test_approval_outranks_analyst_review():
    s = state(
        claims=[claim(contradiction=True)],
        recommended_actions=[{"action": "x", "category": "legal"}],
    )
    assert route_after_recommendation(s) == "human_approval"