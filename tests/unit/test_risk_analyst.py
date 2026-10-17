import json

import pytest

from argus.agents.risk_analyst import (
    compute_risk_score,
    make_risk_analyst_node,
    parse_assessment,
    risk_level,
    uncertainty_reasons,
)
from argus.config.models import Tier
from argus.graph.routing import route_after_recommendation
from argus.graph.state import initial_state
from argus.guardrails.budgets import ControlBudget
from argus.guardrails.evidence import Claim
from argus.models.cascade import ModelResponse

GOOD = {
    "impact": 9,
    "probability": 7,
    "urgency": 8,
    "expected_impact": "Supply interruption within 45-60 days",
    "rationale": "Supplier ABC provides category X components",
    "impacted_products": ["Product Alpha"],
    "impacted_suppliers": ["ABC Semiconductor"],
    "scenarios": [
        {"name": "Best case", "description": "no interruption", "probability": 0.2},
        {"name": "Most likely", "description": "4-6 week delay", "probability": 0.55},
        {"name": "Worst case", "description": "supplier unavailable", "probability": 0.25},
    ],
}


def claim(**kw):
    data = dict(
        claim="Restrictions begin September 1", claim_type="effective_date", subject="rule",
        value="September 1", sources=["gov_1"], primary_source=True, source_authority=0.95,
        recency_score=0.9, corroboration_score=0.8, contradiction=False, confidence=0.87,
    )
    data.update(kw)
    return Claim(**data)


def state_with(**fields):
    s = initial_state("Will it affect us?")
    s.update(claims=[claim()], verified_claims=[claim()], confidence_score=0.87,
             investigation_iterations=2, investigation_objective="Assess chip rules")
    s.update(fields)
    return s


def budget():
    return ControlBudget(max_calls=10, max_tokens=1_000_000, max_cost_usd=100.0)


# ---- scoring -------------------------------------------------------------

def test_risk_score_is_weighted_average():
    assert compute_risk_score(9, 7, 8) == pytest.approx(8.05)


def test_custom_weights_are_normalised():
    weights = {"impact": 2, "probability": 1, "urgency": 1}
    assert compute_risk_score(10, 0, 0, weights) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "weights",
    [
        {"impact": -1, "probability": 1, "urgency": 1},
        {"impact": 0, "probability": 0, "urgency": 0},
    ],
)
def test_bad_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        compute_risk_score(5, 5, 5, weights)


@pytest.mark.parametrize(
    "score, level",
    [(10, "HIGH"), (7.0, "HIGH"), (6.99, "MEDIUM"), (4.0, "MEDIUM"), (3.99, "LOW"), (0, "LOW")],
)
def test_risk_level_bands(score, level):
    assert risk_level(score) == level


# ---- parsing -------------------------------------------------------------

def test_parse_assessment_reads_all_fields():
    a = parse_assessment(json.dumps(GOOD))
    assert (a["impact"], a["probability"], a["urgency"]) == (9.0, 7.0, 8.0)
    assert a["expected_impact"] == "Supply interruption within 45-60 days"
    assert a["impacted_products"] == ["Product Alpha"]
    assert a["impacted_suppliers"] == ["ABC Semiconductor"]
    assert [s["probability"] for s in a["scenarios"]] == [0.2, 0.55, 0.25]


def test_scores_are_clamped_to_0_10():
    a = parse_assessment(json.dumps({"impact": 15, "probability": -3, "urgency": "7"}))
    assert (a["impact"], a["probability"], a["urgency"]) == (10.0, 0.0, 7.0)


@pytest.mark.parametrize(
    "data",
    [
        {"impact": 9, "probability": "high", "urgency": 8},
        {"impact": 9, "urgency": 8},
        {"impact": True, "probability": 5, "urgency": 5},
    ],
)
def test_unusable_dimensions_give_none(data):
    assert parse_assessment(json.dumps(data)) is None


def test_garbage_reply_gives_none():
    assert parse_assessment("no idea") is None


def test_scenario_percentages_are_normalised():
    data = {**GOOD, "scenarios": [
        {"name": "a", "probability": 20}, {"name": "b", "probability": 55}, {"name": "c", "probability": 25},
    ]}
    a = parse_assessment(json.dumps(data))
    assert [s["probability"] for s in a["scenarios"]] == [0.2, 0.55, 0.25]


def test_invalid_scenarios_are_dropped():
    data = {**GOOD, "scenarios": [
        {"name": "", "probability": 0.5}, "junk", {"name": "ok", "probability": "x"},
        {"name": "keep", "probability": 1},
    ]}
    a = parse_assessment(json.dumps(data))
    assert [(s["name"], s["probability"]) for s in a["scenarios"]] == [("keep", 1.0)]


def test_product_and_supplier_lists_are_cleaned():
    data = {**GOOD, "impacted_products": ["A", "A", " ", 5], "impacted_suppliers": "ABC"}
    a = parse_assessment(json.dumps(data))
    assert a["impacted_products"] == ["A", "5"]
    assert a["impacted_suppliers"] == []


# ---- uncertainty ---------------------------------------------------------

def test_confident_clean_state_has_no_uncertainty():
    assert uncertainty_reasons(state_with()) == []


@pytest.mark.parametrize(
    "fields, expected",
    [
        (dict(confidence_score=0.6), "0.60"),
        (dict(claims=[claim(contradiction=True)]), "unresolved"),
        (dict(missing_information=["inventory data"]), "inventory data"),
        (dict(investigation_iterations=5), "iteration limit"),
        (dict(claims=[claim(primary_source=False)]), "no primary source"),
    ],
)
def test_uncertainty_reasons(fields, expected):
    reasons = uncertainty_reasons(state_with(**fields))
    assert any(expected in r for r in reasons)


# ---- node ----------------------------------------------------------------

def caller_returning(text):
    tiers, prompts = [], []

    def call(tier, prompt):
        tiers.append(tier)
        prompts.append(prompt)
        return ModelResponse(
            text=text, confidence=0.9, model="m",
            input_tokens=10, output_tokens=5, latency_ms=1.0,
        )

    call.tiers, call.prompts = tiers, prompts
    return call


def run(reply, state, b=None, **kw):
    call = caller_returning(reply)
    node = make_risk_analyst_node(call, b or budget(), **kw)
    return node(state), call


def test_node_scores_risk_and_keeps_confidence_separate():
    update, _ = run(json.dumps(GOOD), state_with())
    assert update["risk_score"] == pytest.approx(8.05)
    assert update["impacted_products"] == ["Product Alpha"]
    assert update["impacted_suppliers"] == ["ABC Semiconductor"]
    assessment = update["risk_assessment"]
    assert assessment["status"] == "assessed"
    assert assessment["risk_level"] == "HIGH"
    assert assessment["confidence"] == 0.87
    assert assessment["uncertainty"] == []
    assert len(assessment["scenarios"]) == 3
    assert "confidence_score" not in update


def test_node_uses_tier3_untrusted_blocks_and_budget():
    b = budget()
    _, call = run(json.dumps(GOOD), state_with(), b)
    assert call.tiers == [Tier.TIER_3]
    assert '<untrusted_content source="claims">' in call.prompts[0]
    assert "Assess chip rules" in call.prompts[0]
    assert b.calls == 1


def test_node_records_uncertainty_reasons():
    update, _ = run(json.dumps(GOOD), state_with(confidence_score=0.6))
    assert any("0.60" in r for r in update["risk_assessment"]["uncertainty"])


def test_node_custom_weights():
    weights = {"impact": 1, "probability": 0, "urgency": 0}
    update, _ = run(json.dumps(GOOD), state_with(), weights=weights)
    assert update["risk_score"] == pytest.approx(9.0)


def test_node_without_claims_makes_no_model_call():
    update, call = run(json.dumps(GOOD), state_with(claims=[], verified_claims=[]))
    assert call.tiers == []
    assert update["risk_assessment"]["status"] == "unavailable"
    assert update["risk_score"] == 0.0
    assert update["confidence_score"] == 0.0


def test_node_with_garbage_reply_zeroes_confidence():
    update, _ = run("not json", state_with())
    assert update["risk_assessment"]["status"] == "unavailable"
    assert any("no usable risk assessment" in r for r in update["risk_assessment"]["uncertainty"])
    assert update["confidence_score"] == 0.0


def test_routing_after_success_notifies_reviewer():
    state = state_with()
    update, _ = run(json.dumps(GOOD), state)
    state.update(update)
    assert route_after_recommendation(state) == "notify_reviewer"


def test_routing_after_failure_goes_to_analyst_review():
    state = state_with()
    update, _ = run("not json", state)
    state.update(update)
    assert route_after_recommendation(state) == "analyst_review"