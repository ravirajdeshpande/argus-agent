import json

import pytest

from argus.agents.recommender import make_recommender_node, parse_actions
from argus.config.models import Tier
from argus.graph.routing import route_after_recommendation
from argus.graph.state import initial_state
from argus.guardrails.budgets import ControlBudget
from argus.guardrails.evidence import Claim
from argus.models.cascade import ModelResponse

VALID = {"gov_1", "news_1"}

GOOD = {
    "actions": [
        {"action": "Contact Supplier ABC", "category": "supplier_communication",
         "owner": "Procurement lead", "urgency": "immediate",
         "rationale": "Supplies category X components", "sources": ["gov_1"]},
        {"action": "Verify inventory and committed orders", "category": "inventory",
         "owner": "Supply chain manager", "urgency": "high",
         "rationale": "Coverage is limited", "sources": ["gov_1", "ghost"]},
        {"action": "Continue monitoring regulatory updates", "category": "monitoring",
         "owner": "Risk analyst", "urgency": "low", "rationale": "Rules may change",
         "sources": []},
    ]
}


def act(action="Do something", **kw):
    return {"action": action, "category": "monitoring", "owner": "Owner",
            "urgency": "medium", "rationale": "r", "sources": [], **kw}


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
    s.update(
        investigation_objective="Assess chip rules",
        claims=[claim()],
        confidence_score=0.87,
        risk_score=8.05,
        impacted_products=["Product Alpha"],
        impacted_suppliers=["ABC Semiconductor"],
        risk_assessment={
            "status": "assessed", "risk_level": "HIGH", "risk_score": 8.05,
            "expected_impact": "Supply interruption", "rationale": "Category X affected",
            "uncertainty": [], "scenarios": [],
        },
    )
    s.update(fields)
    return s


def budget():
    return ControlBudget(max_calls=10, max_tokens=1_000_000, max_cost_usd=100.0)


# ---- parse_actions -------------------------------------------------------

def test_parse_actions_reads_fields_priority_and_approval_flag():
    actions = parse_actions(json.dumps(GOOD), VALID)
    assert [a["priority"] for a in actions] == [1, 2, 3]
    first = actions[0]
    assert first["action"] == "Contact Supplier ABC"
    assert first["category"] == "supplier_communication"
    assert first["owner"] == "Procurement lead"
    assert first["urgency"] == "immediate"
    assert first["sources"] == ["gov_1"]
    assert first["requires_approval"] is True
    assert actions[2]["requires_approval"] is False


def test_actions_are_sorted_by_urgency_keeping_order_for_ties():
    raw = {"actions": [
        act("A", urgency="low"), act("B", urgency="immediate"),
        act("C", urgency="high"), act("D", urgency="immediate"),
    ]}
    actions = parse_actions(json.dumps(raw), VALID)
    assert [a["action"] for a in actions] == ["B", "D", "C", "A"]
    assert [a["priority"] for a in actions] == [1, 2, 3, 4]


def test_category_is_normalised_and_unknown_category_fails_safe():
    raw = {"actions": [
        act("A", category="Supplier Communication"),
        act("B", category="do-whatever"),
        act("C", category="monitoring"),
    ]}
    by_name = {a["action"]: a for a in parse_actions(json.dumps(raw), VALID)}
    assert by_name["A"]["category"] == "supplier_communication"
    assert by_name["A"]["requires_approval"] is True
    assert by_name["B"]["category"] == "uncategorized"
    assert by_name["B"]["requires_approval"] is True
    assert by_name["C"]["requires_approval"] is False


def test_unknown_urgency_defaults_to_medium_and_empty_owner_to_unassigned():
    raw = {"actions": [act("A", urgency="asap", owner="  ")]}
    a = parse_actions(json.dumps(raw), VALID)[0]
    assert a["urgency"] == "medium"
    assert a["owner"] == "unassigned"


def test_empty_junk_and_duplicate_actions_are_dropped():
    raw = {"actions": [act(""), "junk", act("Call supplier"), act("call SUPPLIER")]}
    actions = parse_actions(json.dumps(raw), VALID)
    assert [a["action"] for a in actions] == ["Call supplier"]


def test_unknown_sources_are_dropped():
    raw = {"actions": [act("A", sources=["gov_1", "ghost", "news_1"])]}
    assert parse_actions(json.dumps(raw), VALID)[0]["sources"] == ["gov_1", "news_1"]


def test_max_actions_caps_the_list_after_sorting():
    raw = {"actions": [act(f"A{i}", urgency="low") for i in range(8)] + [act("Urgent", urgency="immediate")]}
    actions = parse_actions(json.dumps(raw), VALID, max_actions=3)
    assert len(actions) == 3
    assert actions[0]["action"] == "Urgent"


def test_garbage_gives_no_actions():
    assert parse_actions("nope", VALID) == []
    assert parse_actions(json.dumps({"actions": "x"}), VALID) == []


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


def run(reply, state, b=None):
    call = caller_returning(reply)
    node = make_recommender_node(call, b or budget())
    return node(state), call


def test_node_recommends_with_tier3_untrusted_blocks_and_budget():
    b = budget()
    update, call = run(json.dumps(GOOD), state_with(), b)
    assert call.tiers == [Tier.TIER_3]
    assert b.calls == 1
    assert '<untrusted_content source="risk_assessment">' in call.prompts[0]
    assert '<untrusted_content source="claims">' in call.prompts[0]
    assert "Assess chip rules" in call.prompts[0]
    actions = update["recommended_actions"]
    assert actions[0]["action"] == "Contact Supplier ABC"
    assert actions[1]["sources"] == ["gov_1"]
    assert "confidence_score" not in update


def test_node_without_assessment_makes_no_call_and_asks_for_manual_review():
    state = state_with(risk_assessment={"status": "unavailable", "uncertainty": ["no primary source found"]})
    update, call = run(json.dumps(GOOD), state)
    assert call.tiers == []
    [action] = update["recommended_actions"]
    assert action["category"] == "monitoring"
    assert action["requires_approval"] is False
    assert "no primary source found" in action["rationale"]


def test_node_with_unusable_reply_falls_back_and_zeroes_confidence():
    update, _ = run("not json", state_with())
    [action] = update["recommended_actions"]
    assert "manual" in action["action"].lower()
    assert update["confidence_score"] == 0.0


def test_sensitive_action_routes_to_human_approval():
    state = state_with()
    update, _ = run(json.dumps(GOOD), state)
    state.update(update)
    assert route_after_recommendation(state) == "human_approval"


def test_monitoring_only_actions_notify_reviewer_for_high_risk():
    reply = json.dumps({"actions": [act("Watch notices")]})
    state = state_with()
    update, _ = run(reply, state)
    state.update(update)
    assert route_after_recommendation(state) == "notify_reviewer"


def test_failed_recommendation_routes_to_analyst_review():
    state = state_with()
    update, _ = run("not json", state)
    state.update(update)
    assert route_after_recommendation(state) == "analyst_review"