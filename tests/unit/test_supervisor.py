import json

from argus.agents.supervisor import (
    SPECIALISTS,
    make_supervisor_node,
    parse_plan,
    route_after_supervisor,
)
from argus.graph.state import initial_state
from argus.guardrails.budgets import ControlBudget
from argus.models.cascade import ModelResponse

GOOD = {
    "objective": "Assess impact of new chip export rules",
    "entities": ["export restriction"],
    "companies": ["ASML"],
    "countries": ["NL", "CN"],
    "time_horizon": "next 90 days",
    "requested_output": "executive report",
    "specialists": ["news_investigator", "internal_analyst"],
}


def budget():
    return ControlBudget(max_calls=10, max_tokens=1_000_000, max_cost_usd=100.0)


def caller_returning(text, confidence=0.9):
    prompts = []

    def call(tier, prompt):
        prompts.append(prompt)
        return ModelResponse(
            text=text, confidence=confidence, model="m",
            input_tokens=10, output_tokens=5, latency_ms=1.0,
        )

    call.prompts = prompts
    return call


def test_parse_plan_reads_all_fields():
    plan = parse_plan(json.dumps(GOOD), "q")
    assert plan == GOOD


def test_parse_plan_accepts_json_inside_code_fence():
    plan = parse_plan("```json\n" + json.dumps(GOOD) + "\n```", "q")
    assert plan["objective"] == GOOD["objective"]


def test_parse_plan_invalid_json_falls_back():
    plan = parse_plan("sorry, no idea", "my question")
    assert plan["objective"] == "my question"
    assert plan["entities"] == []
    assert plan["specialists"] == list(SPECIALISTS)


def test_parse_plan_drops_unknown_specialists():
    plan = parse_plan(json.dumps({"specialists": ["news_investigator", "hacker"]}), "q")
    assert plan["specialists"] == ["news_investigator"]


def test_parse_plan_all_unknown_specialists_falls_back_to_all():
    plan = parse_plan(json.dumps({"specialists": ["hacker"]}), "q")
    assert plan["specialists"] == list(SPECIALISTS)


def test_parse_plan_non_list_fields_become_empty_lists():
    plan = parse_plan(json.dumps({"entities": "ASML", "companies": None}), "q")
    assert plan["entities"] == []
    assert plan["companies"] == []


def test_supervisor_node_returns_state_updates():
    node = make_supervisor_node(caller_returning(json.dumps(GOOD)), budget())
    update = node(initial_state("Will the new rule hit us?"))
    assert update["investigation_objective"] == GOOD["objective"]
    assert update["companies"] == ["ASML"]
    assert update["countries"] == ["NL", "CN"]
    assert update["investigation_plan"]["time_horizon"] == "next 90 days"


def test_supervisor_node_puts_query_in_prompt_and_uses_budget():
    call = caller_returning(json.dumps(GOOD))
    b = budget()
    make_supervisor_node(call, b)(initial_state("Will the new rule hit us?"))
    assert "Will the new rule hit us?" in call.prompts[0]
    assert b.calls == 1


def test_supervisor_node_survives_garbage_reply():
    node = make_supervisor_node(caller_returning("not json"), budget())
    update = node(initial_state("my question"))
    assert update["investigation_objective"] == "my question"
    assert update["investigation_plan"]["specialists"] == list(SPECIALISTS)


def test_route_after_supervisor_returns_planned_specialists():
    state = initial_state("q")
    state["investigation_plan"] = {"specialists": ["internal_analyst"]}
    assert route_after_supervisor(state) == ["internal_analyst"]


def test_route_after_supervisor_defaults_to_all_specialists():
    assert route_after_supervisor(initial_state("q")) == list(SPECIALISTS)