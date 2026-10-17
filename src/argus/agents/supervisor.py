import json

from argus.guardrails.budgets import ControlBudget
from argus.models.cascade import ModelCaller, run_cascade

SPECIALISTS = ("news_investigator", "internal_analyst")

PLAN_PROMPT = (
    "You are the supervisor of a global risk-intelligence agent. "
    "Analyse the user's question and plan the investigation. "
    'Put a JSON object in the "answer" field with these keys: '
    "objective (string), entities (list of strings), companies (list of strings), "
    "countries (list of strings), time_horizon (string), requested_output (string), "
    "specialists (list chosen from: news_investigator, internal_analyst). "
    "User question: {query}"
)


def _as_list(value) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


def parse_plan(text: str, query: str) -> dict:
    """Parse the model's plan; anything unusable falls back to a safe default."""
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        data = {}
    specialists = [s for s in dict.fromkeys(_as_list(data.get("specialists"))) if s in SPECIALISTS]
    return {
        "objective": str(data.get("objective") or query),
        "entities": _as_list(data.get("entities")),
        "companies": _as_list(data.get("companies")),
        "countries": _as_list(data.get("countries")),
        "time_horizon": str(data.get("time_horizon") or ""),
        "requested_output": str(data.get("requested_output") or ""),
        "specialists": specialists or list(SPECIALISTS),
    }


def make_supervisor_node(call: ModelCaller, budget: ControlBudget):
    """Build the LangGraph node that analyses the query and plans the work."""

    def supervisor(state: dict) -> dict:
        query = state["user_query"]
        result = run_cascade(PLAN_PROMPT.format(query=query), call, budget)
        plan = parse_plan(result.text, query)
        return {
            "investigation_objective": plan["objective"],
            "entities": plan["entities"],
            "companies": plan["companies"],
            "countries": plan["countries"],
            "investigation_plan": plan,
        }

    return supervisor


def route_after_supervisor(state: dict) -> list[str]:
    """Specialists to run in parallel (a LangGraph conditional edge can return a list)."""
    return state.get("investigation_plan", {}).get("specialists") or list(SPECIALISTS)