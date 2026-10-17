from collections.abc import Callable
from datetime import date

from langgraph.graph import END, START, StateGraph

from argus.agents.evaluator import make_evaluator_node
from argus.agents.internal_analyst import make_internal_analyst_node
from argus.agents.news_investigator import make_news_investigator_node
from argus.agents.recommender import make_recommender_node, manual_review_action
from argus.agents.reporter import make_reporter_node
from argus.agents.risk_analyst import make_risk_analyst_node
from argus.agents.supervisor import SPECIALISTS, make_supervisor_node, route_after_supervisor
from argus.agents.verifier import make_verifier_node
from argus.graph.checkpoints import thread_config
from argus.graph.routing import route_after_gate, route_after_recommendation
from argus.graph.state import IntelligenceState, initial_state
from argus.guardrails.access_control import Confidentiality
from argus.guardrails.budgets import ControlBudget
from argus.guardrails.injection import check_input
from argus.models.cascade import BudgetExhaustedError, ModelCaller

RECURSION_LIMIT = 60
BLOCKING_GATES = ["human_approval", "analyst_review"]


class InputRejected(ValueError):
    """The user's question was blocked by the input guardrail."""


# ---- graceful degradation when the budget runs out (brief, Section 21) ----

def _noted(state: dict, note: str) -> list[str]:
    return list(dict.fromkeys([*state["missing_information"], note]))


def _fallback_supervisor(state: dict, reason: str) -> dict:
    return {
        "investigation_objective": state["user_query"],
        "investigation_plan": {"specialists": list(SPECIALISTS)},
        "missing_information": _noted(state, f"budget exhausted ({reason}): planning skipped"),
    }


def _fallback_news(state: dict, reason: str) -> dict:
    return {
        "investigation_iterations": state["investigation_iterations"] + 1,
        "missing_information": _noted(state, f"budget exhausted ({reason}): research stopped"),
    }


def _fallback_verifier(state: dict, reason: str) -> dict:
    return {"missing_information": _noted(state, f"budget exhausted ({reason}): verification skipped")}


def _fallback_risk(state: dict, reason: str) -> dict:
    return {
        "risk_score": 0.0,
        "confidence_score": 0.0,
        "impacted_products": [],
        "impacted_suppliers": [],
        "risk_assessment": {
            "status": "unavailable",
            "uncertainty": [f"budget exhausted ({reason}) before the risk assessment"],
        },
    }


def _fallback_recommender(state: dict, reason: str) -> dict:
    return {
        "recommended_actions": [manual_review_action(f"budget exhausted ({reason})")],
        "confidence_score": 0.0,
    }


def _budget_safe(node: Callable[[dict], dict], fallback: Callable[[dict, str], dict]):
    def safe(state: dict) -> dict:
        try:
            return node(state)
        except BudgetExhaustedError as exc:
            return fallback(state, str(exc))

    return safe


# ---- human-in-the-loop gates (brief, Section 22) --------------------------

def _gate_node(name: str, reason_fn: Callable[[dict], str], *, status: str, notify=None):
    def gate(state: dict) -> dict:
        entry = {"gate": name, "reason": reason_fn(state), "status": status}
        if notify is not None:
            notify(entry)
        return {"review_log": [*state["review_log"], entry]}

    return gate


def _approval_reason(state: dict) -> str:
    flagged = [a["action"] for a in state["recommended_actions"] if a.get("requires_approval")]
    return "; ".join(flagged) or "sensitive action"


def _analyst_reason(state: dict) -> str:
    conflicts = sum(1 for c in state["claims"] if c.contradiction)
    return f"confidence {state['confidence_score']:.2f}; unresolved conflicts: {conflicts}"


def _notify_reason(state: dict) -> str:
    return f"high-risk finding ({state['risk_score']:.1f} / 10) with strong evidence"


def build_graph(
    *,
    call: ModelCaller,
    budget: ControlBudget,
    store,
    clearance: Confidentiality,
    tools: dict,
    checkpointer=None,
    notify: Callable[[dict], None] | None = None,
    pause_at_gates: bool = False,
    today: Callable[[], date] = date.today,
):
    """Wire every ARGUS node into one LangGraph and compile it."""
    if pause_at_gates and checkpointer is None:
        raise ValueError("pause_at_gates needs a checkpointer to save the paused investigation")

    paused = "released" if pause_at_gates else "not_paused"
    g = StateGraph(IntelligenceState)

    g.add_node("supervisor", _budget_safe(make_supervisor_node(call, budget), _fallback_supervisor))
    g.add_node(
        "news_investigator",
        _budget_safe(make_news_investigator_node(call, budget, tools), _fallback_news),
    )
    g.add_node("internal_analyst", make_internal_analyst_node(store, clearance, today=today))
    g.add_node("verifier", _budget_safe(make_verifier_node(call, budget, today=today), _fallback_verifier))
    g.add_node("risk_analyst", _budget_safe(make_risk_analyst_node(call, budget), _fallback_risk))
    g.add_node("recommender", _budget_safe(make_recommender_node(call, budget), _fallback_recommender))
    g.add_node("human_approval", _gate_node("human_approval", _approval_reason, status=paused))
    g.add_node("analyst_review", _gate_node("analyst_review", _analyst_reason, status=paused))
    g.add_node(
        "notify_reviewer",
        _gate_node("notify_reviewer", _notify_reason, status="notified", notify=notify),
    )
    g.add_node("evaluator", make_evaluator_node())
    g.add_node("reporter", make_reporter_node())

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_after_supervisor, list(SPECIALISTS))
    g.add_edge("news_investigator", "verifier")
    g.add_edge("internal_analyst", "verifier")
    g.add_conditional_edges(
        "verifier",
        lambda state: route_after_gate(state, budget),
        {
            "continue_research": "news_investigator",
            "risk_assessment": "risk_analyst",
            "risk_assessment_with_uncertainty": "risk_analyst",
        },
    )
    g.add_edge("risk_analyst", "recommender")
    g.add_conditional_edges(
        "recommender",
        route_after_recommendation,
        {
            "human_approval": "human_approval",
            "analyst_review": "analyst_review",
            "notify_reviewer": "notify_reviewer",
            "final_evaluation": "evaluator",
        },
    )
    for gate in ("human_approval", "analyst_review", "notify_reviewer"):
        g.add_edge(gate, "evaluator")
    g.add_edge("evaluator", "reporter")
    g.add_edge("reporter", END)

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=BLOCKING_GATES if pause_at_gates else None,
    )


def run_investigation(graph, query: str, *, thread_id: str = "default") -> dict:
    """Guard the question, then run the investigation under a thread id."""
    check = check_input(query)
    if not check.allowed:
        raise InputRejected(check.reason)
    config = {**thread_config(thread_id), "recursion_limit": RECURSION_LIMIT}
    return graph.invoke(initial_state(check.text), config)

def resume_investigation(graph, *, thread_id: str = "default") -> dict:
    """Continue a paused investigation from its last checkpoint."""
    config = {**thread_config(thread_id), "recursion_limit": RECURSION_LIMIT}
    return graph.invoke(None, config)