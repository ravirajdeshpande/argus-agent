from argus.guardrails.budgets import ControlBudget
from argus.guardrails.evidence import (
    CONFIDENCE_THRESHOLD,
    MAX_ITERATIONS,
    material_unresolved_conflicts,
    next_step,
)

HIGH_RISK_THRESHOLD = 7.0
SENSITIVE_CATEGORIES = {"supplier_communication", "procurement", "legal"}


def route_after_gate(
    state: dict,
    budget: ControlBudget,
    *,
    max_iterations: int = MAX_ITERATIONS,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> str:
    """Evidence gate plus budget exit: the graph decides when research ends."""
    decision = next_step(
        state, max_iterations=max_iterations, confidence_threshold=confidence_threshold
    )
    if decision == "continue_research" and budget.exhausted:
        return "risk_assessment_with_uncertainty"
    return decision


def route_after_recommendation(
    state: dict,
    *,
    high_risk_threshold: float = HIGH_RISK_THRESHOLD,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> str:
    """Human-in-the-loop gates (brief, Section 22), strictest rule first."""
    actions = state.get("recommended_actions", [])
    if any(isinstance(a, dict) and a.get("category") in SENSITIVE_CATEGORIES for a in actions):
        return "human_approval"
    if state["confidence_score"] < confidence_threshold or material_unresolved_conflicts(state):
        return "analyst_review"
    if state["risk_score"] >= high_risk_threshold:
        return "notify_reviewer"
    return "final_evaluation"