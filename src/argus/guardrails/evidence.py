from pydantic import BaseModel, ConfigDict, Field

MAX_ITERATIONS = 5
CONFIDENCE_THRESHOLD = 0.75


class Claim(BaseModel):
    """One material assertion with its evidence trail."""

    model_config = ConfigDict(frozen=True)

    claim: str
    claim_type: str
    subject: str = ""
    value: str = ""
    sources: list[str]
    primary_source: bool
    source_authority: float = Field(ge=0, le=1)
    recency_score: float = Field(ge=0, le=1)
    corroboration_score: float = Field(ge=0, le=1)
    contradiction: bool
    confidence: float = Field(ge=0, le=1)


def material_unresolved_conflicts(state: dict) -> bool:
    return any(c.contradiction for c in state.get("claims", []))


def next_step(
    state: dict,
    *,
    max_iterations: int = MAX_ITERATIONS,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> str:
    """Evidence Sufficiency Gate: the graph, not the LLM, decides when research ends."""
    if state["investigation_iterations"] >= max_iterations:
        return "risk_assessment_with_uncertainty"
    if state["confidence_score"] < confidence_threshold:
        return "continue_research"
    if state.get("missing_information"):
        return "continue_research"
    if material_unresolved_conflicts(state):
        return "continue_research"
    return "risk_assessment"