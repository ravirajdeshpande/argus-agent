from typing import TypedDict


class IntelligenceState(TypedDict):
    """Shared investigation memory and audit surface (brief, Section 7)."""

    user_query: str
    investigation_objective: str
    investigation_plan: dict
    entities: list
    companies: list
    countries: list
    internal_documents: list
    news_articles: list
    government_sources: list
    market_signals: list
    claims: list
    conflicting_claims: list
    verified_claims: list
    impacted_products: list
    impacted_suppliers: list
    missing_information: list
    investigation_iterations: int
    risk_score: float
    risk_assessment: dict
    confidence_score: float
    evidence_quality_score: float
    recommended_actions: list
    evaluation: dict
    review_log: list
    final_report: str


def initial_state(user_query: str) -> IntelligenceState:
    return IntelligenceState(
        user_query=user_query,
        investigation_objective="",
        investigation_plan={},
        entities=[],
        companies=[],
        countries=[],
        internal_documents=[],
        news_articles=[],
        government_sources=[],
        market_signals=[],
        claims=[],
        conflicting_claims=[],
        verified_claims=[],
        impacted_products=[],
        impacted_suppliers=[],
        missing_information=[],
        investigation_iterations=0,
        risk_score=0.0,
        risk_assessment={},
        confidence_score=0.0,
        evidence_quality_score=0.0,
        recommended_actions=[],
        evaluation={},
        review_log=[],
        final_report="",
    )
    
def test_risk_assessment_defaults_to_empty_dict_not_shared():
    a, b = initial_state("a"), initial_state("b")
    assert a["risk_assessment"] == {}
    a["risk_assessment"]["x"] = 1
    assert b["risk_assessment"] == {}