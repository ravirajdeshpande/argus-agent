from argus.graph.state import IntelligenceState, initial_state

LIST_FIELDS = [
    "entities", "companies", "countries", "internal_documents", "news_articles",
    "government_sources", "market_signals", "claims", "conflicting_claims",
    "verified_claims", "impacted_products", "impacted_suppliers",
    "missing_information", "recommended_actions", "review_log"
]


def test_initial_state_has_every_field():
    assert set(initial_state("q")) == set(IntelligenceState.__annotations__)


def test_initial_state_defaults():
    s = initial_state("Will the restriction affect us?")
    assert s["user_query"] == "Will the restriction affect us?"
    assert s["investigation_objective"] == ""
    assert s["investigation_iterations"] == 0
    assert s["risk_score"] == 0.0
    assert s["confidence_score"] == 0.0
    assert s["evidence_quality_score"] == 0.0
    assert s["final_report"] == ""
    for name in LIST_FIELDS:
        assert s[name] == []


def test_initial_states_do_not_share_lists():
    a, b = initial_state("a"), initial_state("b")
    a["claims"].append("x")
    assert b["claims"] == []
    
def test_investigation_plan_defaults_to_empty_dict_not_shared():
    a, b = initial_state("a"), initial_state("b")
    assert a["investigation_plan"] == {}
    a["investigation_plan"]["x"] = 1
    assert b["investigation_plan"] == {}
    
def test_evaluation_defaults_to_empty_dict_not_shared():
    a, b = initial_state("a"), initial_state("b")
    assert a["evaluation"] == {}
    a["evaluation"]["x"] = 1
    assert b["evaluation"] == {}