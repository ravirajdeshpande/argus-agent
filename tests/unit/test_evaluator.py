import pytest

from argus.agents.evaluator import citation_coverage, make_evaluator_node
from argus.graph.state import initial_state
from argus.guardrails.evidence import Claim


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
        claims=[claim()],
        government_sources=[
            {"source_id": "gov_1", "tool": "search_regulation", "args": {},
             "result": {"notice": "Category X components are controlled"}}
        ],
        internal_documents=[
            {"source_id": "internal_1",
             "text": "Supplier ABC Semiconductor provides category X components to Product Alpha.",
             "metadata": {"supplier": "ABC Semiconductor", "product": "Product Alpha", "source": "c.txt"}}
        ],
        impacted_suppliers=["ABC Semiconductor"],
        impacted_products=["Product Alpha"],
        risk_assessment={"status": "assessed"},
        recommended_actions=[{"action": "Contact supplier", "sources": ["gov_1"]}],
    )
    s.update(fields)
    return s


def run(state, **kw):
    return make_evaluator_node(**kw)(state)


def test_grounded_state_passes_without_repairs():
    update = run(state_with())
    ev = update["evaluation"]
    assert ev["passed"] is True
    assert ev["citation_coverage"] == 1.0
    assert ev["repairs"] == []
    assert ev["issues"] == []
    assert update["impacted_suppliers"] == ["ABC Semiconductor"]
    assert update["impacted_products"] == ["Product Alpha"]


def test_unsupported_supplier_is_removed_and_reported():
    update = run(state_with(impacted_suppliers=["ABC Semiconductor", "Phantom Corp"]))
    assert update["impacted_suppliers"] == ["ABC Semiconductor"]
    assert update["evaluation"]["repairs"] == ["removed unsupported supplier: Phantom Corp"]
    assert update["evaluation"]["passed"] is True


def test_unsupported_product_is_removed_and_reported():
    update = run(state_with(impacted_products=["Product Alpha", "Product Zeta"]))
    assert update["impacted_products"] == ["Product Alpha"]
    assert update["evaluation"]["repairs"] == ["removed unsupported product: Product Zeta"]


def test_name_matching_ignores_case_and_punctuation():
    update = run(state_with(impacted_suppliers=["abc semiconductor."]))
    assert update["impacted_suppliers"] == ["abc semiconductor."]


def test_partial_word_does_not_count_as_support():
    update = run(state_with(impacted_suppliers=["Semi"]))
    assert update["impacted_suppliers"] == []


def test_external_evidence_can_support_a_name():
    state = state_with(
        impacted_suppliers=["Globex"],
        news_articles=[
            {"source_id": "news_1", "tool": "search_company_news", "args": {},
             "result": {"title": "Globex halts deliveries"}}
        ],
    )
    assert run(state)["impacted_suppliers"] == ["Globex"]


def test_claims_citing_unknown_sources_lower_coverage_and_fail():
    state = state_with(claims=[claim(), claim(claim="other", sources=["ghost_1"])])
    ev = run(state)["evaluation"]
    assert ev["citation_coverage"] == 0.5
    assert ev["passed"] is False
    assert any("citation coverage" in i for i in ev["issues"])


def test_min_citation_coverage_is_configurable():
    state = state_with(claims=[claim(), claim(claim="other", sources=["ghost_1"])])
    assert run(state, min_citation_coverage=0.4)["evaluation"]["passed"] is True


def test_no_claims_is_incomplete():
    ev = run(state_with(claims=[]))["evaluation"]
    assert ev["citation_coverage"] == 0.0
    assert "claims" in ev["missing_sections"]
    assert ev["passed"] is False


def test_missing_assessment_and_actions_are_reported():
    ev = run(state_with(risk_assessment={"status": "unavailable"}, recommended_actions=[]))["evaluation"]
    assert set(ev["missing_sections"]) == {"risk assessment", "recommended actions"}
    assert ev["passed"] is False


def test_unresolved_conflicts_are_counted_but_do_not_fail():
    state = state_with(conflicting_claims=[{"resolved": False}, {"resolved": True}])
    ev = run(state)["evaluation"]
    assert ev["unresolved_conflicts"] == 1
    assert ev["passed"] is True


def test_action_citation_coverage():
    state = state_with(
        recommended_actions=[{"action": "a", "sources": ["gov_1"]}, {"action": "b", "sources": []}]
    )
    assert run(state)["evaluation"]["action_citation_coverage"] == 0.5


def test_citation_coverage_helper():
    assert citation_coverage([["a"], ["x"], []], {"a"}) == pytest.approx(1 / 3)
    assert citation_coverage([], {"a"}) == 0.0