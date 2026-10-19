import json
from datetime import date

import pytest

from argus.agents.verifier import (
    build_claim,
    index_evidence,
    make_verifier_node,
    parse_claims,
    prune_missing,
    resolve_claims,
)
from argus.config.models import Tier
from argus.graph.state import initial_state
from argus.guardrails.budgets import ControlBudget
from argus.guardrails.evidence import next_step
from argus.models.cascade import ModelResponse
from argus.guardrails.evidence import Claim
from argus.agents.verifier import resolve_claims  # skip if already imported

AS_OF = date(2026, 10, 17)


def news(n, **result):
    return {"source_id": f"news_{n}", "tool": "search_news", "args": {}, "result": result,
            "injection_flagged": False}


def gov(n, **result):
    return {"source_id": f"gov_{n}", "tool": "search_regulation", "args": {}, "result": result,
            "injection_flagged": False}


def internal(n, text, **meta):
    return {"source_id": f"internal_{n}", "text": text, "metadata": meta}


def state_with(**fields):
    s = initial_state("Will it affect us?")
    s.update(fields)
    return s


def raw(claim="Restrictions begin September 1", sources=("news_1",),
        claim_type="effective_date", subject="export restriction", value="September 1"):
    return {"claim": claim, "claim_type": claim_type, "subject": subject,
            "value": value, "sources": list(sources)}


def budget():
    return ControlBudget(max_calls=10, max_tokens=1_000_000, max_cost_usd=100.0)


# ---- index and scoring ---------------------------------------------------

def test_index_evidence_collects_all_evidence_fields():
    state = state_with(
        news_articles=[news(1)],
        government_sources=[gov(1)],
        market_signals=[{"source_id": "market_1", "tool": "get_fx_rate", "args": {}, "result": {}}],
        internal_documents=[internal(1, "t", source="a.txt")],
    )
    assert set(index_evidence(state)) == {"news_1", "gov_1", "market_1", "internal_1"}


def test_government_source_is_primary_and_authoritative():
    idx = index_evidence(state_with(government_sources=[gov(1)]))
    c = build_claim(raw(sources=["gov_1"]), idx, AS_OF)
    assert c.primary_source is True
    assert c.source_authority == 0.95
    assert c.recency_score == 0.5
    assert c.corroboration_score == pytest.approx(0.5)
    assert c.confidence == pytest.approx(0.68)
    assert c.contradiction is False


def test_news_only_claim_is_not_primary():
    idx = index_evidence(state_with(news_articles=[news(1)]))
    c = build_claim(raw(sources=["news_1"]), idx, AS_OF)
    assert c.primary_source is False
    assert c.source_authority == 0.6


def test_same_publisher_counts_as_one_independent_source():
    idx = index_evidence(
        state_with(news_articles=[news(1, publisher="Reuters"), news(2, publisher="reuters")])
    )
    c = build_claim(raw(sources=["news_1", "news_2"]), idx, AS_OF)
    assert c.corroboration_score == pytest.approx(0.5)


def test_independent_publishers_increase_corroboration():
    idx = index_evidence(
        state_with(
            news_articles=[
                news(1, publisher="Reuters"),
                news(2, publisher="FT"),
                news(3, url="https://bbc.co.uk/x"),
            ]
        )
    )
    c = build_claim(raw(sources=["news_1", "news_2", "news_3"]), idx, AS_OF)
    assert c.corroboration_score == pytest.approx(0.875)


def test_recency_uses_freshest_source_date():
    idx = index_evidence(
        state_with(news_articles=[news(1, date="2026-10-17"), news(2, date="2020-01-01")])
    )
    c = build_claim(raw(sources=["news_1", "news_2"]), idx, AS_OF)
    assert c.recency_score == pytest.approx(1.0)


def test_old_source_has_low_recency():
    idx = index_evidence(state_with(news_articles=[news(1, date="2026-04-20")]))
    c = build_claim(raw(sources=["news_1"]), idx, AS_OF)
    assert c.recency_score == pytest.approx(0.25)


def test_internal_document_date_comes_from_metadata():
    idx = index_evidence(
        state_with(internal_documents=[internal(1, "t", source="c.txt", date="2026-10-17")])
    )
    c = build_claim(raw(sources=["internal_1"]), idx, AS_OF)
    assert c.recency_score == pytest.approx(1.0)
    assert c.primary_source is True


# ---- conflicts -----------------------------------------------------------

def test_conflict_without_primary_evidence_is_unresolved():
    idx = index_evidence(
        state_with(news_articles=[news(1, publisher="A"), news(2, publisher="B")])
    )
    a = build_claim(raw(sources=["news_1"], value="September 1"), idx, AS_OF)
    b = build_claim(
        raw(claim="Effective immediately", sources=["news_2"], value="immediately"), idx, AS_OF
    )
    final, verified, conflicts = resolve_claims([a, b])
    assert all(c.contradiction for c in final)
    assert final[0].confidence == pytest.approx(a.confidence * 0.7)
    assert verified == []
    assert conflicts[0]["resolved"] is False


def test_primary_source_resolves_conflict():
    idx = index_evidence(state_with(government_sources=[gov(1)], news_articles=[news(1)]))
    g = build_claim(raw(sources=["gov_1"], value="September 1"), idx, AS_OF)
    n = build_claim(
        raw(claim="Effective immediately", sources=["news_1"], value="immediately"), idx, AS_OF
    )
    final, verified, conflicts = resolve_claims([g, n])
    assert [c.claim for c in final] == [g.claim]
    assert final[0].contradiction is False
    assert verified == final
    assert conflicts[0]["resolved"] is True
    assert conflicts[0]["winning_value"] == "september 1"
    assert conflicts[0]["rejected_claims"] == ["Effective immediately"]


def test_two_disagreeing_primary_sources_stay_unresolved():
    idx = index_evidence(state_with(government_sources=[gov(1), gov(2)]))
    g1 = build_claim(raw(claim="a", sources=["gov_1"], value="September 1"), idx, AS_OF)
    g2 = build_claim(raw(claim="b", sources=["gov_2"], value="October 1"), idx, AS_OF)
    final, _, conflicts = resolve_claims([g1, g2])
    assert conflicts[0]["resolved"] is False
    assert all(c.contradiction for c in final)


@pytest.mark.parametrize(
    "a_kw, b_kw",
    [
        (dict(value="September 1"), dict(value="september 1.")),
        (dict(subject="A"), dict(subject="B", value="immediately")),
        (dict(claim_type="effective_date"), dict(claim_type="scope", value="immediately")),
        (dict(subject=""), dict(subject="", value="immediately")),
    ],
)
def test_claims_that_do_not_really_conflict(a_kw, b_kw):
    idx = index_evidence(
        state_with(news_articles=[news(1, publisher="A"), news(2, publisher="B")])
    )
    a = build_claim(raw(claim="a", sources=["news_1"], **a_kw), idx, AS_OF)
    b = build_claim(raw(claim="b", sources=["news_2"], **b_kw), idx, AS_OF)
    final, _, conflicts = resolve_claims([a, b])
    assert conflicts == []
    assert not any(c.contradiction for c in final)


def test_verified_needs_primary_source_or_independent_corroboration():
    idx = index_evidence(
        state_with(news_articles=[news(1, publisher="A"), news(2, publisher="B")])
    )
    single = build_claim(raw(sources=["news_1"]), idx, AS_OF)
    double = build_claim(raw(claim="d", sources=["news_1", "news_2"], subject="other"), idx, AS_OF)
    _, verified, _ = resolve_claims([single, double])
    assert verified == [double]


# ---- parsing and housekeeping --------------------------------------------

def test_parse_claims_merges_repeated_claims_and_unions_sources():
    text = json.dumps(
        {
            "claims": [
                {"claim": "a", "claim_type": "effective_date", "subject": "Rule X",
                 "value": "Sept 1", "sources": ["news_1"]},
                {"claim": "b", "claim_type": "effective_date", "subject": "rule x",
                 "value": "sept 1", "sources": ["news_2", "news_1"]},
            ]
        }
    )
    out = parse_claims(text, {"news_1", "news_2"})
    assert len(out) == 1
    assert out[0]["sources"] == ["news_1", "news_2"]


def test_parse_claims_drops_unknown_sources_and_empty_claims():
    text = json.dumps(
        {
            "claims": [
                {"claim": "ok", "sources": ["news_1", "ghost"]},
                {"claim": "no valid source", "sources": ["ghost"]},
                {"claim": "", "sources": ["news_1"]},
                "junk",
            ]
        }
    )
    out = parse_claims(text, {"news_1"})
    assert [c["claim"] for c in out] == ["ok"]
    assert out[0]["sources"] == ["news_1"]
    assert out[0]["claim_type"] == "other"


def test_parse_claims_garbage_returns_empty_list():
    assert parse_claims("nope", {"news_1"}) == []


def test_prune_missing_drops_failures_that_later_succeeded():
    state = state_with(news_articles=[news(1)])
    missing = ["search_news failed: slow", "get_notice failed: down", "need inventory data"]
    assert prune_missing(missing, state) == ["get_notice failed: down", "need inventory data"]


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


CLAIMS_REPLY = json.dumps(
    {
        "claims": [
            {"claim": "Restrictions begin September 1", "claim_type": "effective_date",
             "subject": "export restriction", "value": "September 1", "sources": ["gov_1"]}
        ]
    }
)


def run(reply, state, b=None):
    call = caller_returning(reply)
    node = make_verifier_node(call, b or budget(), today=lambda: AS_OF)
    return node(state), call


def test_node_extracts_scores_and_verifies_claims():
    b = budget()
    state = state_with(
        government_sources=[gov(1, notice="Controls start Sept 1")],
        investigation_objective="Assess chip rules",
    )
    update, call = run(CLAIMS_REPLY, state, b)
    assert call.tiers == [Tier.TIER_2]
    assert b.calls == 1
    assert '<untrusted_content source="gov_1">' in call.prompts[0]
    assert "Assess chip rules" in call.prompts[0]
    [claim] = update["claims"]
    assert claim.sources == ["gov_1"]
    assert update["verified_claims"] == [claim]
    assert update["conflicting_claims"] == []
    assert update["confidence_score"] == pytest.approx(0.68)
    assert update["evidence_quality_score"] == pytest.approx(0.725)


def test_node_without_evidence_makes_no_model_call():
    update, call = run(CLAIMS_REPLY, state_with())
    assert call.tiers == []
    assert update["claims"] == []
    assert update["confidence_score"] == 0.0
    assert update["evidence_quality_score"] == 0.0


def test_node_with_garbage_reply_yields_no_claims():
    update, _ = run("not json", state_with(news_articles=[news(1)]))
    assert update["claims"] == []
    assert update["confidence_score"] == 0.0


def test_unresolved_conflict_flows_into_evidence_gate():
    reply = json.dumps(
        {
            "claims": [
                {"claim": "begin Sept 1", "claim_type": "effective_date", "subject": "rule",
                 "value": "September 1", "sources": ["news_1"]},
                {"claim": "effective immediately", "claim_type": "effective_date",
                 "subject": "rule", "value": "immediately", "sources": ["news_2"]},
            ]
        }
    )
    state = state_with(news_articles=[news(1, publisher="A"), news(2, publisher="B")])
    update, _ = run(reply, state)
    assert len(update["conflicting_claims"]) == 1
    state.update(update)
    state["confidence_score"] = 0.99
    assert next_step(state) == "continue_research"


def test_node_prunes_resolved_tool_failures():
    state = state_with(
        news_articles=[news(1)],
        missing_information=["search_news failed: slow", "need inventory"],
    )
    update, _ = run("{}", state)
    assert update["missing_information"] == ["need inventory"]
    
def _claim(text, claim_type, subject, value):
    return Claim(
        claim=text, claim_type=claim_type, subject=subject, value=value,
        sources=["internal_1"], primary_source=True, source_authority=0.85,
        recency_score=0.5, corroboration_score=0.5, contradiction=False, confidence=0.6,
    )
def test_non_exclusive_claim_types_never_conflict():
    a = _claim("ABC supplies X", "supplier_status", "abc semiconductor", "supplier of x")
    b = _claim("ABC is sole approved supplier", "supplier_status", "abc semiconductor", "sole approved supplier")
    c = _claim("45 days", "inventory", "category x", "45 days")
    d = _claim("2 months committed", "inventory", "category x", "2 months")
    _, _, conflicts = resolve_claims([a, b, c, d])
    assert conflicts == []

def test_same_date_in_different_formats_is_not_a_conflict():
    a = _claim("a", "effective_date", "export controls on advanced equipment", "2027-01-15")
    b = _claim("b", "effective_date", "export restrictions on advanced equipment", "January 15, 2027")
    assert resolve_claims([a, b])[2] == []


def test_primary_source_resolves_differently_worded_subjects():
    gov = _claim("gov", "effective_date",
                 "export controls on advanced semiconductor manufacturing equipment", "January 15, 2027")
    news = _claim("news", "effective_date",
                  "new export restrictions on advanced chip making equipment",
                  "immediately").model_copy(update={"primary_source": False})
    final, _, conflicts = resolve_claims([gov, news])
    assert conflicts[0]["resolved"] and conflicts[0]["winning_value"] == "2027-01-15"
    assert [c.claim for c in final] == ["gov"]

def test_date_inside_a_sentence_matches_a_bare_date():
    a = _claim("a", "effective_date", "export controls on advanced equipment", "take effect January 15, 2027")
    b = _claim("b", "effective_date", "export restrictions on advanced equipment", "2027-01-15")
    assert resolve_claims([a, b])[2] == []