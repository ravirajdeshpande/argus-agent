from types import SimpleNamespace

from argus.evaluation.metrics import (
    contradiction_handled,
    grounding_rate,
    retrieval_scores,
    run_stats,
    text_checks,
    unresolved_conflicts,
    unsupported_entities,
)
from argus.graph.state import initial_state
from argus.guardrails.evidence import Claim


def _claim(text, sources):
    return Claim(
        claim=text, claim_type="effective_date", subject="s", value="v", sources=sources,
        primary_source=True, source_authority=0.9, recency_score=0.5,
        corroboration_score=0.5, contradiction=False, confidence=0.7,
    )


def make_state(**over):
    state = initial_state("q")
    state["government_sources"] = [{"source_id": "gov_1", "result": {"summary": "Rule applies to ABC Semiconductor"}}]
    state["internal_documents"] = [
        {"source_id": "internal_1", "text": "Contract with ABC Semiconductor for Product Alpha",
         "metadata": {"source": "contract_c104.txt"}},
        {"source_id": "internal_2", "text": "Board memo", "metadata": {"source": "memo.txt"}},
    ]
    state.update(over)
    return state


def test_grounding_counts_claims_and_actions_with_valid_sources():
    state = make_state(
        verified_claims=[_claim("a", ["gov_1"]), _claim("b", ["internal_9"])],
        recommended_actions=[{"sources": ["internal_1"]}, {"sources": []}],
    )
    assert grounding_rate(state) == 0.5


def test_grounding_is_one_when_nothing_is_asserted():
    assert grounding_rate(make_state()) == 1.0


def test_unsupported_entities_are_names_absent_from_all_evidence():
    state = make_state(impacted_suppliers=["ABC Semiconductor", "Zeta Chips"], impacted_products=["Product Alpha"])
    assert unsupported_entities(state) == ["Zeta Chips"]


def test_retrieval_precision_and_recall():
    scores = retrieval_scores(make_state(), {"contract_c104.txt", "procurement.txt"})
    assert scores == {"precision": 0.5, "recall": 0.5}


def test_retrieval_with_nothing_retrieved_scores_zero():
    state = make_state(internal_documents=[])
    assert retrieval_scores(state, {"a.txt"}) == {"precision": 0.0, "recall": 0.0}


def conflict(resolved=True, winning="2027-01-15", rejected=("old claim",)):
    return {"claim_type": "effective_date", "resolved": resolved, "winning_value": winning,
            "rejected_claims": list(rejected)}


def test_contradiction_handled_when_primary_value_wins():
    state = make_state(conflicting_claims=[conflict()], verified_claims=[_claim("good", ["gov_1"])])
    assert contradiction_handled(state, "effective_date", "2027-01-15")


def test_contradiction_not_handled_when_unresolved_or_wrong_winner_or_missing():
    assert not contradiction_handled(make_state(conflicting_claims=[conflict(resolved=False)]), "effective_date", "2027-01-15")
    assert not contradiction_handled(make_state(conflicting_claims=[conflict(winning="2026-01-01")]), "effective_date", "2027-01-15")
    assert not contradiction_handled(make_state(), "effective_date", "2027-01-15")


def test_contradiction_not_handled_if_a_rejected_claim_is_still_verified():
    state = make_state(conflicting_claims=[conflict()], verified_claims=[_claim("old claim", ["gov_1"])])
    assert not contradiction_handled(state, "effective_date", "2027-01-15")


def test_unresolved_conflicts_counts_only_open_ones():
    state = make_state(conflicting_claims=[conflict(), conflict(resolved=False)])
    assert unresolved_conflicts(state) == 1


def test_run_stats_reads_budget_and_tolerates_missing_fields():
    budget = SimpleNamespace(calls=7, tokens=1234, cost_usd=0.05)
    assert run_stats(budget, 3.456) == {"elapsed_s": 3.46, "calls": 7, "tokens": 1234, "cost_usd": 0.05}
    assert run_stats(SimpleNamespace(), 1.0)["calls"] is None


def test_text_checks_report_failures_case_insensitively():
    report = "Controls take effect January 15, 2027"
    assert text_checks(report, ["january 15, 2027"], ["immediately"]) == []
    failures = text_checks(report, ["45 days"], ["JANUARY"])
    assert failures == ["missing: 45 days", "forbidden: JANUARY"]
    
def test_no_conflict_is_fine_when_the_right_value_is_verified():
    right = _claim("good", ["gov_1"]).model_copy(update={"value": "January 15, 2027"})
    assert contradiction_handled(make_state(verified_claims=[right]), "effective_date", "2027-01-15")