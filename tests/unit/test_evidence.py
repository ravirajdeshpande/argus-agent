import pytest
from pydantic import ValidationError

from argus.guardrails.evidence import Claim, material_unresolved_conflicts, next_step


def make_claim(**overrides):
    data = dict(
        claim="Restrictions begin September 1",
        claim_type="effective_date",
        sources=["source_17", "source_31", "gov_04"],
        primary_source=True,
        source_authority=0.96,
        recency_score=0.99,
        corroboration_score=0.91,
        contradiction=False,
        confidence=0.94,
    )
    data.update(overrides)
    return Claim(**data)


def good_state(**overrides):
    state = {
        "investigation_iterations": 1,
        "confidence_score": 0.9,
        "missing_information": [],
        "claims": [make_claim()],
    }
    state.update(overrides)
    return state


def test_claim_holds_all_fields():
    c = make_claim()
    assert c.claim_type == "effective_date"
    assert c.sources == ["source_17", "source_31", "gov_04"]
    assert c.confidence == pytest.approx(0.94)


@pytest.mark.parametrize(
    "field", ["source_authority", "recency_score", "corroboration_score", "confidence"]
)
def test_claim_scores_must_be_between_0_and_1(field):
    with pytest.raises(ValidationError):
        make_claim(**{field: 1.5})
    with pytest.raises(ValidationError):
        make_claim(**{field: -0.1})


def test_claim_is_immutable():
    with pytest.raises(ValidationError):
        make_claim().confidence = 0.1


def test_sufficient_evidence_goes_to_risk_assessment():
    assert next_step(good_state()) == "risk_assessment"


def test_max_iterations_forces_uncertain_assessment_even_if_low_confidence():
    state = good_state(investigation_iterations=5, confidence_score=0.1)
    assert next_step(state, max_iterations=5) == "risk_assessment_with_uncertainty"


def test_low_confidence_continues_research():
    assert next_step(good_state(confidence_score=0.74)) == "continue_research"


def test_confidence_exactly_at_threshold_is_enough():
    assert next_step(good_state(confidence_score=0.75)) == "risk_assessment"


def test_threshold_is_configurable():
    state = good_state(confidence_score=0.8)
    assert next_step(state, confidence_threshold=0.9) == "continue_research"


def test_missing_information_continues_research():
    state = good_state(missing_information=["effective date of phase 2"])
    assert next_step(state) == "continue_research"


def test_unresolved_conflict_continues_research():
    state = good_state(claims=[make_claim(contradiction=True)])
    assert material_unresolved_conflicts(state)
    assert next_step(state) == "continue_research"


def test_no_conflicts_when_claims_agree_or_absent():
    assert not material_unresolved_conflicts(good_state())
    assert not material_unresolved_conflicts({})
    
def test_claim_subject_and_value_default_to_empty_and_can_be_set():
    assert make_claim().subject == ""
    c = make_claim(subject="export restriction", value="September 1")
    assert (c.subject, c.value) == ("export restriction", "September 1")