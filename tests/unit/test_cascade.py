import pytest

from argus.config.models import Tier
from argus.guardrails.budgets import ControlBudget
from argus.models.cascade import BudgetExhaustedError, ModelResponse, run_cascade


def make_budget(max_calls=10):
    return ControlBudget(max_calls=max_calls, max_tokens=1_000_000, max_cost_usd=100.0)


def make_caller(confidences):
    calls = []

    def call(tier, prompt):
        calls.append(tier)
        return ModelResponse(
            text=f"answer from {tier.name}",
            confidence=confidences[tier],
            model=f"model-{tier.value}",
            input_tokens=10,
            output_tokens=5,
            latency_ms=12.0,
        )

    call.calls = calls
    return call


def test_confident_tier1_answer_stops_immediately():
    call = make_caller({Tier.TIER_1: 0.9})
    result = run_cascade("q", call, make_budget())
    assert call.calls == [Tier.TIER_1]
    assert len(result.attempts) == 1
    assert result.attempts[0].escalation_reason is None


def test_low_confidence_escalates_with_reason():
    call = make_caller({Tier.TIER_1: 0.3, Tier.TIER_2: 0.9})
    result = run_cascade("q", call, make_budget())
    assert call.calls == [Tier.TIER_1, Tier.TIER_2]
    assert "0.30" in result.attempts[0].escalation_reason
    assert result.attempts[1].escalation_reason is None
    assert result.text == "answer from TIER_2"


def test_high_impact_starts_at_tier3():
    call = make_caller({Tier.TIER_3: 0.9})
    result = run_cascade("q", call, make_budget(), high_impact=True)
    assert call.calls == [Tier.TIER_3]
    assert result.attempts[0].tier == Tier.TIER_3


def test_top_tier_low_confidence_returns_anyway():
    call = make_caller({Tier.TIER_1: 0.1, Tier.TIER_2: 0.1, Tier.TIER_3: 0.1})
    result = run_cascade("q", call, make_budget())
    assert call.calls == [Tier.TIER_1, Tier.TIER_2, Tier.TIER_3]
    assert result.attempts[-1].escalation_reason is None


def test_budget_exhausted_blocks_call():
    call = make_caller({Tier.TIER_1: 0.9})
    with pytest.raises(BudgetExhaustedError):
        run_cascade("q", call, make_budget(max_calls=0))
    assert call.calls == []


def test_budget_tracks_calls_tokens_and_cost():
    call = make_caller({Tier.TIER_1: 0.3, Tier.TIER_2: 0.9})
    budget = make_budget()
    result = run_cascade("q", call, budget)
    assert budget.calls == 2
    assert budget.tokens == 30
    assert sum(a.cost_usd for a in result.attempts) == pytest.approx(budget.cost_usd)

def test_start_tier_overrides_default_start():
    call = make_caller({Tier.TIER_2: 0.9})
    run_cascade("q", call, make_budget(), start_tier=Tier.TIER_2)
    assert call.calls == [Tier.TIER_2]


def test_start_tier_still_escalates_on_low_confidence():
    call = make_caller({Tier.TIER_2: 0.1, Tier.TIER_3: 0.9})
    run_cascade("q", call, make_budget(), start_tier=Tier.TIER_2)
    assert call.calls == [Tier.TIER_2, Tier.TIER_3]