import pytest

from argus.config.models import ModelSpec, Tier
from argus.config.settings import Settings
from argus.guardrails.budgets import ControlBudget

SPEC = ModelSpec(
    tier=Tier.TIER_1, provider="openai", model="x",
    input_cost_per_1m=2.0, output_cost_per_1m=4.0,
)


def new(**overrides) -> ControlBudget:
    limits = dict(max_calls=10, max_tokens=1_000, max_cost_usd=1.0)
    limits.update(overrides)
    return ControlBudget(**limits)


def test_fresh_budget_is_unused_and_not_exhausted():
    b = new()
    assert (b.calls, b.tokens, b.cost_usd) == (0, 0, 0.0)
    assert not b.exhausted
    assert b.exhausted_reason is None


def test_record_counts_calls_and_tokens():
    b = new()
    b.record(SPEC, input_tokens=100, output_tokens=50)
    b.record(SPEC, input_tokens=10, output_tokens=5)
    assert b.calls == 2
    assert b.tokens == 165


def test_record_computes_cost_from_spec_prices():
    b = new(max_tokens=10_000_000, max_cost_usd=100.0)
    b.record(SPEC, input_tokens=500_000, output_tokens=100_000)
    assert b.cost_usd == pytest.approx(1.4)  # 0.5 * $2 + 0.1 * $4


def test_exhausted_by_calls():
    b = new(max_calls=2)
    b.record(SPEC, 1, 1)
    assert not b.exhausted
    b.record(SPEC, 1, 1)
    assert b.exhausted
    assert b.exhausted_reason == "calls"


def test_exhausted_by_tokens():
    b = new(max_tokens=100)
    b.record(SPEC, 60, 40)
    assert b.exhausted_reason == "tokens"


def test_exhausted_by_cost():
    b = new(max_tokens=10_000_000, max_cost_usd=1.0)
    b.record(SPEC, 500_000, 0)  # exactly $1.00
    assert b.exhausted_reason == "cost"


def test_recording_past_the_limit_does_not_raise():
    b = new(max_calls=1)
    b.record(SPEC, 1, 1)
    b.record(SPEC, 1, 1)
    assert b.calls == 2
    assert b.exhausted


def test_negative_tokens_are_rejected():
    with pytest.raises(ValueError):
        new().record(SPEC, -1, 0)


def test_built_from_settings():
    s = Settings(_env_file=None, budget_max_calls=5, budget_max_tokens=123, budget_max_cost_usd=0.5)
    b = ControlBudget.from_settings(s)
    assert (b.max_calls, b.max_tokens, b.max_cost_usd) == (5, 123, 0.5)