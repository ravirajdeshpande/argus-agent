import pytest
from pydantic import ValidationError

from argus.config.models import Tier, get_model_spec, next_tier
from argus.config.settings import Settings
from argus.guardrails.budgets import ControlBudget


def cfg(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_tiers_are_ordered():
    assert (Tier.TIER_1, Tier.TIER_2, Tier.TIER_3) == (1, 2, 3)
    assert Tier.TIER_1 < Tier.TIER_2 < Tier.TIER_3


def test_openai_tier_mapping():
    s = cfg(llm_provider="openai", openai_guardrail_model="g",
            openai_mid_model="m", openai_agent_model="a")
    assert get_model_spec(Tier.TIER_1, s).model == "g"
    assert get_model_spec(Tier.TIER_2, s).model == "m"
    assert get_model_spec(Tier.TIER_3, s).model == "a"


def test_ollama_tier_mapping():
    s = cfg(llm_provider="ollama", ollama_guardrail_model="g",
            ollama_mid_model="m", ollama_agent_model="a")
    assert [get_model_spec(t, s).model for t in Tier] == ["g", "m", "a"]


def test_spec_carries_tier_and_provider():
    spec = get_model_spec(Tier.TIER_2, cfg(llm_provider="ollama"))
    assert spec.tier == Tier.TIER_2
    assert spec.provider == "ollama"


def test_spec_prices_default_to_non_negative():
    spec = get_model_spec(Tier.TIER_3, cfg())
    assert spec.input_cost_per_1m >= 0
    assert spec.output_cost_per_1m >= 0


def test_spec_is_immutable():
    spec = get_model_spec(Tier.TIER_1, cfg())
    with pytest.raises(ValidationError):
        spec.model = "other"


def test_next_tier_escalates_and_stops_at_top():
    assert next_tier(Tier.TIER_1) == Tier.TIER_2
    assert next_tier(Tier.TIER_2) == Tier.TIER_3
    assert next_tier(Tier.TIER_3) is None
    
def _settings(**over):
    return Settings(_env_file=None, **over)


def test_known_openai_models_carry_their_prices():
    s = _settings(llm_provider="openai", openai_guardrail_model="gpt-4.1-nano",
                  openai_mid_model="gpt-4.1-mini", openai_agent_model="gpt-4.1")
    assert (get_model_spec(Tier.TIER_1, s).input_cost_per_1m, get_model_spec(Tier.TIER_1, s).output_cost_per_1m) == (0.10, 0.40)
    assert (get_model_spec(Tier.TIER_2, s).input_cost_per_1m, get_model_spec(Tier.TIER_2, s).output_cost_per_1m) == (0.40, 1.60)
    assert (get_model_spec(Tier.TIER_3, s).input_cost_per_1m, get_model_spec(Tier.TIER_3, s).output_cost_per_1m) == (2.00, 8.00)


def test_unknown_openai_model_is_priced_at_zero():
    spec = get_model_spec(Tier.TIER_3, _settings(llm_provider="openai", openai_agent_model="some-new-model"))
    assert (spec.input_cost_per_1m, spec.output_cost_per_1m) == (0.0, 0.0)


def test_ollama_models_are_free_even_if_the_name_matches_an_openai_model():
    spec = get_model_spec(Tier.TIER_3, _settings(llm_provider="ollama", ollama_agent_model="gpt-4.1"))
    assert (spec.input_cost_per_1m, spec.output_cost_per_1m) == (0.0, 0.0)


def test_budget_cost_reflects_the_model_price():
    spec = get_model_spec(Tier.TIER_3, _settings(llm_provider="openai", openai_agent_model="gpt-4.1"))
    budget = ControlBudget(max_calls=10, max_tokens=10_000_000, max_cost_usd=100.0)
    budget.record(spec, 1_000_000, 1_000_000)
    assert budget.cost_usd == pytest.approx(10.0)