import pytest
from pydantic import ValidationError

from argus.config.models import Tier, get_model_spec, next_tier
from argus.config.settings import Settings


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