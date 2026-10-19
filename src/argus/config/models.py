from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from argus.config.settings import Settings, get_settings


class Tier(IntEnum):
    TIER_1 = 1  # small / fast: relevance, language, entities, duplicates
    TIER_2 = 2  # mid: summaries, claims, preliminary risk, contradictions
    TIER_3 = 3  # strong: cross-source reasoning, scenarios, recommendation


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    tier: Tier
    provider: Literal["openai", "ollama"]
    model: str
    input_cost_per_1m: float = 0.0   
    output_cost_per_1m: float = 0.0

# USD per 1M tokens (input, output), from OpenAI's public price list. Check before relying on it.
OPENAI_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.1": (1.25, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.4-nano": (0.20, 1.25),
}


def model_prices(provider: str, model: str) -> tuple[float, float]:
    """(input, output) USD per 1M tokens; local and unknown models cost 0."""
    if provider != "openai":
        return (0.0, 0.0)
    return OPENAI_PRICES.get(model, (0.0, 0.0))

def get_model_spec(tier: Tier, settings: Settings | None = None) -> ModelSpec:
    s = settings or get_settings()
    by_provider = {
        "openai": (s.openai_guardrail_model, s.openai_mid_model, s.openai_agent_model),
        "ollama": (s.ollama_guardrail_model, s.ollama_mid_model, s.ollama_agent_model),
    }
    models = by_provider[s.llm_provider]
    model = models[tier - 1]
    input_price, output_price = model_prices(s.llm_provider, model)
    return ModelSpec(
        tier=tier,
        provider=s.llm_provider,
        model=model,
        input_cost_per_1m=input_price,
        output_cost_per_1m=output_price,
    )


def next_tier(tier: Tier) -> Tier | None:
    """Next tier up for escalation, or None if already at the top."""
    return Tier(tier + 1) if tier < Tier.TIER_3 else None