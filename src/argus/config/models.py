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
    input_cost_per_1m: float = 0.0   # ??USD per 1M tokens; 
    output_cost_per_1m: float = 0.0


def get_model_spec(tier: Tier, settings: Settings | None = None) -> ModelSpec:
    s = settings or get_settings()
    by_provider = {
        "openai": (s.openai_guardrail_model, s.openai_mid_model, s.openai_agent_model),
        "ollama": (s.ollama_guardrail_model, s.ollama_mid_model, s.ollama_agent_model),
    }
    models = by_provider[s.llm_provider]
    return ModelSpec(tier=tier, provider=s.llm_provider, model=models[tier - 1])


def next_tier(tier: Tier) -> Tier | None:
    """Next tier up for escalation, or None if already at the top."""
    return Tier(tier + 1) if tier < Tier.TIER_3 else None