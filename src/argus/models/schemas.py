from pydantic import BaseModel, ConfigDict, Field

from argus.config.models import Tier


class AttemptRecord(BaseModel):
    """One model call in the cascade: the logbook entry."""

    model_config = ConfigDict(frozen=True)

    tier: Tier
    model: str
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    escalation_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens