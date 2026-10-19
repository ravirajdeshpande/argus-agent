from dataclasses import dataclass

from argus.config.models import ModelSpec
from argus.config.settings import Settings


@dataclass
class ControlBudget:
    # limits (required when creating the object)
    max_calls: int
    max_tokens: int
    max_cost_usd: float
    # usage so far (start at zero)
    calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0

    @classmethod
    def from_settings(cls, settings: Settings) -> "ControlBudget":
        return cls(
            max_calls=settings.budget_max_calls,
            max_tokens=settings.budget_max_tokens,
            max_cost_usd=settings.budget_max_cost_usd,
        )

    def record(self, spec: ModelSpec, input_tokens: int, output_tokens: int) -> None:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be >= 0")
        self.calls += 1
        self.tokens += input_tokens + output_tokens
        self.cost_usd += (
            input_tokens * spec.input_cost_per_1m
            + output_tokens * spec.output_cost_per_1m
        ) / 1_000_000

    @property
    def exhausted_reason(self) -> str | None:
        if self.calls >= self.max_calls:
            return "calls"
        if self.tokens >= self.max_tokens:
            return "tokens"
        if self.cost_usd >= self.max_cost_usd:
            return "cost"
        return None

    @property
    def exhausted(self) -> bool:
        return self.exhausted_reason is not None