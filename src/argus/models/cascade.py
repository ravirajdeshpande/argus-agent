from collections.abc import Callable
from dataclasses import dataclass

from argus.config.models import Tier, get_model_spec
from argus.guardrails.budgets import ControlBudget
from argus.models.schemas import AttemptRecord


class BudgetExhaustedError(RuntimeError):
    """Raised when the control budget is used up before a model call."""


@dataclass(frozen=True)
class ModelResponse:
    text: str
    confidence: float
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class CascadeResult:
    text: str
    confidence: float
    attempts: list[AttemptRecord]


ModelCaller = Callable[[Tier, str], ModelResponse]


def run_cascade(
    prompt: str,
    call: ModelCaller,
    budget: ControlBudget,
    *,
    confidence_threshold: float = 0.7,
    high_impact: bool = False,
    start_tier: Tier | None = None,
) -> CascadeResult:
    if start_tier is not None:
        start = start_tier
    else:
        start = Tier.TIER_3 if high_impact else Tier.TIER_1
    tiers = [t for t in Tier if t >= start]
    attempts: list[AttemptRecord] = []

    for tier in tiers:
        if budget.exhausted:
            raise BudgetExhaustedError(budget.exhausted_reason)

        resp = call(tier, prompt)

        cost_before = budget.cost_usd
        budget.record(get_model_spec(tier), resp.input_tokens, resp.output_tokens)
        call_cost = budget.cost_usd - cost_before

        is_last = tier == tiers[-1]
        escalate = resp.confidence < confidence_threshold and not is_last
        reason = (
            f"confidence {resp.confidence:.2f} below threshold {confidence_threshold:.2f}"
            if escalate
            else None
        )
        attempts.append(
            AttemptRecord(
                tier=tier,
                model=resp.model,
                latency_ms=resp.latency_ms,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost_usd=call_cost,
                escalation_reason=reason,
            )
        )
        if not escalate:
            return CascadeResult(resp.text, resp.confidence, attempts)

    raise RuntimeError("unreachable: last tier always returns")