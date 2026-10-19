import pytest
from pydantic import ValidationError

from argus.config.models import Tier
from argus.models.schemas import AttemptRecord


def test_attempt_record_holds_all_fields():
    rec = AttemptRecord(
        tier=Tier.TIER_1,
        model="gpt-4o-mini",
        latency_ms=120.5,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.0004,
        escalation_reason=None,
    )
    assert rec.tier == Tier.TIER_1
    assert rec.total_tokens == 150


def test_attempt_record_is_immutable():
    rec = AttemptRecord(
        tier=Tier.TIER_1, model="m", latency_ms=1.0,
        input_tokens=1, output_tokens=1, cost_usd=0.0,
    )
    with pytest.raises(ValidationError):
        rec.model = "other"


def test_negative_values_rejected():
    with pytest.raises(ValidationError):
        AttemptRecord(
            tier=Tier.TIER_1, model="m", latency_ms=-1.0,
            input_tokens=1, output_tokens=1, cost_usd=0.0,
        )