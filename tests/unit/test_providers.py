from types import SimpleNamespace

import pytest
import json

from argus.config.models import Tier, get_model_spec
from argus.models.providers import make_caller


class FakeClient:
    """Mimics the OpenAI-style client: client.chat.completions.create(...)."""

    def __init__(self, content, prompt_tokens=11, completion_tokens=7):
        self.last_kwargs = None
        self._reply = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            ),
        )
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._reply


def test_parses_answer_and_confidence():
    client = FakeClient('{"answer": "Apply online", "confidence": 0.85}')
    resp = make_caller(client)(Tier.TIER_1, "How do I apply?")
    assert resp.text == "Apply online"
    assert resp.confidence == pytest.approx(0.85)


def test_invalid_json_gives_zero_confidence():
    client = FakeClient("just some plain text")
    resp = make_caller(client)(Tier.TIER_1, "q")
    assert resp.text == "just some plain text"
    assert resp.confidence == 0.0


def test_json_inside_code_fence_is_parsed():
    client = FakeClient('```json\n{"answer": "ok", "confidence": 0.9}\n```')
    resp = make_caller(client)(Tier.TIER_1, "q")
    assert resp.text == "ok"
    assert resp.confidence == pytest.approx(0.9)


def test_confidence_is_clamped_to_0_1():
    client = FakeClient('{"answer": "x", "confidence": 7}')
    resp = make_caller(client)(Tier.TIER_1, "q")
    assert resp.confidence == 1.0


def test_uses_tier_model_prompt_and_usage():
    client = FakeClient('{"answer": "x", "confidence": 0.5}')
    resp = make_caller(client)(Tier.TIER_2, "my question")
    expected_model = get_model_spec(Tier.TIER_2).model
    assert client.last_kwargs["model"] == expected_model
    assert any(
        m["role"] == "user" and m["content"] == "my question"
        for m in client.last_kwargs["messages"]
    )
    assert resp.model == expected_model
    assert resp.input_tokens == 11
    assert resp.output_tokens == 7
    assert resp.latency_ms >= 0
    
def test_object_answer_is_returned_as_json_text():
    client = FakeClient('{"answer": {"objective": "x"}, "confidence": 0.8}')
    resp = make_caller(client)(Tier.TIER_1, "q")
    assert json.loads(resp.text) == {"objective": "x"}