import json
import time

from argus.config.models import Tier, get_model_spec
from argus.config.settings import Settings, get_settings
from argus.models.cascade import ModelCaller, ModelResponse

SYSTEM_PROMPT = (
    "Answer the user's question. Reply ONLY with JSON in this form: "
    '{"answer": "<your answer>", "confidence": <number between 0 and 1>}'
)


def _parse(content: str) -> tuple[str, float]:
    """Extract (answer, confidence). Unparseable output -> confidence 0.0."""
    start, end = content.find("{"), content.rfind("}")
    try:
        data = json.loads(content[start : end + 1])
        answer = data["answer"]
        if not isinstance(answer, str):
            answer = json.dumps(answer)
        confidence = float(data["confidence"])
    except (ValueError, KeyError, TypeError):
        return content, 0.0
    return answer, min(max(confidence, 0.0), 1.0)


def make_caller(client, settings: Settings | None = None) -> ModelCaller:
    """Build a cascade caller around any OpenAI-compatible client."""

    def call(tier: Tier, prompt: str) -> ModelResponse:
        model = get_model_spec(tier, settings).model
        started = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        latency_ms = (time.perf_counter() - started) * 1000
        text, confidence = _parse(resp.choices[0].message.content or "")
        return ModelResponse(
            text=text,
            confidence=confidence,
            model=model,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_ms=latency_ms,
        )

    return call


def build_client(settings: Settings | None = None):
    """Real client: OpenAI, or Ollama through its OpenAI-compatible endpoint."""
    from openai import OpenAI

    s = settings or get_settings()
    if s.llm_provider == "ollama":
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return OpenAI(api_key=s.openai_api_key or None)