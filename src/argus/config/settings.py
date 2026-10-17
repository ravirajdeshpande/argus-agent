from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

ROOT = Path(__file__).resolve().parents[3]  # repo root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    env: str = "dev"
    log_level: str = "INFO"

    openai_api_key: SecretStr | None = None
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    
    llm_provider: Literal["openai", "ollama"] = "openai"
    openai_agent_model: str = "gpt-5.1" 
    openai_guardrail_model: str = "gpt-5.4-nano" # Tier 1 model for guardrails (e.g., safety, content filtering)
    ollama_agent_model: str = "gemma3:4b"  # Tier 3 model for agent tasks (e.g., reasoning, planning)
    ollama_guardrail_model: str = "gemma3:4b"
    openai_mid_model: str = "gpt-5.4-mini"   # Tier 2 model for mid-level tasks (e.g., summarization, classification)
    ollama_mid_model: str = "gemma3:4b"

    chunk_size: int = 200
    chunk_overlap: int = 40
    top_k: int = 4
    
    budget_max_calls: int = 200
    budget_max_tokens: int = 500_000
    budget_max_cost_usd: float = 2.0

    data_dir: Path = ROOT / "data"
    faiss_index_dir: Path = ROOT / "data" / "processed" / "faiss"
    checkpoint_db: Path = ROOT / "data" / "processed" / "checkpoints.sqlite"


@lru_cache
def get_settings() -> Settings:
    return Settings()