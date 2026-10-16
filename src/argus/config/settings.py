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
    ollama_base_url: str = "http://localhost:11434"
    
    llm_provider: Literal["openai", "ollama"] = "openai"
    openai_agent_model: str = "gpt-5.1"
    openai_guardrail_model: str = "gpt-5.4-nano"
    ollama_agent_model: str = "gemma3:4b"
    ollama_guardrail_model: str = "gemma3:4b"

    chunk_size: int = 200
    chunk_overlap: int = 40
    top_k: int = 4

    data_dir: Path = ROOT / "data"
    faiss_index_dir: Path = ROOT / "data" / "processed" / "faiss"
    checkpoint_db: Path = ROOT / "data" / "processed" / "checkpoints.sqlite"


@lru_cache
def get_settings() -> Settings:
    return Settings()