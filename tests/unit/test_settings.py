import pytest
from pydantic import ValidationError

from argus.config.settings import ROOT, Settings


def make() -> Settings:
    return Settings(_env_file=None)  # ignore the real .env


def test_root_is_repo_root():
    assert (ROOT / "pyproject.toml").exists()


def test_defaults(monkeypatch):
    for k in ("LLM_PROVIDER", "OPENAI_API_KEY", "TOP_K", "CHUNK_SIZE", "CHUNK_OVERLAP"):
        monkeypatch.delenv(k, raising=False)
    s = make()
    assert s.llm_provider == "openai"
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (200, 40, 4)
    assert s.openai_api_key is None


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("TOP_K", "7")
    s = make()
    assert s.llm_provider == "ollama"
    assert s.top_k == 7


def test_invalid_provider_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "foo")
    with pytest.raises(ValidationError):
        make()


def test_api_key_is_masked(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    s = make()
    assert "sk-secret" not in repr(s)
    assert s.openai_api_key.get_secret_value() == "sk-secret"


def test_paths_live_under_data():
    s = make()
    assert s.data_dir == ROOT / "data"
    assert s.faiss_index_dir.is_relative_to(s.data_dir)
    

def test_openai_api_key_defaults_to_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert Settings(_env_file=None).openai_api_key is None


def test_openai_api_key_read_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert Settings(_env_file=None).openai_api_key.get_secret_value() == "sk-test"