"""Gemini client tests, plus the settings it depends on.

The SDK is never called: an injected fake client stands in, so the suite runs
without an API key or network access.
"""
import pathlib

import pytest

from backend.core.config import (
    DEFAULT_GEMINI_MODEL,
    ConfigurationError,
    Settings,
    get_settings,
)
from backend.llm.client import LLMError, GeminiClient

CLIENT_SOURCE = pathlib.Path("backend/llm/client.py").read_text(encoding="utf-8")


class _FakeResponse:
    def __init__(self, text) -> None:
        self.text = text


class _FakeModels:
    """Records generate_content calls; optionally raises or returns a response."""

    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def generate_content(self, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error is not None:
            raise self.error
        return self.response


class _FakeSdkClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.models = _FakeModels(response=response, error=error)


@pytest.fixture
def configured_settings() -> Settings:
    return Settings(gemini_api_key="test-key-not-real", gemini_model="gemini-test-model")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Keep the lru_cache'd settings from leaking between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_generate_returns_the_model_text(configured_settings):
    sdk = _FakeSdkClient(response=_FakeResponse("nDCG measures ranking quality [1]."))
    client = GeminiClient(settings=configured_settings, client=sdk)

    assert client.generate("a prompt") == "nDCG measures ranking quality [1]."


def test_generate_sends_configured_model_prompt_and_zero_temperature(configured_settings):
    sdk = _FakeSdkClient(response=_FakeResponse("answer"))
    client = GeminiClient(settings=configured_settings, client=sdk)

    client.generate("  the grounded prompt  ")

    call = sdk.models.calls[0]
    assert call["model"] == "gemini-test-model"
    assert call["contents"] == "the grounded prompt"
    assert call["config"].temperature == 0.0


def test_model_name_falls_back_to_env_default():
    client = GeminiClient(settings=Settings(gemini_api_key="k"), client=_FakeSdkClient())

    assert client.model_name == DEFAULT_GEMINI_MODEL


def test_generate_without_api_key_raises_configuration_error():
    client = GeminiClient(settings=Settings(gemini_api_key=None))

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        client.generate("a prompt")
    assert client.is_configured is False


def test_blank_api_key_is_treated_as_missing():
    assert Settings(gemini_api_key="   ").has_gemini_api_key is False
    assert Settings(gemini_api_key="abc").has_gemini_api_key is True


def test_generate_wraps_sdk_failures_as_llm_error(configured_settings):
    sdk = _FakeSdkClient(error=RuntimeError("connection reset"))
    client = GeminiClient(settings=configured_settings, client=sdk)

    with pytest.raises(LLMError, match="Gemini request failed: connection reset"):
        client.generate("a prompt")


def test_generate_raises_on_empty_or_missing_text(configured_settings):
    empty = GeminiClient(settings=configured_settings, client=_FakeSdkClient(_FakeResponse("   ")))
    missing = GeminiClient(settings=configured_settings, client=_FakeSdkClient(_FakeResponse(None)))

    with pytest.raises(LLMError, match="empty response"):
        empty.generate("a prompt")
    with pytest.raises(LLMError, match="empty response"):
        missing.generate("a prompt")


def test_error_messages_never_contain_the_api_key():
    settings = Settings(gemini_api_key="super-secret-key-123")
    sdk = _FakeSdkClient(error=RuntimeError("auth failed for key super-secret-key-123"))
    client = GeminiClient(settings=settings, client=sdk)

    with pytest.raises(LLMError) as excinfo:
        client.generate("a prompt")

    assert "super-secret-key-123" not in str(excinfo.value)
    assert "***" in str(excinfo.value)


def test_generate_rejects_a_blank_prompt(configured_settings):
    client = GeminiClient(settings=configured_settings, client=_FakeSdkClient())

    with pytest.raises(ValueError, match="non-empty"):
        client.generate("   ")


def test_settings_repr_hides_the_api_key():
    assert "super-secret-key-123" not in repr(Settings(gemini_api_key="super-secret-key-123"))


def test_get_settings_reads_model_and_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-from-env")

    settings = get_settings()

    assert settings.gemini_model == "gemini-from-env"
    assert settings.has_gemini_api_key is True


def test_get_settings_uses_the_default_model_when_env_is_blank(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.setenv("GEMINI_MODEL", "  ")

    assert get_settings().gemini_model == DEFAULT_GEMINI_MODEL


def test_llm_module_is_isolated_from_retrieval_and_reranking():
    # The Gemini client must stay text-in/text-out, so the provider can be
    # swapped (or faked in tests) without touching the RAG pipeline.
    assert "backend.retrieval" not in CLIENT_SOURCE
    assert "backend.reranking" not in CLIENT_SOURCE
    assert "faiss" not in CLIENT_SOURCE
