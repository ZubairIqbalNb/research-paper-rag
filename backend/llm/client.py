"""Answer generation through the Google Gemini API.

The only module in the project that imports ``google.genai``. It is deliberately
text-in / text-out: it knows nothing about chunks, citations, prompts or FAISS,
so retrieval and reranking can be tested against a fake generator and the
provider can be swapped without touching the RAG pipeline.

The API key is read from the environment (see ``backend.core.config``), never
hardcoded, never defaulted, and scrubbed from error messages.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from backend.core.config import ConfigurationError, Settings, get_settings

DEFAULT_TEMPERATURE = 0.0
_MAX_ERROR_CHARS = 300


class LLMError(Exception):
    """Raised when the language model call fails or returns no usable text."""


@runtime_checkable
class AnswerGenerator(Protocol):
    """Minimal prompt -> answer contract shared by the real and fake clients."""

    @property
    def model_name(self) -> str:  # noqa: D102 - protocol property
        ...

    def generate(self, prompt: str) -> str:
        """Return the model's text answer for ``prompt``."""
        ...


class GeminiClient:
    """Gemini answer generator with a lazily created SDK client.

    Args:
        settings: Configuration to use; defaults to the environment settings.
        model_name: Overrides the configured model id.
        client: Pre-built SDK client (anything with ``models.generate_content``);
            injected by tests so no network call or API key is needed.
        temperature: Sampling temperature; 0.0 keeps grounded answers stable.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model_name: str | None = None,
        client: object | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._model_name = model_name or self._settings.gemini_model
        self._temperature = temperature
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_configured(self) -> bool:
        """Whether an SDK client exists or an API key is available for one."""
        return self._client is not None or self._settings.has_gemini_api_key

    def _get_client(self) -> object:
        if self._client is None:
            if not self._settings.has_gemini_api_key:
                raise ConfigurationError(
                    "GEMINI_API_KEY is not set; add it to the environment or to a "
                    "gitignored .env file"
                )
            # Imported lazily and kept here only: the API key never leaves this
            # module and never reaches retrieval, reranking or the prompt layer.
            from google import genai

            self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    def _build_request_config(self) -> object:
        from google.genai import types

        return types.GenerateContentConfig(temperature=self._temperature)

    def generate(self, prompt: str) -> str:
        """Send ``prompt`` to Gemini and return the answer text.

        Raises:
            ValueError: ``prompt`` is blank.
            ConfigurationError: No ``GEMINI_API_KEY`` is configured.
            LLMError: The API call failed or returned no usable text.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt must be a non-empty string")

        client = self._get_client()
        try:
            response = client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=self._build_request_config(),
            )
        except ConfigurationError:
            raise
        except Exception as exc:  # SDK-specific errors, mapped to one type
            raise LLMError(f"Gemini request failed: {self._safe_message(exc)}") from exc

        text = getattr(response, "text", None)
        if not text or not str(text).strip():
            raise LLMError("Gemini returned an empty response (it may have been blocked)")
        return str(text)

    def _safe_message(self, exc: Exception) -> str:
        """Short, key-free error text safe to include in an API response."""
        message = str(exc).strip() or exc.__class__.__name__
        api_key = self._settings.gemini_api_key
        if api_key:
            message = message.replace(api_key, "***")
        return message[:_MAX_ERROR_CHARS]
