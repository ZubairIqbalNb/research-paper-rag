"""Environment-backed configuration for Phase 3.

Secrets come from the environment (optionally via a gitignored ``.env``) and are
never defaulted, logged, echoed in errors or committed. The API key is excluded
from ``repr`` so it cannot leak through debug output.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


class ConfigurationError(Exception):
    """Raised when required configuration (e.g. ``GEMINI_API_KEY``) is missing."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from the environment."""

    # repr=False: the key must never appear in logs or tracebacks.
    gemini_api_key: str | None = field(default=None, repr=False)
    gemini_model: str = DEFAULT_GEMINI_MODEL

    @property
    def has_gemini_api_key(self) -> bool:
        """Whether a usable (non-blank) API key is configured."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read settings from the environment, loading a ``.env`` file once.

    Real environment variables take precedence over ``.env`` values, and an
    empty ``GEMINI_MODEL`` falls back to the default model id.
    """
    load_dotenv(override=False)
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip() or None
    model = (os.getenv("GEMINI_MODEL") or "").strip() or DEFAULT_GEMINI_MODEL
    return Settings(gemini_api_key=api_key, gemini_model=model)
