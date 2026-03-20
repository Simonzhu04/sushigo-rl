"""Provider abstraction for LLM backends (OpenAI, Gemini, fallback)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import random
import time
from typing import Any, Callable

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency optional in tests
    OpenAI = None  # type: ignore[assignment]

try:
    from google import genai
except ImportError:  # pragma: no cover - dependency optional in tests
    genai = None  # type: ignore[assignment]


DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class ProviderResponse:
    """Generated text and whether fallback text was used."""

    text: str
    fallback_used: bool


class BaseLLMProvider(ABC):
    """Common interface used by the assistant for explain/coach generation."""

    def __init__(self, provider_name: str, model_name: str) -> None:
        self.provider_name = provider_name
        self.model_name = model_name

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether provider is configured and able to attempt API calls."""

    @property
    @abstractmethod
    def fallback_mode(self) -> bool:
        """Whether provider is currently operating in fallback mode."""

    @abstractmethod
    def generate_explain(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        cache_key: str | None = None,
    ) -> ProviderResponse:
        """Generate explanation text."""

    @abstractmethod
    def generate_coach(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        cache_key: str | None = None,
    ) -> ProviderResponse:
        """Generate coach text."""


class TemplateFallbackProvider(BaseLLMProvider):
    """Deterministic template fallback provider."""

    def __init__(self) -> None:
        super().__init__(provider_name="fallback", model_name="template")

    @property
    def available(self) -> bool:
        return True

    @property
    def fallback_mode(self) -> bool:
        return True

    def generate_explain(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        cache_key: str | None = None,
    ) -> ProviderResponse:
        del system_prompt, user_prompt, cache_key
        return ProviderResponse(text=fallback_text, fallback_used=True)

    def generate_coach(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        cache_key: str | None = None,
    ) -> ProviderResponse:
        del system_prompt, user_prompt, cache_key
        return ProviderResponse(text=fallback_text, fallback_used=True)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI responses API provider with cache + circuit breaker."""

    def __init__(
        self,
        api_key: str | None,
        model_name: str | None = None,
        max_failures: int = 3,
        client: Any | None = None,
    ) -> None:
        super().__init__(provider_name="openai", model_name=model_name or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
        self._client = client
        if self._client is None and OpenAI is not None and api_key:
            self._client = OpenAI(api_key=api_key)
        self._cache: dict[str, str] = {}
        self._max_failures = max_failures
        self._failure_count = 0
        self._disabled = False

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def fallback_mode(self) -> bool:
        return (not self.available) or self._disabled

    def generate_explain(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        cache_key: str | None = None,
    ) -> ProviderResponse:
        return self._generate(system_prompt, user_prompt, fallback_text, cache_key)

    def generate_coach(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        cache_key: str | None = None,
    ) -> ProviderResponse:
        return self._generate(system_prompt, user_prompt, fallback_text, cache_key)

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        cache_key: str | None,
    ) -> ProviderResponse:
        if cache_key and cache_key in self._cache:
            return ProviderResponse(text=self._cache[cache_key], fallback_used=False)

        if self.fallback_mode:
            reason = "provider unavailable" if not self.available else "circuit breaker open"
            return ProviderResponse(
                text=f"{fallback_text}\n\n(LLM API unavailable, fallback used: {reason})",
                fallback_used=True,
            )

        try:
            response = self._client.responses.create(
                model=self.model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = (getattr(response, "output_text", "") or "").strip()
            if text:
                self._failure_count = 0
                if cache_key:
                    self._cache[cache_key] = text
                return ProviderResponse(text=text, fallback_used=False)
        except Exception as exc:  # pragma: no cover - requires live API/network
            self._failure_count += 1
            if self._failure_count >= self._max_failures:
                self._disabled = True
            return ProviderResponse(
                text=f"{fallback_text}\n\n(LLM API unavailable, fallback used: {type(exc).__name__})",
                fallback_used=True,
            )

        self._failure_count += 1
        if self._failure_count >= self._max_failures:
            self._disabled = True
        return ProviderResponse(text=fallback_text, fallback_used=True)


class GeminiProvider(BaseLLMProvider):
    """Gemini Developer API provider with retry/backoff, cache, and breaker."""

    def __init__(
        self,
        api_key: str | None,
        model_name: str | None = None,
        max_failures: int = 3,
        max_retries: int = 3,
        base_backoff_s: float = 0.3,
        jitter_s: float = 0.2,
        client: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        super().__init__(
            provider_name="gemini",
            model_name=model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        )
        self._client = client
        if self._client is None and genai is not None and api_key:
            self._client = genai.Client(api_key=api_key)
        self._cache: dict[str, str] = {}
        self._max_failures = max_failures
        self._failure_count = 0
        self._disabled = False

        self._max_retries = max_retries
        self._base_backoff_s = base_backoff_s
        self._jitter_s = jitter_s
        self._sleep_fn = sleep_fn
        self._random_fn = random_fn

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def fallback_mode(self) -> bool:
        return (not self.available) or self._disabled

    def generate_explain(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        cache_key: str | None = None,
    ) -> ProviderResponse:
        return self._generate(system_prompt, user_prompt, fallback_text, cache_key)

    def generate_coach(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        cache_key: str | None = None,
    ) -> ProviderResponse:
        return self._generate(system_prompt, user_prompt, fallback_text, cache_key)

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        cache_key: str | None,
    ) -> ProviderResponse:
        if cache_key and cache_key in self._cache:
            return ProviderResponse(text=self._cache[cache_key], fallback_used=False)

        if self.fallback_mode:
            reason = "provider unavailable" if not self.available else "circuit breaker open"
            return ProviderResponse(
                text=f"{fallback_text}\n\n(LLM API unavailable, fallback used: {reason})",
                fallback_used=True,
            )

        prompt = f"{system_prompt}\n\n{user_prompt}"
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                text = (getattr(response, "text", "") or "").strip()
                if text:
                    self._failure_count = 0
                    if cache_key:
                        self._cache[cache_key] = text
                    return ProviderResponse(text=text, fallback_used=False)
                raise RuntimeError("Empty Gemini response text")
            except Exception as exc:  # pragma: no cover - requires live API/network
                last_exc = exc
                if attempt < self._max_retries and self._is_retryable(exc):
                    delay = (self._base_backoff_s * (2 ** attempt)) + (self._random_fn() * self._jitter_s)
                    self._sleep_fn(delay)
                    continue
                break

        self._failure_count += 1
        if self._failure_count >= self._max_failures:
            self._disabled = True
        reason = type(last_exc).__name__ if last_exc is not None else "GeminiError"
        return ProviderResponse(
            text=f"{fallback_text}\n\n(LLM API unavailable, fallback used: {reason})",
            fallback_used=True,
        )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        retry_markers = (
            "429",
            "rate limit",
            "resource exhausted",
            "too many requests",
            "unavailable",
            "deadline exceeded",
            "timeout",
            "temporarily",
        )
        return any(marker in text for marker in retry_markers)


def _get_gemini_api_key(explicit_key: str | None = None) -> str | None:
    if explicit_key:
        return explicit_key
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def create_llm_provider(
    provider_choice: str | None = None,
    openai_api_key: str | None = None,
    gemini_api_key: str | None = None,
    openai_model: str | None = None,
    gemini_model: str | None = None,
) -> BaseLLMProvider:
    """Create provider according to explicit choice or env-based auto-selection."""
    choice = (provider_choice or os.getenv("LLM_PROVIDER", "auto")).strip().lower()

    if choice == "fallback":
        return TemplateFallbackProvider()

    if choice == "openai":
        key = openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY")
        provider = OpenAIProvider(api_key=key, model_name=openai_model)
        return provider if provider.available else TemplateFallbackProvider()

    if choice == "gemini":
        key = _get_gemini_api_key(gemini_api_key)
        provider = GeminiProvider(api_key=key, model_name=gemini_model)
        return provider if provider.available else TemplateFallbackProvider()

    if choice != "auto":
        raise ValueError(f"Unknown provider choice: {choice}")

    openai_key = openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY")
    if openai_key:
        provider = OpenAIProvider(api_key=openai_key, model_name=openai_model)
        if provider.available:
            return provider

    gemini_key = _get_gemini_api_key(gemini_api_key)
    if gemini_key:
        provider = GeminiProvider(api_key=gemini_key, model_name=gemini_model)
        if provider.available:
            return provider

    return TemplateFallbackProvider()


__all__ = [
    "BaseLLMProvider",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "GeminiProvider",
    "OpenAIProvider",
    "ProviderResponse",
    "TemplateFallbackProvider",
    "create_llm_provider",
]
