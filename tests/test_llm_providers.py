"""Tests for LLM provider selection and Gemini provider behavior."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from sushigo_rl import llm_providers
from sushigo_rl.llm_providers import GeminiProvider, ProviderResponse, create_llm_provider


def test_provider_selection_auto_fallback_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    provider = create_llm_provider(provider_choice="auto")
    assert provider.provider_name == "fallback"
    assert provider.fallback_mode is True


def test_provider_selection_respects_env_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    @dataclass
    class FakeGemini:
        provider_name: str = "gemini"
        model_name: str = "fake-gemini-model"
        available: bool = True
        fallback_mode: bool = False

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def generate_explain(self, *args, **kwargs) -> ProviderResponse:
            del args, kwargs
            return ProviderResponse(text="ok", fallback_used=False)

        def generate_coach(self, *args, **kwargs) -> ProviderResponse:
            del args, kwargs
            return ProviderResponse(text="ok", fallback_used=False)

    monkeypatch.setattr(llm_providers, "GeminiProvider", FakeGemini)
    provider = create_llm_provider(provider_choice=None)
    assert provider.provider_name == "gemini"
    assert provider.fallback_mode is False


def test_provider_selection_auto_prefers_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    calls: list[str] = []

    @dataclass
    class FakeOpenAI:
        provider_name: str = "openai"
        model_name: str = "fake-openai-model"
        available: bool = True
        fallback_mode: bool = False

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            calls.append("openai")

        def generate_explain(self, *args, **kwargs) -> ProviderResponse:
            del args, kwargs
            return ProviderResponse(text="ok", fallback_used=False)

        def generate_coach(self, *args, **kwargs) -> ProviderResponse:
            del args, kwargs
            return ProviderResponse(text="ok", fallback_used=False)

    @dataclass
    class FakeGemini:
        provider_name: str = "gemini"
        model_name: str = "fake-gemini-model"
        available: bool = True
        fallback_mode: bool = False

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            calls.append("gemini")

        def generate_explain(self, *args, **kwargs) -> ProviderResponse:
            del args, kwargs
            return ProviderResponse(text="ok", fallback_used=False)

        def generate_coach(self, *args, **kwargs) -> ProviderResponse:
            del args, kwargs
            return ProviderResponse(text="ok", fallback_used=False)

    monkeypatch.setattr(llm_providers, "OpenAIProvider", FakeOpenAI)
    monkeypatch.setattr(llm_providers, "GeminiProvider", FakeGemini)

    provider = create_llm_provider(provider_choice="auto")
    assert provider.provider_name == "openai"
    assert calls == ["openai"]


def test_gemini_provider_mock_client_uses_cache() -> None:
    class MockModels:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(self, *, model: str, contents: str):
            del model, contents
            self.calls += 1

            class Resp:
                text = "mocked gemini response"

            return Resp()

    class MockClient:
        def __init__(self) -> None:
            self.models = MockModels()

    client = MockClient()
    provider = GeminiProvider(api_key="fake", client=client)

    r1 = provider.generate_explain("sys", "user", "fallback", cache_key="key1")
    r2 = provider.generate_explain("sys", "user", "fallback", cache_key="key1")
    assert r1.fallback_used is False
    assert r2.fallback_used is False
    assert r1.text == "mocked gemini response"
    assert client.models.calls == 1


def test_gemini_provider_retries_then_succeeds() -> None:
    class MockModels:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(self, *, model: str, contents: str):
            del model, contents
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("429 resource exhausted")

            class Resp:
                text = "ok after retry"

            return Resp()

    class MockClient:
        def __init__(self) -> None:
            self.models = MockModels()

    sleeps: list[float] = []
    provider = GeminiProvider(
        api_key="fake",
        client=MockClient(),
        sleep_fn=lambda delay: sleeps.append(delay),
        random_fn=lambda: 0.0,
    )

    result = provider.generate_coach("sys", "user", "fallback", cache_key="k2")
    assert result.fallback_used is False
    assert result.text == "ok after retry"
    assert len(sleeps) == 1
