"""Tests for LLM assistant state summary and fallback behavior."""

from __future__ import annotations

import json

from sushigo_rl.env import SushiGoEnv
from sushigo_rl.llm_assistant import LLMAssistant
from sushigo_rl.llm_providers import ProviderResponse


def test_llm_summary_json_serializable() -> None:
    env = SushiGoEnv()
    _, info = env.reset(seed=123)

    assistant = LLMAssistant(policy_advisor=None, api_key="")
    summary = assistant.build_state_summary(env=env, info=info, topk=3)

    required_keys = {
        "turn",
        "cards_left",
        "my_played_counts",
        "opp_played_counts",
        "my_maki_icons",
        "opp_maki_icons",
        "my_unpaired_wasabi",
        "current_hand",
        "action_mask_indices",
        "agent_recommendation",
    }
    assert required_keys.issubset(summary.keys())
    json.dumps(summary)

    env.close()


def test_llm_fallback_nonempty() -> None:
    env = SushiGoEnv()
    _, info = env.reset(seed=7)

    assistant = LLMAssistant(policy_advisor=None, api_key="")
    summary = assistant.build_state_summary(env=env, info=info, topk=3)

    coach_text = assistant.coach_user(summary, topk=3)
    explain_text = assistant.explain_agent_move(summary, chosen_action=summary["action_mask_indices"][0], topk=3)

    assert isinstance(coach_text, str) and coach_text.strip()
    assert isinstance(explain_text, str) and explain_text.strip()

    env.close()


def test_coach_user_rejects_invented_move_and_uses_topk_only() -> None:
    class MockProvider:
        provider_name = "mock"
        model_name = "mock-model"
        fallback_mode = False

        def generate_explain(self, *args, **kwargs) -> ProviderResponse:
            del args, kwargs
            return ProviderResponse(text="explain", fallback_used=False)

        def generate_coach(self, *args, **kwargs) -> ProviderResponse:
            del args, kwargs
            return ProviderResponse(
                text='{"recommendations":[{"action_index":9,"reason":"invented move not in topk","tradeoff":"bad"}]}',
                fallback_used=False,
            )

    summary = {
        "turn": 0,
        "cards_left": 3,
        "my_played_counts": {},
        "opp_played_counts": {},
        "my_maki_icons": 1,
        "opp_maki_icons": 2,
        "my_unpaired_wasabi": 0,
        "current_hand": ["tempura", "sashimi", "maki3"],
        "action_mask_indices": [0, 1, 2],
        "agent_recommendation": [
            {"action_index": 0, "card": "tempura", "probability": 0.7},
            {"action_index": 1, "card": "sashimi", "probability": 0.2},
            {"action_index": 2, "card": "maki3", "probability": 0.1},
        ],
    }
    assistant = LLMAssistant(policy_advisor=None, provider=MockProvider())
    text = assistant.coach_user(summary, topk=3)

    assert "idx 9" not in text
    assert text.startswith("Top moves:")
    assert "idx 0" in text and "tempura" in text and "p=0.700" in text
    assert "idx 1" in text and "sashimi" in text and "p=0.200" in text
    assert "idx 2" in text and "maki3" in text and "p=0.100" in text


def test_coach_user_renders_structured_provider_output() -> None:
    class MockProvider:
        provider_name = "mock"
        model_name = "mock-model"
        fallback_mode = False

        def generate_explain(self, *args, **kwargs) -> ProviderResponse:
            del args, kwargs
            return ProviderResponse(text="explain", fallback_used=False)

        def generate_coach(self, *args, **kwargs) -> ProviderResponse:
            del args, kwargs
            return ProviderResponse(
                text=(
                    '{"recommendations":['
                    '{"action_index":0,"reason":"keeps tempura pair equity live","tradeoff":"you pass on immediate maki pressure"},'
                    '{"action_index":1,"reason":"adds sashimi progress toward three","tradeoff":"it scores later than tempura"},'
                    '{"action_index":2,"reason":"pushes the maki race now","tradeoff":"it does not help your set-building"}'
                    "]} "
                ),
                fallback_used=False,
            )

    summary = {
        "turn": 0,
        "round_idx": 1,
        "num_rounds": 3,
        "hand_size": 10,
        "cards_left": 3,
        "my_played_counts": {},
        "opp_played_counts": {},
        "my_maki_icons": 1,
        "opp_maki_icons": 2,
        "my_unpaired_wasabi": 0,
        "my_pudding_count": 1,
        "opp_pudding_count": 0,
        "my_total_score": 10.0,
        "opp_total_score": 7.0,
        "current_hand": ["tempura", "sashimi", "maki3"],
        "action_mask_indices": [0, 1, 2],
        "agent_recommendation": [
            {"action_index": 0, "card": "tempura", "action_label": "idx 0 -> tempura", "probability": 0.7},
            {"action_index": 1, "card": "sashimi", "action_label": "idx 1 -> sashimi", "probability": 0.2},
            {"action_index": 2, "card": "maki3", "action_label": "idx 2 -> maki3", "probability": 0.1},
        ],
    }
    assistant = LLMAssistant(policy_advisor=None, provider=MockProvider())
    text = assistant.coach_user(summary, topk=3)

    assert text.startswith("Top moves:")
    assert "idx 0 -> tempura (p=0.700): keeps tempura pair equity live." in text
    assert "Tradeoff: you pass on immediate maki pressure." in text
    assert "idx 1 -> sashimi (p=0.200): adds sashimi progress toward three." in text
    assert "idx 2 -> maki3 (p=0.100): pushes the maki race now." in text
