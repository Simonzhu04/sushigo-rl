"""Tests for deterministic environment behavior and legality constraints."""

from __future__ import annotations

import numpy as np
import pytest

from sushigo_rl import rules
from sushigo_rl.env import EMPTY_HAND_SLOT, PolicyInput, SushiGoEnv


def _always_first_action(_: PolicyInput) -> int:
    return 0


def test_reset_with_same_seed_is_deterministic() -> None:
    env1 = SushiGoEnv()
    env2 = SushiGoEnv()

    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)

    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_equal(info1["action_mask"], info2["action_mask"])
    assert info1["my_hand"] == info2["my_hand"]
    assert info1["opp_hand"] == info2["opp_hand"]


def test_invalid_action_is_rejected() -> None:
    env = SushiGoEnv(opponent_policy=_always_first_action)
    _, info = env.reset(seed=7)

    with pytest.raises(ValueError):
        env.step(len(info["my_hand"]))


def test_episode_length_equals_hand_size() -> None:
    hand_size = rules.HAND_SIZE
    env = SushiGoEnv(hand_size=hand_size, opponent_policy=_always_first_action)

    _, _ = env.reset(seed=0)
    terminated = False
    step_count = 0

    while not terminated:
        _, reward, terminated, truncated, info = env.step(0)
        step_count += 1
        assert truncated is False
        if not terminated:
            assert reward == 0.0

    assert step_count == hand_size
    assert info["turn"] == hand_size


def test_terminal_score_matches_rules_module() -> None:
    env = SushiGoEnv(opponent_policy=_always_first_action)
    _, _ = env.reset(seed=42)

    terminated = False
    reward = 0.0
    info: dict[str, object] = {}

    while not terminated:
        _, reward, terminated, _, info = env.step(0)

    my_played = tuple(info["my_played"])
    opp_played = tuple(info["opp_played"])

    expected_my = rules.score_total(my_played, opp_played)
    expected_opp = rules.score_total(opp_played, my_played)

    assert info["my_score"] == expected_my
    assert info["opp_score"] == expected_opp
    assert reward == pytest.approx(expected_my - expected_opp)


def test_observation_and_mask_dtypes() -> None:
    env = SushiGoEnv(opponent_policy=_always_first_action)
    obs, info = env.reset(seed=11)

    assert obs.dtype == np.float32
    assert info["action_mask"].dtype == np.bool_


def test_observation_shape_is_fixed() -> None:
    env = SushiGoEnv(opponent_policy=_always_first_action)
    obs, _ = env.reset(seed=11)
    expected_size = (len(rules.CARD_TYPES) * 3) + (rules.HAND_SIZE * (len(rules.CARD_TYPES) + 1)) + 5
    assert obs.shape == (expected_size,)


def test_observation_decoding_matches_visible_state() -> None:
    env = SushiGoEnv(opponent_policy=_always_first_action)
    obs, info = env.reset(seed=5)
    decoded = SushiGoEnv.decode_observation(obs)

    my_played_counts = rules.count_cards(tuple(info["my_played"]))
    opp_played_counts = rules.count_cards(tuple(info["opp_played"]))
    my_hand_counts = rules.count_cards(tuple(info["my_hand"]))

    for idx, card in enumerate(rules.CARD_TYPES):
        assert decoded["my_played_counts"][idx] == float(my_played_counts[card])
        assert decoded["opp_played_counts"][idx] == float(opp_played_counts[card])
        assert decoded["my_hand_counts"][idx] == float(my_hand_counts[card])

    expected_slots = tuple(info["my_hand"]) + (EMPTY_HAND_SLOT,) * (rules.HAND_SIZE - len(info["my_hand"]))
    assert decoded["my_hand_slots"] == expected_slots
    assert decoded["turn"] == 0.0
    assert decoded["current_hand_size"] == float(rules.HAND_SIZE)
    assert decoded["my_unpaired_wasabi"] == 0.0
    assert decoded["my_maki_icons"] == 0.0
    assert decoded["opp_maki_icons"] == 0.0


def test_observation_turn_and_hand_size_update_after_step() -> None:
    env = SushiGoEnv(opponent_policy=_always_first_action)
    env.reset(seed=7)
    obs, _, _, _, info = env.step(0)
    decoded = SushiGoEnv.decode_observation(obs)
    assert decoded["turn"] == 1.0
    assert decoded["current_hand_size"] == float(rules.HAND_SIZE - 1)
    expected_slots = tuple(info["my_hand"]) + (EMPTY_HAND_SLOT,) * (rules.HAND_SIZE - len(info["my_hand"]))
    assert decoded["my_hand_slots"] == expected_slots
