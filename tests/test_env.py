"""Tests for deterministic multi-round environment behavior."""

from __future__ import annotations

import numpy as np
import pytest

from sushigo_rl import rules
from sushigo_rl.env import EMPTY_HAND_SLOT, GameState, PolicyInput, SushiGoEnv


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
    assert info1["round_idx"] == info2["round_idx"] == 0


def test_invalid_action_is_rejected() -> None:
    env = SushiGoEnv(opponent_policy=_always_first_action)
    _, info = env.reset(seed=7)

    with pytest.raises(ValueError):
        env.step(len(info["action_mask"]))


def test_episode_length_equals_hand_size_times_num_rounds() -> None:
    env = SushiGoEnv(hand_size=rules.HAND_SIZE, num_rounds=3, opponent_policy=_always_first_action)

    _, info = env.reset(seed=0)
    terminated = False
    step_count = 0

    while not terminated:
        action = int(np.flatnonzero(info["action_mask"])[0])
        _, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        assert truncated is False
        if not terminated:
            assert reward == 0.0

    assert step_count == rules.HAND_SIZE * 3
    assert info["round_idx"] == 2
    assert info["turn"] == rules.HAND_SIZE


def test_round_transition_resets_turn_and_advances_round() -> None:
    env = SushiGoEnv(opponent_policy=_always_first_action)
    _, info = env.reset(seed=42)

    for _ in range(rules.HAND_SIZE):
        action = int(np.flatnonzero(info["action_mask"])[0])
        _, _, terminated, _, info = env.step(action)

    assert terminated is False
    assert info["round_idx"] == 1
    assert info["turn"] == 0
    assert info["total_scores"] != (0.0, 0.0)
    assert len(info["my_played"]) == 0
    assert len(info["opp_played"]) == 0


def test_terminal_score_matches_round_totals_plus_pudding() -> None:
    env = SushiGoEnv(opponent_policy=_always_first_action)
    _, info = env.reset(seed=42)

    terminated = False
    reward = 0.0

    while not terminated:
        action = int(np.flatnonzero(info["action_mask"])[0])
        _, reward, terminated, _, info = env.step(action)

    pudding_my, pudding_opp = rules.score_pudding(
        int(info["my_pudding"]),
        int(info["opp_pudding"]),
        penalty_for_last=False,
    )
    expected_my = float(info["total_scores"][0]) + pudding_my
    expected_opp = float(info["total_scores"][1]) + pudding_opp

    assert info["my_score"] == pytest.approx(expected_my)
    assert info["opp_score"] == pytest.approx(expected_opp)
    assert reward == pytest.approx(expected_my - expected_opp)


def test_observation_and_mask_dtypes() -> None:
    env = SushiGoEnv(opponent_policy=_always_first_action)
    obs, info = env.reset(seed=11)

    assert obs.dtype == np.float32
    assert info["action_mask"].dtype == np.bool_


def test_observation_shape_is_fixed() -> None:
    env = SushiGoEnv(opponent_policy=_always_first_action)
    obs, _ = env.reset(seed=11)
    expected_size = (len(rules.CARD_TYPES) * 3) + (rules.HAND_SIZE * (len(rules.CARD_TYPES) + 1)) + 10
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
    assert decoded["round_idx"] == 0.0
    assert decoded["my_pudding_count"] == 0.0
    assert decoded["opp_pudding_count"] == 0.0
    assert decoded["my_total_score"] == 0.0
    assert decoded["opp_total_score"] == 0.0


def test_observation_turn_and_hand_size_update_after_step() -> None:
    env = SushiGoEnv(opponent_policy=_always_first_action)
    env.reset(seed=7)
    obs, _, _, _, info = env.step(0)
    decoded = SushiGoEnv.decode_observation(obs)
    assert decoded["turn"] == 1.0
    assert decoded["current_hand_size"] == float(rules.HAND_SIZE - 1)
    assert decoded["round_idx"] == 0.0
    expected_slots = tuple(info["my_hand"]) + (EMPTY_HAND_SLOT,) * (rules.HAND_SIZE - len(info["my_hand"]))
    assert decoded["my_hand_slots"] == expected_slots


def test_chopsticks_actions_become_legal_when_chopsticks_in_play() -> None:
    env = SushiGoEnv(opponent_policy=_always_first_action)
    env._state = GameState(
        hands=[
            [rules.NIGIRI_3, rules.TEMPURA, rules.SASHIMI],
            [rules.NIGIRI_1, rules.DUMPLING, rules.MAKI_1],
        ],
        played=[[rules.CHOPSTICKS], []],
        pudding=[[], []],
        turn=0,
        round_idx=0,
        total_scores=(0.0, 0.0),
        last_actions=None,
    )

    mask = env.action_mask(player_index=0)

    assert mask[:3].tolist() == [True, True, True]
    chop_action = env.max_hand + 0 * (env.max_hand - 1) + 0
    assert bool(mask[chop_action]) is True


def test_action_description_formats_single_and_chopsticks_actions() -> None:
    hand = [rules.CHOPSTICKS, rules.NIGIRI_3, rules.TEMPURA]
    assert SushiGoEnv.describe_action(1, hand, hand_size=rules.HAND_SIZE) == "idx 1 -> nigiri3"

    chop_action = rules.HAND_SIZE + 1 * (rules.HAND_SIZE - 1) + 1
    assert "chopsticks(" in SushiGoEnv.describe_action(chop_action, hand, hand_size=rules.HAND_SIZE)


def test_policy_input_includes_multi_round_features() -> None:
    env = SushiGoEnv(opponent_policy=_always_first_action)
    env._state = GameState(
        hands=[[rules.PUDDING, rules.NIGIRI_2], [rules.TEMPURA, rules.MAKI_2]],
        played=[[rules.CHOPSTICKS], [rules.WASABI]],
        pudding=[[rules.PUDDING], []],
        turn=3,
        round_idx=1,
        total_scores=(12.0, 9.0),
        last_actions=None,
    )

    policy_input = env._policy_input_for_player(0, env.action_mask(player_index=0))

    assert policy_input.round_idx == 1
    assert policy_input.num_rounds == 3
    assert policy_input.my_pudding_count == 1
    assert policy_input.opp_pudding_count == 0
    assert policy_input.my_total_score == 12.0
    assert policy_input.opp_total_score == 9.0
