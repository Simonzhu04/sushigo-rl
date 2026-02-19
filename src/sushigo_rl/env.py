"""Deterministic Gymnasium-style Sushi Go environment (2 players, 1 round)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import os
import warnings

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sushigo_rl import rules


OpponentPolicy = Callable[["PolicyInput"], int]
OpponentSampler = Callable[[np.random.Generator], OpponentPolicy]
EMPTY_HAND_SLOT: str = "__empty__"
HAND_SLOT_TYPES: tuple[str, ...] = rules.CARD_TYPES + (EMPTY_HAND_SLOT,)
HAND_SLOT_TO_INDEX: dict[str, int] = {card: idx for idx, card in enumerate(HAND_SLOT_TYPES)}


@dataclass(frozen=True)
class PolicyInput:
    """Visible state passed to a baseline policy."""

    hand: tuple[str, ...]
    my_played: tuple[str, ...]
    opp_played: tuple[str, ...]
    turn: int
    hand_size: int
    action_mask: np.ndarray


@dataclass
class RoundState:
    """Mutable round state for a single episode."""

    hands: list[list[str]]
    played: list[list[str]]
    turn: int = 0
    last_actions: tuple[int, int] | None = None


class SushiGoEnv(gym.Env[np.ndarray, int]):
    """2-player, 1-round Sushi Go environment with deterministic seeding."""

    def __init__(
        self,
        hand_size: int = rules.HAND_SIZE,
        opponent_policy: OpponentPolicy | None = None,
        opponent_sampler: OpponentSampler | None = None,
        fixed_episode_seed: int | None = None,
        env_debug_mask: bool | None = None,
        env_debug_obs: bool | None = None,
    ) -> None:
        if hand_size <= 0:
            raise ValueError("hand_size must be positive")
        if hand_size > len(rules.build_deck()) // 2:
            raise ValueError("hand_size is too large for the fixed deck composition")
        if opponent_policy is not None and opponent_sampler is not None:
            raise ValueError("Provide either opponent_policy or opponent_sampler, not both")
        if hand_size != rules.HAND_SIZE:
            warnings.warn(
                "Using a hand_size different from rules.HAND_SIZE may violate RULES.md constraints.",
                stacklevel=2,
            )

        self.hand_size = hand_size
        self._rng = np.random.default_rng()
        self._state: RoundState | None = None
        self._base_opponent_policy: OpponentPolicy = opponent_policy or self._default_opponent_policy
        self._opponent_sampler = opponent_sampler
        self._episode_opponent_policy: OpponentPolicy = self._base_opponent_policy
        self._fixed_episode_seed = fixed_episode_seed
        self._env_debug_mask = self._resolve_debug_mask(env_debug_mask)
        self._env_debug_obs = self._resolve_debug_obs(env_debug_obs)
        self._episode_counter = 0

        num_card_types = len(rules.CARD_TYPES)
        slot_width = len(HAND_SLOT_TYPES)
        obs_size = (num_card_types * 3) + (self.hand_size * slot_width) + 5
        low = np.zeros(obs_size, dtype=np.float32)
        scalar_high = np.array(
            [
                float(self.hand_size),  # turn
                float(self.hand_size),  # current_hand_size
                float(self.hand_size),  # my_unpaired_wasabi
                float(self.hand_size * 3),  # my_maki_icons
                float(self.hand_size * 3),  # opp_maki_icons
            ],
            dtype=np.float32,
        )
        high = np.concatenate(
            [
                np.full(num_card_types * 3, float(self.hand_size), dtype=np.float32),
                np.ones(self.hand_size * slot_width, dtype=np.float32),
                scalar_high,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.hand_size)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the round and return (obs, info)."""
        del options
        if self._fixed_episode_seed is not None:
            self._rng = np.random.default_rng(self._fixed_episode_seed)
        elif seed is not None:
            self._rng = np.random.default_rng(seed)

        deck = np.array(rules.build_deck(), dtype=object)
        shuffled = deck[self._rng.permutation(deck.size)].tolist()

        hand0 = shuffled[: self.hand_size]
        hand1 = shuffled[self.hand_size : 2 * self.hand_size]

        self._state = RoundState(
            hands=[hand0, hand1],
            played=[[], []],
            turn=0,
            last_actions=None,
        )
        self._episode_opponent_policy = (
            self._opponent_sampler(self._rng) if self._opponent_sampler is not None else self._base_opponent_policy
        )
        self._episode_counter += 1

        obs = self._observation()
        info = self._build_info(my_score=0.0, opp_score=0.0, terminal=False)
        if self._env_debug_obs and self._episode_counter <= 5:
            self._print_observation_debug(obs, info, self._episode_counter)
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Resolve one simultaneous turn and return Gymnasium-style outputs."""
        state = self._require_state()
        if state.turn >= self.hand_size:
            raise RuntimeError("Episode has already terminated. Call reset().")

        my_mask = self.action_mask(player_index=0)
        if self._env_debug_mask:
            self._assert_action_matches_mask(
                action=action,
                mask=my_mask,
                player_index=0,
                actor_label="agent",
            )
        self._validate_action(action, my_mask)

        opp_mask = self.action_mask(player_index=1)
        opp_action = int(self._episode_opponent_policy(self._policy_input_for_player(1, opp_mask)))
        if self._env_debug_mask:
            self._assert_action_matches_mask(
                action=opp_action,
                mask=opp_mask,
                player_index=1,
                actor_label="opponent",
            )
        self._validate_action(opp_action, opp_mask)

        my_card = state.hands[0].pop(int(action))
        opp_card = state.hands[1].pop(opp_action)
        state.played[0].append(my_card)
        state.played[1].append(opp_card)
        state.last_actions = (int(action), opp_action)

        # Remaining cards are swapped after simultaneous card resolution.
        state.hands[0], state.hands[1] = state.hands[1], state.hands[0]
        state.turn += 1

        terminated = state.turn == self.hand_size
        truncated = False

        if terminated:
            my_score, opp_score = rules.score_round(state.played[0], state.played[1])
            reward = float(my_score - opp_score)
        else:
            my_score = 0.0
            opp_score = 0.0
            reward = 0.0

        obs = self._observation()
        info = self._build_info(my_score=float(my_score), opp_score=float(opp_score), terminal=terminated)
        return obs, reward, terminated, truncated, info

    def set_opponent_policy(self, opponent_policy: OpponentPolicy) -> None:
        """Set opponent policy callable used at each step."""
        self._opponent_sampler = None
        self._base_opponent_policy = opponent_policy
        self._episode_opponent_policy = opponent_policy

    def action_mask(self, player_index: int = 0) -> np.ndarray:
        """Return a fixed-size boolean action mask for the selected player."""
        state = self._require_state()
        if player_index not in (0, 1):
            raise ValueError("player_index must be 0 or 1")

        legal = len(state.hands[player_index])
        mask = np.zeros(self.hand_size, dtype=np.bool_)
        mask[:legal] = True
        return mask

    def action_masks(self) -> np.ndarray:
        """Return the learning-agent mask in sb3-contrib compatible format."""
        return self.action_mask(player_index=0)

    def _observation(self) -> np.ndarray:
        state = self._require_state()
        return self.encode_observation(
            my_played=state.played[0],
            opp_played=state.played[1],
            my_hand=state.hands[0],
            turn=state.turn,
            current_hand_size=len(state.hands[0]),
            hand_size=self.hand_size,
        )

    @staticmethod
    def _counts_vector(cards: list[str]) -> np.ndarray:
        counts = rules.count_cards(cards)
        return np.array([counts[card] for card in rules.CARD_TYPES], dtype=np.float32)

    @staticmethod
    def encode_observation(
        my_played: Sequence[str],
        opp_played: Sequence[str],
        my_hand: Sequence[str],
        turn: int,
        current_hand_size: int,
        hand_size: int = rules.HAND_SIZE,
    ) -> np.ndarray:
        """Encode observation features into the fixed-size float32 vector."""
        my_played_counts = SushiGoEnv._counts_vector(list(my_played))
        opp_played_counts = SushiGoEnv._counts_vector(list(opp_played))
        my_hand_counts = SushiGoEnv._counts_vector(list(my_hand))

        slot_width = len(HAND_SLOT_TYPES)
        hand_slots = np.zeros((hand_size, slot_width), dtype=np.float32)
        for slot_idx in range(hand_size):
            card = my_hand[slot_idx] if slot_idx < len(my_hand) else EMPTY_HAND_SLOT
            if card not in HAND_SLOT_TO_INDEX:
                raise ValueError(f"Unknown card for hand slot encoding: {card}")
            hand_slots[slot_idx, HAND_SLOT_TO_INDEX[card]] = 1.0

        scalars = np.array(
            [
                float(turn),
                float(current_hand_size),
                float(rules.count_available_wasabi(my_played)),
                float(rules.count_maki_icons(my_played)),
                float(rules.count_maki_icons(opp_played)),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                my_played_counts,
                opp_played_counts,
                my_hand_counts,
                hand_slots.reshape(-1),
                scalars,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def decode_observation(obs: np.ndarray, hand_size: int = rules.HAND_SIZE) -> dict[str, Any]:
        """Decode observation vector into interpretable feature groups."""
        n = len(rules.CARD_TYPES)
        slot_width = len(HAND_SLOT_TYPES)
        expected_size = (n * 3) + (hand_size * slot_width) + 5
        if obs.shape != (expected_size,):
            raise ValueError(f"Unexpected observation shape {obs.shape}, expected {(expected_size,)}")

        slot_start = 3 * n
        slot_end = slot_start + (hand_size * slot_width)
        slot_one_hot = obs[slot_start:slot_end].reshape(hand_size, slot_width).astype(np.float32)
        slot_indices = np.argmax(slot_one_hot, axis=1)
        slot_cards = tuple(HAND_SLOT_TYPES[idx] for idx in slot_indices)

        return {
            "my_played_counts": obs[0:n].astype(np.float32),
            "opp_played_counts": obs[n : 2 * n].astype(np.float32),
            "my_hand_counts": obs[2 * n : 3 * n].astype(np.float32),
            "my_hand_slots_one_hot": slot_one_hot,
            "my_hand_slots": slot_cards,
            "turn": float(obs[slot_end]),
            "current_hand_size": float(obs[slot_end + 1]),
            "my_unpaired_wasabi": float(obs[slot_end + 2]),
            "my_maki_icons": float(obs[slot_end + 3]),
            "opp_maki_icons": float(obs[slot_end + 4]),
        }

    def _policy_input_for_player(self, player_index: int, mask: np.ndarray) -> PolicyInput:
        state = self._require_state()
        return PolicyInput(
            hand=tuple(state.hands[player_index]),
            my_played=tuple(state.played[player_index]),
            opp_played=tuple(state.played[1 - player_index]),
            turn=state.turn,
            hand_size=len(state.hands[player_index]),
            action_mask=mask.copy(),
        )

    def _build_info(self, my_score: float, opp_score: float, terminal: bool) -> dict[str, Any]:
        state = self._require_state()

        info: dict[str, Any] = {
            "my_score": my_score,
            "opp_score": opp_score,
            "turn": state.turn,
            "hand_size": self.hand_size,
            "last_actions": state.last_actions,
            "action_mask": self.action_mask(player_index=0),
            "my_played": tuple(state.played[0]),
            "opp_played": tuple(state.played[1]),
            "my_hand": tuple(state.hands[0]),
            "opp_hand": tuple(state.hands[1]),
        }

        if terminal:
            info["my_breakdown"] = rules.score_breakdown(state.played[0], state.played[1])
            info["opp_breakdown"] = rules.score_breakdown(state.played[1], state.played[0])

        return info

    @staticmethod
    def _validate_action(action: int, mask: np.ndarray) -> None:
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be an int index into the hand, got: {action!r}")
        if action < 0 or action >= len(mask) or not bool(mask[int(action)]):
            raise ValueError(f"Illegal action index: {action}")

    def _assert_action_matches_mask(
        self,
        action: int,
        mask: np.ndarray,
        player_index: int,
        actor_label: str,
    ) -> None:
        state = self._require_state()
        is_valid_type = isinstance(action, (int, np.integer))
        mask_ok = bool(mask[int(action)]) if is_valid_type and 0 <= int(action) < len(mask) else False
        if mask_ok:
            return

        details = {
            "actor": actor_label,
            "player_index": player_index,
            "turn": state.turn,
            "hand_size": len(state.hands[player_index]),
            "mask": mask.astype(np.int8).tolist(),
            "action": action,
        }
        raise AssertionError(f"Mask violation detected: {details}")

    @staticmethod
    def _resolve_debug_mask(env_debug_mask: bool | None) -> bool:
        if env_debug_mask is not None:
            return env_debug_mask
        env_value = os.getenv("SUSHIGO_ENV_DEBUG_MASK", "").strip().lower()
        return env_value in {"1", "true", "yes", "on"}

    @staticmethod
    def _resolve_debug_obs(env_debug_obs: bool | None) -> bool:
        if env_debug_obs is not None:
            return env_debug_obs
        env_value = os.getenv("SUSHIGO_ENV_DEBUG_OBS", "").strip().lower()
        return env_value in {"1", "true", "yes", "on"}

    def _print_observation_debug(self, obs: np.ndarray, info: dict[str, Any], episode_index: int) -> None:
        decoded = self.decode_observation(obs, hand_size=self.hand_size)
        card_names = list(rules.CARD_TYPES)
        my_played = {card_names[idx]: int(value) for idx, value in enumerate(decoded["my_played_counts"])}
        opp_played = {card_names[idx]: int(value) for idx, value in enumerate(decoded["opp_played_counts"])}
        my_hand = {card_names[idx]: int(value) for idx, value in enumerate(decoded["my_hand_counts"])}
        slot_preview_width = min(5, self.hand_size)
        slot_preview = [f"{idx}:{card}" for idx, card in enumerate(decoded["my_hand_slots"][:slot_preview_width])]
        print(
            "[SushiGoEnv obs-debug] "
            f"episode={episode_index} "
            f"turn={int(decoded['turn'])} "
            f"current_hand_size={int(decoded['current_hand_size'])} "
            f"mask={info['action_mask'].astype(np.int8).tolist()}"
        )
        print(f"[SushiGoEnv obs-debug] my_played_counts={my_played}")
        print(f"[SushiGoEnv obs-debug] opp_played_counts={opp_played}")
        print(f"[SushiGoEnv obs-debug] my_hand_counts={my_hand}")
        print(f"[SushiGoEnv obs-debug] hand_slot_preview={slot_preview}")
        print(
            "[SushiGoEnv obs-debug] "
            f"my_unpaired_wasabi={decoded['my_unpaired_wasabi']:.0f} "
            f"my_maki_icons={decoded['my_maki_icons']:.0f} "
            f"opp_maki_icons={decoded['opp_maki_icons']:.0f}"
        )

    def _default_opponent_policy(self, policy_input: PolicyInput) -> int:
        legal_actions = np.flatnonzero(policy_input.action_mask)
        if legal_actions.size == 0:
            raise ValueError("No legal actions available for opponent")
        return int(self._rng.choice(legal_actions))

    def _require_state(self) -> RoundState:
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset().")
        return self._state
