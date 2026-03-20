"""Uniform-random baseline agent over legal actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sushigo_rl.env import PolicyInput


@dataclass
class RandomAgent:
    """Random baseline that samples uniformly from legal actions."""

    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def select_action(self, policy_input: PolicyInput) -> int:
        """Choose a uniformly random legal action index."""
        legal = np.flatnonzero(policy_input.action_mask)
        if legal.size == 0:
            raise ValueError("No legal actions available")
        return int(self._rng.choice(legal))


__all__ = ["RandomAgent"]
