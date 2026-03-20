"""Deterministic heuristic baseline for the 3-round Sushi Go variant."""

from __future__ import annotations

from dataclasses import dataclass

from sushigo_rl import rules
from sushigo_rl.env import PolicyInput


@dataclass(frozen=True)
class HeuristicAgent:
    """Rule-based baseline using only visible state information."""

    def select_action(self, policy_input: PolicyInput) -> int:
        """Select the legal action with the best deterministic heuristic value."""
        counts = rules.count_cards(policy_input.my_played)
        my_maki = rules.count_maki_icons(policy_input.my_played)
        opp_maki = rules.count_maki_icons(policy_input.opp_played)
        available_wasabi = rules.count_available_wasabi(policy_input.my_played)

        best_action: int | None = None
        best_score = float("-inf")

        max_hand = rules.HAND_SIZE  # 10

        for action, is_legal in enumerate(policy_input.action_mask):
            if not bool(is_legal):
                continue

            if action < max_hand:
                card = policy_input.hand[action]
                score = self._card_value(
                    card=card,
                    hand=policy_input.hand,
                    counts=counts,
                    available_wasabi=available_wasabi,
                    my_maki=my_maki,
                    opp_maki=opp_maki,
                    hand_size=policy_input.hand_size,
                )
            else:
                k = action - max_hand
                i = k // (max_hand - 1)
                j = k % (max_hand - 1)
                if j >= i:
                    j += 1

                c1 = policy_input.hand[i]
                c2 = policy_input.hand[j]
                score = (
                    self._card_value(c1, policy_input.hand, counts, available_wasabi, my_maki, opp_maki, policy_input.hand_size)
                    + self._card_value(c2, policy_input.hand, counts, available_wasabi, my_maki, opp_maki, policy_input.hand_size)
                    + 0.25  
                )

            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None:
            raise ValueError("No legal action available")
        return best_action

    @staticmethod
    def _card_value(
        card: str,
        hand: tuple[str, ...],
        counts: dict[str, int],
        available_wasabi: int,
        my_maki: int,
        opp_maki: int,
        hand_size: int,
    ) -> float:
        if card == rules.CHOPSTICKS:
            if hand_size >= 8:
                return 1.6
            if hand_size >= 5:
                return 1.0
            return 0.4
        if card == rules.TEMPURA:
            return 5.0 if counts[rules.TEMPURA] % 2 == 1 else 1.0

        if card == rules.SASHIMI:
            mod = counts[rules.SASHIMI] % 3
            if mod == 2:
                return 10.0
            if mod == 1:
                return 2.5
            return 1.0

        if card == rules.DUMPLING:
            current = counts[rules.DUMPLING]
            return float(rules.score_dumplings(current + 1) - rules.score_dumplings(current))

        if card in rules.NIGIRI_VALUES:
            value = float(rules.NIGIRI_VALUES[card])
            if available_wasabi > 0:
                return value * 3.0
            return value

        if card == rules.WASABI:
            if available_wasabi > 0:
                return 0.25
            best_follow_up = max((rules.NIGIRI_VALUES[c] for c in hand if c in rules.NIGIRI_VALUES), default=0)
            return float(best_follow_up * 2)

        if card in rules.MAKI_ICONS:
            icons = float(rules.MAKI_ICONS[card])
            race_gap = opp_maki - my_maki
            urgency = 1.4 if race_gap >= 0 else 0.8
            if hand_size <= 3 and abs(race_gap) <= 3:
                urgency += 0.6
            return icons * urgency
        
        if card == rules.PUDDING:
            # Simple heuristic: puddings matter more early than late.
            # (End-of-game scoring; we don't model full game state here.)
            if hand_size >= 8:
                return 2.0
            if hand_size >= 5:
                return 1.2
            return 0.6

        raise ValueError(f"Unknown card: {card}")


__all__ = ["HeuristicAgent"]
