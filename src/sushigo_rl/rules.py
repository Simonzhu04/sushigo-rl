"""Deterministic Sushi Go v1 rules and scoring helpers.

This module is pure logic only. It contains no environment state or randomness
outside deck construction order (which is deterministic and unshuffled).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

HAND_SIZE: int = 10

TEMPURA: str = "tempura"
SASHIMI: str = "sashimi"
DUMPLING: str = "dumpling"
MAKI_1: str = "maki1"
MAKI_2: str = "maki2"
MAKI_3: str = "maki3"
NIGIRI_1: str = "nigiri1"
NIGIRI_2: str = "nigiri2"
NIGIRI_3: str = "nigiri3"
WASABI: str = "wasabi"

CARD_TYPES: tuple[str, ...] = (
    TEMPURA,
    SASHIMI,
    DUMPLING,
    MAKI_1,
    MAKI_2,
    MAKI_3,
    NIGIRI_1,
    NIGIRI_2,
    NIGIRI_3,
    WASABI,
)

DECK_COMPOSITION: Mapping[str, int] = {
    TEMPURA: 14,
    SASHIMI: 14,
    DUMPLING: 14,
    MAKI_1: 6,
    MAKI_2: 12,
    MAKI_3: 8,
    NIGIRI_1: 10,
    NIGIRI_2: 5,
    NIGIRI_3: 5,
    WASABI: 6,
}

DUMPLING_POINTS: tuple[int, ...] = (0, 1, 3, 6, 10, 15)

MAKI_ICONS: Mapping[str, int] = {
    MAKI_1: 1,
    MAKI_2: 2,
    MAKI_3: 3,
}

NIGIRI_VALUES: Mapping[str, int] = {
    NIGIRI_1: 1,
    NIGIRI_2: 2,
    NIGIRI_3: 3,
}


@dataclass(frozen=True)
class ScoreBreakdown:
    """Score components for one player in a single round."""

    tempura: float
    sashimi: float
    dumpling: float
    nigiri: float
    maki: float

    @property
    def total(self) -> float:
        """Total score for this player."""
        return self.tempura + self.sashimi + self.dumpling + self.nigiri + self.maki


def build_deck() -> list[str]:
    """Return the full unshuffled deck for rules v1."""
    deck: list[str] = []
    for card in CARD_TYPES:
        deck.extend([card] * DECK_COMPOSITION[card])
    return deck


def score_tempura(count: int) -> int:
    """Score tempura cards (5 points per pair)."""
    return (count // 2) * 5


def score_sashimi(count: int) -> int:
    """Score sashimi cards (10 points per triple)."""
    return (count // 3) * 10


def score_dumplings(count: int) -> int:
    """Score dumpling cards using the capped progression table."""
    return DUMPLING_POINTS[min(count, 5)]


def count_cards(cards: Sequence[str]) -> dict[str, int]:
    """Count cards by type for all known card types."""
    counts: dict[str, int] = {card: 0 for card in CARD_TYPES}
    for card in cards:
        if card not in counts:
            raise ValueError(f"Unknown card type: {card}")
        counts[card] += 1
    return counts


def count_maki_icons(cards: Sequence[str]) -> int:
    """Return total number of maki icons in a sequence of cards."""
    total = 0
    for card in cards:
        total += MAKI_ICONS.get(card, 0)
    return total


def count_available_wasabi(cards_in_play_order: Sequence[str]) -> int:
    """Count unpaired wasabi after applying Nigiri in play order."""
    available_wasabi = 0
    for card in cards_in_play_order:
        if card == WASABI:
            available_wasabi += 1
        elif card in NIGIRI_VALUES and available_wasabi > 0:
            available_wasabi -= 1
        elif card not in CARD_TYPES:
            raise ValueError(f"Unknown card type: {card}")
    return available_wasabi


def score_nigiri(cards_in_play_order: Sequence[str]) -> int:
    """Score Nigiri cards, applying Wasabi only to later Nigiri in order played."""
    total = 0
    available_wasabi = 0

    for card in cards_in_play_order:
        if card == WASABI:
            available_wasabi += 1
        elif card in NIGIRI_VALUES:
            value = NIGIRI_VALUES[card]
            if available_wasabi > 0:
                total += value * 3
                available_wasabi -= 1
            else:
                total += value
        elif card not in CARD_TYPES:
            raise ValueError(f"Unknown card type: {card}")

    return total


def score_maki(my_maki_icons: int, opp_maki_icons: int) -> tuple[float, float]:
    """Score maki bonus for 2 players only.

    Higher total gets 6, lower gets 3, ties split 9 => 4.5 each.
    """
    if my_maki_icons == opp_maki_icons:
        return 4.5, 4.5
    if my_maki_icons > opp_maki_icons:
        return 6.0, 3.0
    return 3.0, 6.0


def score_non_maki(cards_in_play_order: Sequence[str]) -> tuple[float, float, float, float]:
    """Return (tempura, sashimi, dumpling, nigiri) for one played pile."""
    counts = count_cards(cards_in_play_order)
    tempura_score = float(score_tempura(counts[TEMPURA]))
    sashimi_score = float(score_sashimi(counts[SASHIMI]))
    dumpling_score = float(score_dumplings(counts[DUMPLING]))
    nigiri_score = float(score_nigiri(cards_in_play_order))
    return tempura_score, sashimi_score, dumpling_score, nigiri_score


def score_breakdown(my_cards_in_play_order: Sequence[str], opp_cards_in_play_order: Sequence[str]) -> ScoreBreakdown:
    """Return full component-wise score for one player versus an opponent."""
    tempura_score, sashimi_score, dumpling_score, nigiri_score = score_non_maki(my_cards_in_play_order)
    my_maki_icons = count_maki_icons(my_cards_in_play_order)
    opp_maki_icons = count_maki_icons(opp_cards_in_play_order)
    maki_score, _ = score_maki(my_maki_icons, opp_maki_icons)
    return ScoreBreakdown(
        tempura=tempura_score,
        sashimi=sashimi_score,
        dumpling=dumpling_score,
        nigiri=nigiri_score,
        maki=maki_score,
    )


def score_total(my_cards_in_play_order: Sequence[str], opp_cards_in_play_order: Sequence[str]) -> float:
    """Return the full round score for one player versus an opponent."""
    return score_breakdown(my_cards_in_play_order, opp_cards_in_play_order).total


def score_round(
    player0_cards_in_play_order: Sequence[str],
    player1_cards_in_play_order: Sequence[str],
) -> tuple[float, float]:
    """Return total scores for both players."""
    p0 = score_total(player0_cards_in_play_order, player1_cards_in_play_order)
    p1 = score_total(player1_cards_in_play_order, player0_cards_in_play_order)
    return p0, p1
