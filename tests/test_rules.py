"""Unit tests for deterministic Sushi Go v1 scoring rules."""

from __future__ import annotations

import pytest

from sushigo_rl import rules


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, 0),
        (1, 0),
        (2, 5),
        (3, 5),
        (4, 10),
    ],
)
def test_tempura_pairs(count: int, expected: int) -> None:
    assert rules.score_tempura(count) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, 0),
        (2, 0),
        (3, 10),
        (6, 20),
    ],
)
def test_sashimi_triples(count: int, expected: int) -> None:
    assert rules.score_sashimi(count) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, 0),
        (1, 1),
        (2, 3),
        (3, 6),
        (4, 10),
        (5, 15),
        (7, 15),
    ],
)
def test_dumpling_progression(count: int, expected: int) -> None:
    assert rules.score_dumplings(count) == expected


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        ([rules.WASABI, rules.NIGIRI_3], 9),
        ([rules.WASABI, rules.NIGIRI_2, rules.NIGIRI_3], 9),
        ([rules.NIGIRI_3, rules.WASABI], 3),
        ([rules.WASABI, rules.WASABI, rules.NIGIRI_1, rules.NIGIRI_3], 12),
    ],
)
def test_wasabi_application_order(cards: list[str], expected: int) -> None:
    assert rules.score_nigiri(cards) == expected


@pytest.mark.parametrize(
    ("my_maki", "opp_maki", "expected"),
    [
        (5, 5, (4.5, 4.5)),
        (8, 3, (6.0, 3.0)),
        (2, 7, (3.0, 6.0)),
    ],
)
def test_maki_scoring(my_maki: int, opp_maki: int, expected: tuple[float, float]) -> None:
    assert rules.score_maki(my_maki, opp_maki) == expected


def test_deck_composition_matches_spec() -> None:
    deck = rules.build_deck()
    assert len(deck) == 94
    counts = rules.count_cards(deck)
    for card, expected_count in rules.DECK_COMPOSITION.items():
        assert counts[card] == expected_count


def test_score_round_is_consistent_with_score_total() -> None:
    p0 = [rules.TEMPURA, rules.TEMPURA, rules.MAKI_3, rules.NIGIRI_2]
    p1 = [rules.SASHIMI, rules.SASHIMI, rules.SASHIMI, rules.MAKI_1]
    p0_score, p1_score = rules.score_round(p0, p1)
    assert p0_score == rules.score_total(p0, p1)
    assert p1_score == rules.score_total(p1, p0)
