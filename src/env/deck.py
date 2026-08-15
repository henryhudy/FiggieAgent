"""Deck construction, suit-count assignment, and dealing for Figgie.

Rules: 40 cards, 4 suits (CLUBS, SPADES black; HEARTS, DIAMONDS red). One suit
has 8 cards, two have 10, one has 12. The goal suit is the suit of the same
color as the 12-card suit (it holds 8 or 10 cards). Card ranks never matter:
hands are per-suit count vectors and cards trade as suit-tokens.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import permutations

SUITS: tuple[str, ...] = ("CLUBS", "SPADES", "HEARTS", "DIAMONDS")
COLORS: dict[str, str] = {
    "CLUBS": "BLACK",
    "SPADES": "BLACK",
    "HEARTS": "RED",
    "DIAMONDS": "RED",
}
SUIT_INDEX: dict[str, int] = {s: i for i, s in enumerate(SUITS)}
N_SUITS = 4
TOTAL_CARDS = 40
COUNT_SET = (8, 10, 10, 12)
CARD_VALUE = 10.0


@dataclass(frozen=True)
class Assignment:
    """Per-suit card counts plus the derived 12-card and goal suits."""

    counts: tuple[int, ...]
    twelve_suit: int
    goal_suit: int

    @classmethod
    def from_counts(cls, counts: dict[str, int]) -> "Assignment":
        c = tuple(counts[s] for s in SUITS)
        twelve = next(i for i, n in enumerate(c) if n == 12)
        goal = next(
            i for i in range(N_SUITS) if i != twelve and COLORS[SUITS[i]] == COLORS[SUITS[twelve]]
        )
        return cls(c, twelve, goal)


ALL_ASSIGNMENTS: tuple[Assignment, ...] = tuple(
    Assignment.from_counts(dict(zip(SUITS, perm))) for perm in set(permutations(COUNT_SET))
)
assert len(ALL_ASSIGNMENTS) == 12


def sample_assignment(rng: random.Random) -> Assignment:
    return rng.choice(ALL_ASSIGNMENTS)


def deal(rng: random.Random, n_players: int, assignment: Assignment) -> list[list[int]]:
    deck = [suit for suit, count in enumerate(assignment.counts) for _ in range(count)]
    rng.shuffle(deck)
    assert TOTAL_CARDS % n_players == 0
    per = TOTAL_CARDS // n_players
    hands: list[list[int]] = []
    for p in range(n_players):
        hand = [0] * N_SUITS
        for suit in deck[p * per : (p + 1) * per]:
            hand[suit] += 1
        hands.append(hand)
    return hands


def cards_per_player(n_players: int) -> int:
    return TOTAL_CARDS // n_players


def hand_likelihood(hand: list[int], counts: tuple[int, ...]) -> float:
    """P(hand | assignment counts), unnormalized over assignments."""
    if any(hand[s] > counts[s] for s in range(N_SUITS)):
        return 0.0
    w = 1.0
    for s in range(N_SUITS):
        w *= math.comb(counts[s], hand[s])
    return w
