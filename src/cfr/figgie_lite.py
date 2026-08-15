"""Figgie-Lite: a two-player, zero-sum reduction of Figgie, small enough for
exact counterfactual regret minimization (CFR).

The reduction keeps the structure that makes the full game hard (paper
Section 4.5 and \S\ref{sec:cfr} in main.tex):

* a hidden "world" (which of four suits is the goal suit);
* a world-dependent deck (the goal suit appears three times, the others
  once), so each player's private hand is informative about the world --
  the same "hand prunes impossible assignments" effect as the full game;
* simultaneous, buy-only actions {PASS, BUY(s)} at a fixed price, so a buy
  is a *public signal* of the player's belief (behavior conditioning);
* deterministic resolution in player order, and a settle-style payoff
  (cash + goal-card value) reduced to the two-player payoff difference.

Two players keep exploitability well defined; the full multiplayer game has
no unique equilibrium, so this reduction is where the paper's exploitability
measurement lives. The game is deliberately small (4 worlds x 8 deals x
5 actions x a few ticks) so CFR converges on a laptop.
"""

from __future__ import annotations

import random
from typing import Optional

N_SUITS = 4
CARD_VALUE = 10.0
BUY_PRICE = 6.0          # below CARD_VALUE so buying the goal suit is +EV
START_CASH = 100.0

PASS = 0
ACTION_COUNT = N_SUITS + 1          # PASS + one BUY per suit


def all_hands() -> list[tuple[int, ...]]:
    """All 3-card multisets over the 4 suits (20 types)."""
    hands = []
    for c0 in range(4):
        for c1 in range(4):
            for c2 in range(4):
                for c3 in range(4):
                    if c0 + c1 + c2 + c3 == 3:
                        hands.append((c0, c1, c2, c3))
    return hands


ALL_HANDS: list[tuple[int, ...]] = all_hands()


def reachable_hands() -> list[tuple[int, ...]]:
    """Every hand a player can hold during a game: a multiset of the deck, so
    counts per suit are in 0..3 and total <= 6. Buy actions transfer cards
    between players but never change the multiset in play, so these are exactly
    the hands reachable after any sequence of buys (for any goal)."""
    out = []
    for c0 in range(4):
        for c1 in range(4):
            for c2 in range(4):
                for c3 in range(4):
                    if c0 + c1 + c2 + c3 <= 6:
                        out.append((c0, c1, c2, c3))
    return sorted(out)


REACHABLE_HANDS: list[tuple[int, ...]] = reachable_hands()
HAND_INDEX: dict[tuple[int, ...], int] = {
    h: i for i, h in enumerate(REACHABLE_HANDS)
}


def deck_counts(goal: int) -> tuple[int, ...]:
    """World deck: goal suit x3, every other suit x1 (6 cards)."""
    return tuple(3 if s == goal else 1 for s in range(N_SUITS))


def feasible_hands(goal: int) -> list[tuple[int, ...]]:
    """Hands a single player can hold in world `goal` (subset of the deck)."""
    dc = deck_counts(goal)
    return [h for h in ALL_HANDS if all(h[s] <= dc[s] for s in range(N_SUITS))]


FEASIBLE: tuple[list[tuple[int, ...]], ...] = tuple(
    feasible_hands(g) for g in range(N_SUITS)
)


def deal_splits(goal: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """All ordered (hand0, hand1) splits of the world deck for `goal`."""
    dc = deck_counts(goal)
    splits = []
    for h0 in ALL_HANDS:
        if all(h0[s] <= dc[s] for s in range(N_SUITS)):
            h1 = tuple(dc[s] - h0[s] for s in range(N_SUITS))
            splits.append((h0, h1))
    return splits


SPLITS: tuple[list[tuple[tuple[int, ...], tuple[int, ...]]], ...] = tuple(
    deal_splits(g) for g in range(N_SUITS)
)


def sample_chance(rng: random.Random) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Sample a world and a deal jointly (uniform over worlds and splits)."""
    g = rng.randrange(N_SUITS)
    h0, h1 = SPLITS[g][rng.randrange(len(SPLITS[g]))]
    return g, h0, h1


def sample_goal(rng: random.Random) -> int:
    return rng.randrange(N_SUITS)


def apply_buy(
    hands: list[list[int]], cash: list[float], player: int, action: int
) -> tuple[list[list[int]], list[float], bool]:
    """Resolve one player's action; returns (hands, cash, success).

    PASS does nothing (success=False). BUY(s) transfers one card of suit s
    from the opponent at BUY_PRICE if the opponent holds one (success=True);
    a failed buy is a no-op. In both cases the action and its outcome are
    public, so each tick records (a0, a1, s0, s1) in the history.
    """
    if action == PASS:
        return [h[:] for h in hands], list(cash), False
    suit = action - 1
    opp = 1 - player
    if hands[opp][suit] < 1:
        return [h[:] for h in hands], list(cash), False
    h = [row[:] for row in hands]
    c = list(cash)
    h[player][suit] += 1
    h[opp][suit] -= 1
    c[player] -= BUY_PRICE
    c[opp] += BUY_PRICE
    return h, c, True


def encode_joint(a0: int, a1: int, s0: bool, s1: bool) -> int:
    """Pack one tick's public record (both actions and both buy outcomes)."""
    return ((a0 * ACTION_COUNT + a1) << 2) | (int(s0) << 1) | int(s1)


def payoff(hands: list[list[int]], cash: list[float], goal: int) -> float:
    """Player-0 settle advantage (zero-sum)."""
    s0 = cash[0] + CARD_VALUE * hands[0][goal]
    s1 = cash[1] + CARD_VALUE * hands[1][goal]
    return s0 - s1


def info_key(player: int, hand: tuple[int, ...], history: tuple) -> tuple:
    return (player, HAND_INDEX[hand], history)


def iter_pasts(max_len: int) -> list[tuple]:
    """All public histories up to length max_len (joint actions as ints)."""
    pasts: list[tuple] = [()]
    for _ in range(max_len):
        pasts = pasts + [
            h + (joint,) for h in pasts for joint in range(ACTION_COUNT * ACTION_COUNT)
        ]
    return pasts
