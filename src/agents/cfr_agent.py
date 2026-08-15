"""Full-engine agent driven by the reduced-game CFR policy.

The Figgie-Lite CFR policy is buy-only (PASS / BUY(s) per suit). This agent
uses it as a belief-conditioned buy decision in the full 4-player engine:

* an exact posterior over the goal suit is computed from the private hand
  (the same hand_likelihood weighting as the heuristic bot);
* the learned policy for each world g gives P(buy s | hand, g); averaging
  over the posterior yields a belief-weighted action distribution;
* if the top action is a buy of suit s, the agent works the book to buy s
  (accept a cheap ask, otherwise post a bid), else it passes.

The full-engine hand is projected onto the reduced game's hand space by
clamping each suit count to the reduced deck's range. No neural net is used.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from ..env.deck import ALL_ASSIGNMENTS, N_SUITS, hand_likelihood
from ..env.state import (
    Action,
    BID,
    KIND_ACCEPT_ASK,
    KIND_PASS,
    KIND_POST_BID,
)
from .base import Agent
from .heuristic import HeuristicAgent

FAIR_BASE = 24.0
MAX_FAIR = 30.0
BID_MARKUP = 3.0
MIN_GOAL_CONFIDENCE = 0.3
MIN_BUY_EDGE = 0.2


def _clamp_hand(hand: list[int]) -> tuple[int, ...]:
    """Project a full-engine hand onto the reduced game's hand space."""
    clamped = [min(max(int(c), 0), 3) for c in hand]
    total = sum(clamped)
    if total > 6:
        keep = 6
        out = [0] * N_SUITS
        for s in range(N_SUITS):
            take = min(clamped[s], keep)
            out[s] = take
            keep -= take
        return tuple(out)
    return tuple(clamped)


class CFRAgent(Agent):
    name = "cfr"

    def __init__(self, policy_path: str, seed: Optional[int] = None):
        self.policy_path = policy_path
        self.rng = seed
        with open(policy_path) as f:
            data = json.load(f)
        self.ticks = data["ticks"]
        self.policy = data["policy"]
        from ..cfr.figgie_lite import REACHABLE_HANDS

        self._hand_index = {h: i for i, h in enumerate(REACHABLE_HANDS)}
        self.fallback = HeuristicAgent(seed)

    def new_round(self, state, player: int) -> None:
        self.player = player
        hand = state.hands[player]
        weights = [hand_likelihood(hand, a.counts) for a in ALL_ASSIGNMENTS]
        total = sum(weights)
        self._goal_post = [0.0] * N_SUITS
        for w, a in zip(weights, ALL_ASSIGNMENTS):
            self._goal_post[a.goal_suit] += w
        if total > 0:
            self._goal_post = [w / total for w in self._goal_post]
        self.fallback.new_round(state, player)

    def observe(self, state) -> None:
        pass

    def _world_buy_prob(self, state, player: int, suit: int) -> float:
        """Reduced-policy buy strength for suit s in world s at this hand/tick."""
        hand = _clamp_hand(state.hands[player])
        hi = self._hand_index.get(hand)
        tick = min(state.tick, self.ticks - 1)
        if hi is None:
            return 0.0
        return self.policy[str(suit)][tick][hi][suit + 1]

    def choose_action(self, engine, state, player: int) -> Action:
        suit = max(range(N_SUITS), key=lambda s: self._goal_post[s])
        confidence = self._goal_post[suit]
        if confidence < MIN_GOAL_CONFIDENCE:
            return Action(KIND_PASS)
        if self._world_buy_prob(state, player, suit) < MIN_BUY_EDGE:
            return Action(KIND_PASS)
        cash = state.cash[player]
        fair = min(MAX_FAIR, FAIR_BASE * confidence)
        ba = state.best_ask(suit)
        if ba is not None and ba[1] != player and ba[0] <= fair and cash >= ba[0]:
            return Action(KIND_ACCEPT_ASK, suit)
        bid = max(1.0, min(cash, fair - BID_MARKUP))
        if bid >= 1.0 and cash >= bid:
            own = state.quotes.get(player, {}).get(suit)
            if own is None or own[0] != BID or own[1] > bid + 1.0:
                return Action(KIND_POST_BID, suit, bid)
        return Action(KIND_PASS)
