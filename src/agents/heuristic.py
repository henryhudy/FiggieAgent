"""Heuristic Figgie bot: hand-prior belief over suit-count assignments,
updated by opponent trade signals, driving cheap goal-suit accumulation and
junk-suit dumping.
"""

from __future__ import annotations

import random
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

BUY_IF_GOAL = 1.5
BUY_IF_NOT = 0.95
SELL_IF_GOAL = 0.6
SELL_IF_NOT = 1.0
FAIR_BASE = 24.0
MAX_FAIR = 30.0
MIN_BID_CONFIDENCE = 0.3


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def new_round(self, state, player: int) -> None:
        self.player = player
        hand = state.hands[player]
        weights = [hand_likelihood(hand, a.counts) for a in ALL_ASSIGNMENTS]
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self._seen = 0

    def observe(self, state) -> None:
        for trade in state.trades[self._seen:]:
            self._update(trade)
        self._seen = len(state.trades)

    def _update(self, trade) -> None:
        if trade.buyer == self.player or trade.seller == self.player:
            return
        suit = trade.suit
        actively_bought = trade.aggressor == trade.buyer
        weights = []
        for w, a in zip(self.weights, ALL_ASSIGNMENTS):
            w2 = w
            if actively_bought:
                w2 *= BUY_IF_GOAL if a.goal_suit == suit else BUY_IF_NOT
            else:
                w2 *= SELL_IF_GOAL if a.goal_suit == suit else SELL_IF_NOT
            weights.append(w2)
        total = sum(weights)
        if total > 0:
            self.weights = [w / total for w in weights]

    def goal_marginals(self) -> list[float]:
        gm = [0.0] * N_SUITS
        for w, a in zip(self.weights, ALL_ASSIGNMENTS):
            gm[a.goal_suit] += w
        return gm

    def choose_action(self, engine, state, player: int) -> Action:
        hand = state.hands[player]
        cash = state.cash[player]
        gm = self.goal_marginals()
        t = state.time_remaining() / state.config.n_ticks
        time_factor = 1.0 + 0.8 * (1.0 - t)

        best = max(range(N_SUITS), key=lambda s: gm[s])
        fair = min(MAX_FAIR, FAIR_BASE * gm[best] * time_factor)

        ba = state.best_ask(best)
        if ba is not None and ba[1] != player and ba[0] <= fair and cash >= ba[0]:
            return Action(KIND_ACCEPT_ASK, best)

        own = state.quotes.get(player, {}).get(best)
        bid = max(1.0, min(cash, fair - 3.0))
        if gm[best] >= MIN_BID_CONFIDENCE and bid >= 1.0 and cash >= bid:
            if own is None or own[0] != BID or own[1] > bid + 1.0:
                return Action(KIND_POST_BID, best, bid)

        return Action(KIND_PASS)
