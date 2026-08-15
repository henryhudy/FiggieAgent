"""Uniformly random Figgie bot over a compact action pool."""

from __future__ import annotations

import random
from typing import Optional

from ..env.deck import N_SUITS
from ..env.state import (
    Action,
    KIND_ACCEPT_ASK,
    KIND_ACCEPT_BID,
    KIND_PASS,
    KIND_POST_ASK,
    KIND_POST_BID,
    KIND_WITHDRAW,
)
from .base import Agent

PRICE_MAX = 25


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def choose_action(self, engine, state, player: int) -> Action:
        kinds = [KIND_PASS, KIND_POST_BID, KIND_POST_ASK, KIND_ACCEPT_ASK, KIND_ACCEPT_BID, KIND_WITHDRAW]
        kind = self.rng.choice(kinds)
        suit = self.rng.randrange(N_SUITS)
        cash = state.cash[player]
        hand = state.hands[player]

        if kind == KIND_POST_BID:
            high = min(PRICE_MAX, max(1, int(cash)))
            return Action(kind, suit, float(self.rng.randint(1, high)))
        if kind == KIND_POST_ASK:
            if hand[suit] < 1:
                return Action(KIND_PASS)
            return Action(kind, suit, float(self.rng.randint(1, PRICE_MAX)))
        if kind == KIND_ACCEPT_ASK:
            ba = state.best_ask(suit)
            if ba is None or ba[1] == player or cash < ba[0]:
                return Action(KIND_PASS)
            return Action(kind, suit)
        if kind == KIND_ACCEPT_BID:
            bb = state.best_bid(suit)
            if bb is None or bb[1] == player or hand[suit] < 1:
                return Action(KIND_PASS)
            return Action(kind, suit)
        if kind == KIND_WITHDRAW:
            if suit not in state.quotes.get(player, {}):
                return Action(KIND_PASS)
            return Action(kind, suit)
        return Action(KIND_PASS)
