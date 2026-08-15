"""Interactive human agent for the playable Figgie CLI (P3).

Presents the current round state and a curated legal-action menu on the
terminal, then reads a selection. Optionally shows the strategy-free belief
tracker's goal-suit marginals (the same public signal the bots use) so the
human can make informed trades.
"""

from __future__ import annotations

from typing import Optional

from ..belief.tracker import BeliefTracker
from ..env.deck import N_SUITS, SUITS
from ..env.state import (
    Action,
    KIND_ACCEPT_ASK,
    KIND_ACCEPT_BID,
    KIND_BUNDLE_BID,
    KIND_PASS,
    KIND_POST_ASK,
    KIND_POST_BID,
    KIND_WITHDRAW,
)
from ..learning.obs import BUNDLE_PAIRS, PRICE_GRID
from .base import Agent

SUITS_IDX = tuple(range(N_SUITS))
SUIT_NAMES = SUITS


class HumanAgent(Agent):
    name = "human"

    def __init__(self, seed: int = 0, show_belief: bool = True, n_particles: int = 512):
        self.seed = seed
        self.show_belief = show_belief
        self.n_particles = n_particles
        self.belief: Optional[BeliefTracker] = None

    def new_round(self, state, player: int) -> None:
        self.player = player
        self.belief = BeliefTracker(self.seed, self.n_particles)
        self.belief.init(state, player)

    def observe(self, state) -> None:
        self.belief.observe(state)

    def _menu(self, state) -> list[tuple[str, Action]]:
        player = self.player
        cash = state.cash[player]
        hand = state.hands[player]
        own = state.quotes.get(player, {})
        items: list[tuple[str, Action]] = [(f"PASS", Action(KIND_PASS))]
        for s in SUITS_IDX:
            name = SUIT_NAMES[s]
            ba = state.best_ask(s)
            if ba is not None and ba[1] != player and cash >= ba[0]:
                items.append((f"BUY {name} @ {ba[0]:g} from P{ba[1]}", Action(KIND_ACCEPT_ASK, s)))
            bb = state.best_bid(s)
            if bb is not None and bb[1] != player and hand[s] >= 1:
                items.append((f"SELL {name} @ {bb[0]:g} to P{bb[1]} (accept bid)", Action(KIND_ACCEPT_BID, s)))
            for price in PRICE_GRID:
                if cash >= price:
                    items.append((f"BID {name} @ {price:g}", Action(KIND_POST_BID, s, price)))
                if hand[s] >= 1:
                    items.append((f"ASK {name} @ {price:g}", Action(KIND_POST_ASK, s, price)))
            if s in own:
                items.append((f"WITHDRAW {name}", Action(KIND_WITHDRAW, s)))
        for s1, s2 in BUNDLE_PAIRS:
            n1, n2 = SUIT_NAMES[s1], SUIT_NAMES[s2]
            for price in PRICE_GRID:
                if cash >= price:
                    items.append((f"BUNDLE BID {n1}+{n2} @ {price:g}", Action(KIND_BUNDLE_BID, suit=s1, suit2=s2, price=price)))
        return items

    def choose_action(self, engine, state, player: int) -> Action:
        self.player = player
        items = self._menu(state)
        self._render(state, items)
        return items[self._prompt(len(items))][1]

    def _render(self, state, items: list[tuple[str, Action]]) -> None:
        player = self.player
        print("\n" + "=" * 64)
        print(f"TICK {state.tick}/{state.config.n_ticks}   time left {state.time_remaining()}   you are P{player}")
        print(f"your hand: {self._fmt_hand(state.hands[player])}   cash: ${state.cash[player]:g}")
        if self.show_belief and self.belief is not None:
            gm = self.belief.goal_marginals()
            line = ", ".join(f"{SUIT_NAMES[s]}: {gm[s]:.0%}" for s in SUITS_IDX)
            print(f"belief goal suit: {line}")
        for s in SUITS_IDX:
            name = SUIT_NAMES[s]
            bb = state.best_bid(s)
            ba = state.best_ask(s)
            bid = f"${bb[0]:g} (P{bb[1]})" if bb else "   --   "
            ask = f"${ba[0]:g} (P{ba[1]})" if ba else "   --   "
            print(f"  {name:<7} best bid {bid:<13} best ask {ask}")
        if state.trades:
            print(f"last trades: {self._fmt_trades(state.trades[-4:])}")

    def _prompt(self, n: int) -> int:
        while True:
            try:
                idx = int(input(f"choose 0..{n - 1} > ").strip())
            except (ValueError, EOFError):
                print("invalid input")
                continue
            if 0 <= idx < n:
                return idx
            print("out of range")

    @staticmethod
    def _fmt_hand(hand) -> str:
        return " ".join(f"{SUIT_NAMES[s]}x{c}" for s, c in enumerate(hand) if c)

    @staticmethod
    def _fmt_trades(trades) -> str:
        return " | ".join(f"P{t.buyer} buy {SUIT_NAMES[t.suit]} @ ${t.price:g} from P{t.seller}" for t in trades)
