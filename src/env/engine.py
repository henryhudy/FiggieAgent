"""Game engine: deterministic order-book simulation of one Figgie round.

The continuous 4-minute trading phase is discretized into `n_ticks`
simultaneous-move turns. Resolution is documented and deterministic:

1. Phase 1 (quotes): each player's POST/WITHDRAW updates the book in player
   order. A post replaces the player's own quote on that suit. An ASK requires
   holding a card of the suit; a BID requires the cash to cover it, otherwise
   the post is a no-op.
2. Phase 2 (execution): in player order, ACCEPT_ASK/ACCEPT_BID hit the best
   opposing quote, and POST_BID/POST_ASK that cross the spread execute at the
   resting quote's price. Execution price is the resting limit, not midpoint.

Simultaneous conflicts are resolved by player order; ties on price go to the
lowest player index. Self-trading is prohibited.
"""

from __future__ import annotations

import random
from typing import Optional

from .deck import CARD_VALUE, N_SUITS, deal, sample_assignment
from .state import (
    Action,
    ASK,
    BID,
    BUNDLE_PAIRS,
    KIND_ACCEPT_ASK,
    KIND_ACCEPT_BID,
    KIND_BUNDLE_BID,
    KIND_PASS,
    KIND_POST_ASK,
    KIND_POST_BID,
    KIND_WITHDRAW,
    RoundConfig,
    RoundState,
    Trade,
)

PRICE_GRID = tuple(range(1, 26))


class Engine:
    def __init__(self, config: Optional[RoundConfig] = None, rng: Optional[random.Random] = None):
        self.config = config or RoundConfig()
        self.rng = rng or random.Random()

    def new_round(self) -> RoundState:
        assignment = sample_assignment(self.rng)
        hands = deal(self.rng, self.config.n_players, assignment)
        cash = [self.config.start_cash - self.config.pot / self.config.n_players] * self.config.n_players
        return RoundState(self.config, assignment, hands, cash, {})

    def step(self, state: RoundState, actions: list[Action]) -> None:
        for p, a in enumerate(actions):
            if a.kind == KIND_POST_BID and a.price is not None:
                if state.cash[p] >= a.price:
                    state.quotes.setdefault(p, {})[a.suit] = (BID, a.price)
            elif a.kind == KIND_POST_ASK and a.price is not None:
                if state.hands[p][a.suit] >= 1:
                    state.quotes.setdefault(p, {})[a.suit] = (ASK, a.price)
            elif a.kind == KIND_BUNDLE_BID and a.price is not None and a.suit2 is not None:
                if state.cash[p] >= a.price:
                    state.bundles.setdefault(p, {})[tuple(sorted((a.suit, a.suit2)))] = a.price
            elif a.kind == KIND_WITHDRAW:
                state.quotes.get(p, {}).pop(a.suit, None)

        for p, a in enumerate(actions):
            if a.kind == KIND_ACCEPT_ASK:
                self._execute(state, p, a.suit, buyer=p)
            elif a.kind == KIND_ACCEPT_BID:
                self._execute(state, p, a.suit, seller=p)
            elif a.kind == KIND_POST_BID and a.price is not None:
                self._execute(state, p, a.suit, buyer=p, limit=a.price)
            elif a.kind == KIND_POST_ASK and a.price is not None:
                self._execute(state, p, a.suit, seller=p, limit=a.price)

        self._execute_bundles(state)

        state.tick += 1

    def _execute_bundles(self, state: RoundState) -> None:
        """Fill resting bundle bids against the combined per-suit asks.

        A bundle bid (s1,s2) at total price p crosses when both suits have a
        resting ask summing to <= p. Both legs execute atomically at their
        resting ask prices; the buyer pays the sum, each seller is paid their
        own ask, and both asks are consumed. Runs after the per-suit execution
        loop, in player order, so a bundle fill never races the single-suit
        book within a tick.
        """
        for p in range(state.config.n_players):
            own = state.bundles.get(p, {})
            for pair, total in sorted(own.items()):
                ba = state.best_bundle_ask(pair)
                if ba is None:
                    continue
                combined, seller1, seller2 = ba
                s1, s2 = pair
                if combined > total:
                    continue
                if seller1 == p or seller2 == p:
                    continue
                a1 = state.best_ask(s1)
                a2 = state.best_ask(s2)
                if a1 is None or a2 is None:
                    continue
                price1, seller1 = a1
                price2, seller2 = a2
                if state.cash[p] < price1 + price2:
                    continue
                if state.hands[seller1][s1] < 1 or state.hands[seller2][s2] < 1:
                    continue
                state.cash[p] -= price1 + price2
                state.cash[seller1] += price1
                state.cash[seller2] += price2
                state.hands[p][s1] += 1
                state.hands[seller1][s1] -= 1
                state.hands[p][s2] += 1
                state.hands[seller2][s2] -= 1
                state.quotes[seller1].pop(s1, None)
                state.quotes[seller2].pop(s2, None)
                state.bundles[p].pop(pair, None)
                state.trades.append(Trade(state.tick, p, seller1, s1, price1, aggressor=p, bundle=True))
                state.trades.append(Trade(state.tick, p, seller2, s2, price2, aggressor=p, bundle=True))

    def _execute(
        self,
        state: RoundState,
        player: int,
        suit: int,
        limit: Optional[float] = None,
        buyer: Optional[int] = None,
        seller: Optional[int] = None,
    ) -> None:
        if buyer is not None:
            quote = state.best_ask(suit)
            if quote is None or quote[1] == player:
                return
            price, seller = quote
            if state.hands[seller][suit] < 1:
                state.quotes.get(seller, {}).pop(suit, None)
                return
            if limit is not None and price > limit:
                return
            if state.cash[player] < price:
                return
            state.cash[player] -= price
            state.cash[seller] += price
            state.hands[player][suit] += 1
            state.hands[seller][suit] -= 1
            del state.quotes[seller][suit]
            state.trades.append(Trade(state.tick, player, seller, suit, price, aggressor=player))
        else:
            quote = state.best_bid(suit)
            if quote is None or quote[1] == player:
                return
            price, buyer = quote
            if limit is not None and price < limit:
                return
            if state.hands[player][suit] < 1:
                return
            if state.cash[buyer] < price:
                state.quotes.get(buyer, {}).pop(suit, None)
                return
            state.cash[buyer] -= price
            state.cash[player] += price
            state.hands[buyer][suit] += 1
            state.hands[player][suit] -= 1
            if state.hands[player][suit] == 0:
                q = state.quotes.get(player, {}).get(suit)
                if q is not None and q[0] == ASK:
                    del state.quotes[player][suit]
            del state.quotes[buyer][suit]
            state.trades.append(Trade(state.tick, buyer, player, suit, price, aggressor=player))

    def settle(self, state: RoundState) -> list[float]:
        gc = state.goal_counts()
        bonus = [CARD_VALUE * c for c in gc]
        total_bonus = sum(bonus)
        remainder = state.config.pot - total_bonus
        assert remainder >= -1e-6, f"pot {state.config.pot} < bonuses {total_bonus}"
        mx = max(gc)
        n_winners = gc.count(mx)
        return [
            bonus[i] + (remainder / n_winners if gc[i] == mx else 0.0) - state.ante
            for i in range(len(gc))
        ]

    def legal_actions(self, state: RoundState, player: int) -> list[Action]:
        acts = [Action(KIND_PASS)]
        cash = state.cash[player]
        hand = state.hands[player]
        own = state.quotes.get(player, {})
        for s in range(N_SUITS):
            ba = state.best_ask(s)
            if ba is not None and ba[1] != player and cash >= ba[0]:
                acts.append(Action(KIND_ACCEPT_ASK, s))
            bb = state.best_bid(s)
            if bb is not None and bb[1] != player and hand[s] >= 1:
                acts.append(Action(KIND_ACCEPT_BID, s))
            if s in own:
                acts.append(Action(KIND_WITHDRAW, s))
            for price in PRICE_GRID:
                if cash >= price:
                    acts.append(Action(KIND_POST_BID, s, price))
                if hand[s] >= 1:
                    acts.append(Action(KIND_POST_ASK, s, price))
        for pair in BUNDLE_PAIRS:
            for price in PRICE_GRID:
                if cash >= price:
                    acts.append(Action(KIND_BUNDLE_BID, suit=pair[0], suit2=pair[1], price=price))
        return acts
