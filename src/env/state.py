"""Round state and action types for the Figgie simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .deck import Assignment, N_SUITS

KIND_PASS = "PASS"
KIND_POST_BID = "POST_BID"
KIND_POST_ASK = "POST_ASK"
KIND_WITHDRAW = "WITHDRAW"
KIND_ACCEPT_BID = "ACCEPT_BID"
KIND_ACCEPT_ASK = "ACCEPT_ASK"
KIND_BUNDLE_BID = "BUNDLE_BID"

BID = "bid"
ASK = "ask"

BUNDLE_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (i, j) for i in range(N_SUITS) for j in range(i + 1, N_SUITS)
)


@dataclass(frozen=True)
class Action:
    kind: str
    suit: Optional[int] = None
    price: Optional[float] = None
    suit2: Optional[int] = None


@dataclass(frozen=True)
class Trade:
    tick: int
    buyer: int
    seller: int
    suit: int
    price: float
    aggressor: int
    bundle: bool = False


@dataclass
class RoundConfig:
    n_players: int = 4
    start_cash: float = 350.0
    pot: float = 200.0
    n_ticks: int = 120
    tick_seconds: float = 2.0
    max_price: float = 100.0


@dataclass
class RoundState:
    config: RoundConfig
    assignment: Assignment
    hands: list[list[int]]
    cash: list[float]
    quotes: dict[int, dict[int, tuple[str, float]]]
    bundles: dict[int, dict[tuple[int, int], float]] = field(default_factory=dict)
    tick: int = 0
    trades: list[Trade] = field(default_factory=list)

    @property
    def ante(self) -> float:
        return self.config.pot / self.config.n_players

    def time_remaining(self) -> int:
        return self.config.n_ticks - self.tick

    def best_bid(self, suit: int) -> Optional[tuple[float, int]]:
        best = None
        for p, qs in self.quotes.items():
            q = qs.get(suit)
            if q is not None and q[0] == BID and (best is None or q[1] > best[0]):
                best = (q[1], p)
        return best

    def best_ask(self, suit: int) -> Optional[tuple[float, int]]:
        best = None
        for p, qs in self.quotes.items():
            q = qs.get(suit)
            if q is not None and q[0] == ASK and (best is None or q[1] < best[0]):
                best = (q[1], p)
        return best

    def best_bundle_ask(self, pair: tuple[int, int]) -> Optional[tuple[float, int, int]]:
        """Cheapest resting combined ask for the suit pair (s1 < s2).

        Returns (combined_price, seller1, seller2) if both suits have a resting
        ask (sellers may differ), else None. A bundle bid crosses when its total
        price is at least this combined price.
        """
        s1, s2 = pair
        a1 = self.best_ask(s1)
        a2 = self.best_ask(s2)
        if a1 is None or a2 is None:
            return None
        return a1[0] + a2[0], a1[1], a2[1]

    def goal_counts(self) -> list[int]:
        gi = self.assignment.goal_suit
        return [hand[gi] for hand in self.hands]
