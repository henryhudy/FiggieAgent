"""Per-player observation builders."""

from __future__ import annotations

from .deck import N_SUITS
from .state import RoundState


def observe(state: RoundState, player: int) -> dict:
    return {
        "player": player,
        "hand": tuple(state.hands[player]),
        "cash": state.cash[player],
        "tick": state.tick,
        "time_remaining": state.time_remaining(),
        "best_bids": {s: state.best_bid(s) for s in range(N_SUITS)},
        "best_asks": {s: state.best_ask(s) for s in range(N_SUITS)},
        "own_quotes": dict(state.quotes.get(player, {})),
    }
