"""Observation encoding and action space for the M3 neural agent.

The agent plays a reduced, buy-only abstraction (DESIGN.md §6.3, M2's working
regime): PASS, ACCEPT_ASK(s), POST_BID(s, p) on a coarse price grid. The policy
head covers this fixed 21-action space; legality (including belief top-k
gating) is applied as a mask in search and training.

The observation is a flat 41-float vector (DESIGN.md §6.1), all values
normalized. Trade-per-suit counts and the recent trade-history window are part
of the public history and are constant inside a simulation, so they are
computed once per decision and passed to the net separately
(`trade_counts`, `history_features`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..env.deck import ALL_ASSIGNMENTS, N_SUITS
from ..env.state import (
    Action,
    BID,
    BUNDLE_PAIRS,
    KIND_ACCEPT_ASK,
    KIND_BUNDLE_BID,
    KIND_PASS,
    KIND_POST_BID,
    RoundState,
)

PRICE_GRID = (6, 12, 18, 24)
N_PRICES = len(PRICE_GRID)
N_PAIRS = len(BUNDLE_PAIRS)
# PASS + per-suit (ACCEPT_ASK + bids) + bundle bids (pair x price)
N_ACTIONS = 1 + N_SUITS * (1 + N_PRICES) + N_PAIRS * N_PRICES

N_HISTORY = 8
HIST_FEATS = 5  # suit, price, tick, aggressor-is-buyer, involves-self
HIST_DIM = N_HISTORY * HIST_FEATS

_NORM_HAND = 10.0
_NORM_CASH = 300.0
_NORM_PRICE = 100.0
_NORM_TRADE = 20.0

# ------------------------------------------------------------- action space

_PRICE_INDEX = {p: i for i, p in enumerate(PRICE_GRID)}
_PAIR_INDEX = {pair: i for i, pair in enumerate(BUNDLE_PAIRS)}
_N_BASE = 1 + N_SUITS * (1 + N_PRICES)


def action_index(action: Action) -> int:
    """Map an abstract buy-only action to its fixed index."""
    if action.kind == KIND_PASS:
        return 0
    s = action.suit
    if action.kind == KIND_ACCEPT_ASK:
        return 1 + s * (1 + N_PRICES)
    if action.kind == KIND_POST_BID:
        return 1 + s * (1 + N_PRICES) + 1 + _PRICE_INDEX[int(action.price)]
    if action.kind == KIND_BUNDLE_BID:
        return _N_BASE + _PAIR_INDEX[tuple(sorted((s, action.suit2)))] * N_PRICES + _PRICE_INDEX[int(action.price)]
    raise ValueError(f"action {action} not in the buy-only abstraction")


def index_action(i: int) -> Action:
    if i == 0:
        return Action(KIND_PASS)
    if i < _N_BASE:
        i -= 1
        s, r = divmod(i, 1 + N_PRICES)
        if r == 0:
            return Action(KIND_ACCEPT_ASK, s)
        return Action(KIND_POST_BID, s, float(PRICE_GRID[r - 1]))
    r = i - _N_BASE
    pair_idx, k = divmod(r, N_PRICES)
    s1, s2 = BUNDLE_PAIRS[pair_idx]
    return Action(KIND_BUNDLE_BID, suit=s1, suit2=s2, price=float(PRICE_GRID[k]))


# ------------------------------------------------------------- observation

def trade_counts(state: RoundState) -> np.ndarray:
    counts = np.zeros(N_SUITS, dtype=int)
    for t in state.trades:
        counts[t.suit] += 1
    return counts


def history_features(state: RoundState, player: int, n_history: int = N_HISTORY) -> np.ndarray:
    """Flattened window of the last `n_history` trades (zero-padded at front).

    Each trade -> HIST_FEATS floats: suit/4, price/100, tick/n_ticks,
    aggressor-is-buyer, and whether the seat was a party to the trade. The
    window is part of the public history, so it is constant inside a search
    simulation and is captured once per decision (see module docstring).
    """
    n = state.config.n_players
    feats = np.zeros((n_history, HIST_FEATS), dtype=np.float32)
    for i, t in enumerate(state.trades[-n_history:]):
        row = feats[i]
        row[0] = (t.suit + 0.5) / N_SUITS
        row[1] = t.price / _NORM_PRICE
        row[2] = t.tick / state.config.n_ticks
        row[3] = 1.0 if t.aggressor == t.buyer else 0.0
        row[4] = 1.0 if (t.buyer == player or t.seller == player) else 0.0
    return feats.reshape(-1)


def legal_mask(state: RoundState, player: int, top_suits: Optional[list[int]] = None) -> np.ndarray:
    """Boolean mask over the action space for the current state."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[0] = True
    cash = state.cash[player]
    top = top_suits if top_suits is not None else list(range(N_SUITS))
    for s in top:
        ba = state.best_ask(s)
        if ba is not None and ba[1] != player and cash >= ba[0]:
            mask[1 + s * (1 + N_PRICES)] = True
        for k in range(N_PRICES):
            if cash >= PRICE_GRID[k]:
                mask[1 + s * (1 + N_PRICES) + 1 + k] = True
    for pair_idx, pair in enumerate(BUNDLE_PAIRS):
        for k in range(N_PRICES):
            if cash >= PRICE_GRID[k]:
                mask[_N_BASE + pair_idx * N_PRICES + k] = True
    return mask


def encode(
    state: RoundState,
    player: int,
    gm: np.ndarray,
    post: np.ndarray,
    trade_counts_per_suit: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Encode the observation vector for `player` (see module docstring).

    The first 41 floats are the M4/P1/P2 layout (hand, book, cash, time, goal
    marginals, assignment posterior, own bids, trade counts); the last 12 are
    the P1-bundle extension: per pair the cheapest combined resting ask and a
    flag for the player's own resting bundle bid. Appending keeps older
    observation slices byte-compatible for warm-starting.
    """
    n = state.config.n_players
    out = np.zeros(53, dtype=np.float32)
    hand = state.hands[player]
    out[0:4] = np.array(hand, dtype=np.float32) / _NORM_HAND
    for s in range(N_SUITS):
        bb = state.best_bid(s)
        ba = state.best_ask(s)
        out[4 + s] = (bb[0] / _NORM_PRICE) if bb is not None else 0.0
        out[8 + s] = (ba[0] / _NORM_PRICE) if ba is not None else 0.0
    out[12] = state.cash[player] / _NORM_CASH
    oi = 13
    for p in range(n):
        if p != player:
            out[oi] = state.cash[p] / _NORM_CASH
            oi += 1
    out[16] = state.time_remaining() / state.config.n_ticks
    out[17:21] = np.asarray(gm, dtype=np.float32)
    out[21:33] = np.asarray(post, dtype=np.float32)
    own = state.quotes.get(player, {})
    for s in range(N_SUITS):
        q = own.get(s)
        out[33 + s] = 1.0 if q is not None and q[0] == BID else 0.0
    if trade_counts_per_suit is not None:
        out[37:41] = np.asarray(trade_counts_per_suit, dtype=np.float32) / _NORM_TRADE
    own_bundles = state.bundles.get(player, {})
    for pair_idx, pair in enumerate(BUNDLE_PAIRS):
        ba = state.best_bundle_ask(pair)
        out[41 + pair_idx] = (ba[0] / (2 * _NORM_PRICE)) if ba is not None else 0.0
        out[47 + pair_idx] = 1.0 if pair in own_bundles else 0.0
    return out


def goal_marginals_from_posterior(post: np.ndarray) -> np.ndarray:
    gm = np.zeros(N_SUITS)
    for ai, p in enumerate(post):
        gm[ALL_ASSIGNMENTS[ai].goal_suit] += p
    return gm
