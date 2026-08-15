"""Exact strategy-free assignment posterior via dynamic programming.

P(assignment a | own hand, trade flows) is proportional to

    hand_likelihood(own_hand, a.counts) * N_consistent(a)

where N_consistent(a) is the number of ways to deal the remaining cards to the
opponents such that every observed sale was physically possible. Because deals
of specific cards are exchangeable, a per-suit split (k0,k1,k2) contributes the
multinomial coefficient r!/(k0! k1! k2!).  A small DP over (suit, opponent
totals) counts exactly this, giving the true strategy-free posterior without
any Monte-Carlo noise.  Used by BeliefTracker for the marginal; particles are
kept only for sampling consistent worlds (M2 determinizations).
"""

from __future__ import annotations

import math

import numpy as np

from ..env.deck import ALL_ASSIGNMENTS, N_SUITS, cards_per_player, hand_likelihood

_PER_SUIT_MULT = {}


def _split_mults(r: int, per: int) -> list[list[float]]:
    """mult[k0][k1] = C(r; k0,k1,r-k0-k1) when valid under per, else None."""
    key = (r, per)
    if key not in _PER_SUIT_MULT:
        table: list[list[float]] = []
        for k0 in range(r + 1):
            row: list[float] = []
            for k1 in range(r - k0 + 1):
                row.append(math.comb(r, k0) * math.comb(r - k0, k1))
            table.append(row)
        _PER_SUIT_MULT[key] = table
    return _PER_SUIT_MULT[key]


def assignment_posterior(own_hand, needed, n_players: int, prior=None) -> np.ndarray:
    """Exact posterior over the 12 assignments for the tracker's hidden state.

    own_hand: list[int] (fully observed). needed: list of per-opponent arrays
    (cumulative drawdown bounds). n_players: total players. prior: optional
    12-vector replacing the hand-likelihood prior (the P2 learned belief); the
    exact flow-consistency factor N_consistent(a) is always applied, so the
    result only ever weights physically possible worlds.
    """
    per = cards_per_player(n_players)
    m = n_players - 1
    w = []
    for i, a in enumerate(ALL_ASSIGNMENTS):
        remaining = np.array(a.counts) - np.asarray(own_hand, dtype=int)
        if (remaining < 0).any():
            w.append(0.0)
            continue
        base = float(hand_likelihood(list(own_hand), a.counts)) if prior is None else float(prior[i])
        n = _count_dp(remaining, needed, per, m)
        w.append(base * n)
    w = np.array(w, dtype=float)
    s = w.sum()
    return w / s if s > 0 else np.zeros_like(w)


def _count_dp(remaining: np.ndarray, needed, per: int, m: int) -> float:
    """Number of deals of `remaining` cards (per suit) to m opponents where
    opponent j gets exactly `per` cards and hand[j] >= needed[j] (elementwise)."""
    total = int(remaining.sum())
    assert total == m * per

    if m == 3:
        return _count_dp_3(remaining, needed, per)
    return _count_dp_generic(remaining, needed, per, m)


def _count_dp_3(remaining: np.ndarray, needed, per: int) -> float:
    """Vectorized DP for three opponents (4 players). State is a (per+1)^3
    array over each opponent's running total; each suit is a masked convolution
    with the multinomial split kernel."""
    shape = (per + 1, per + 1, per + 1)
    dp = np.zeros(shape, dtype=float)
    dp[0, 0, 0] = 1.0
    for s in range(N_SUITS):
        r = int(remaining[s])
        nd = [int(needed[j][s]) for j in range(3)]
        mult = _split_mults(r, per)
        new = np.zeros(shape, dtype=float)
        for k0 in range(max(0, nd[0]), min(r, per) + 1):
            for k1 in range(max(0, nd[1]), min(r - k0, per) + 1):
                k2 = r - k0 - k1
                if k2 < nd[2] or k2 > per:
                    continue
                src = dp[: per + 1 - k0, : per + 1 - k1, : per + 1 - k2]
                new[k0:, k1:, k2:] += src * mult[k0][k1]
        dp = new
    return float(dp[per, per, per])


def _count_dp_generic(remaining: np.ndarray, needed, per: int, m: int) -> float:
    """Non-vectorized fallback for other player counts."""
    total = int(remaining.sum())
    assert total == m * per

    def rec(s, totals):
        if s == N_SUITS:
            return 1.0 if all(t == per for t in totals) else 0.0
        r = int(remaining[s])
        nd = [int(needed[j][s]) for j in range(m)]
        acc = 0.0
        for ks in _splits(r, m, nd, per):
            newt = [totals[j] + ks[j] for j in range(m)]
            if any(t > per for t in newt):
                continue
            mult = 1
            rem = r
            for k in ks:
                mult *= math.comb(rem, k)
                rem -= k
            acc += mult * rec(s + 1, newt)
        return acc

    return rec(0, [0] * m)


def _splits(r: int, m: int, lo, per: int):
    out = []

    def rec(idx, rem, vec):
        if idx == m - 1:
            k = rem
            if lo[idx] <= k <= per:
                out.append(vec + [k])
            return
        for k in range(max(0, lo[idx]), min(rem, per) + 1):
            rec(idx + 1, rem - k, vec + [k])

    rec(0, r, [])
    return out
