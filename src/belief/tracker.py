"""Exact card-flow belief tracking for Figgie.

Hidden state of a round: the suit->count assignment and the opponents' initial
hands (our own hand is fully observed). Every trade is public, so each player's
current hand is its initial hand shifted by the observed trade flow. A world is
consistent with everything seen if no observed sale ever made its seller's hand
negative; this is captured per player and suit by `needed` = the maximum
cumulative drawdown, which the world's initial hand must dominate.

The assignment posterior is computed EXACTLY by dynamic programming
(see exact.py): P(a) is proportional to the hand-likelihood prior times the
number of deals of the remaining cards that satisfy every opponent's `needed`
bound. This has no Monte-Carlo noise or sampling bias.

Particles are kept only to sample consistent worlds for M2 IS-MCTS
determinizations (sample_world): each particle is a fully consistent world
(assignment + all current hands), seeded from the prior and re-conditioned on
the accumulated `needed` bound whenever too few survive.

The posterior is strategy-free: it weighs worlds by "could this world have
produced everything we saw", not by how likely a particular opponent strategy
was. The calibration harness (src/eval/calibrate.py) measures whether this
model is calibrated on real agent play.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..env.deck import ALL_ASSIGNMENTS, N_SUITS, cards_per_player, hand_likelihood
from .exact import assignment_posterior as _exact_posterior


class BeliefTracker:
    def __init__(self, seed: Optional[int] = None, n_particles: int = 512):
        self.rng = np.random.default_rng(seed)
        self.n_particles = n_particles
        self.reset()

    def reset(self) -> None:
        self.player = -1
        self.own_hand: np.ndarray = np.zeros(N_SUITS, dtype=int)
        self._n_players = 0
        self._flows: list[np.ndarray] = []
        self._needed: list[np.ndarray] = []
        self._trades: list = []
        self._seen = 0
        self._prior: Optional[np.ndarray] = None
        self.assign_idx = np.zeros(self.n_particles, dtype=int)
        self.hands = np.zeros((self.n_particles, 0, N_SUITS), dtype=int)
        self.count = 0

    # ---------------------------------------------------------------- setup

    def init(self, state, player: int) -> None:
        self.reset()
        self.player = player
        self.own_hand = np.array(state.hands[player], dtype=int)
        n = state.config.n_players
        self._flows = [np.zeros(N_SUITS, dtype=int) for _ in range(n)]
        self._needed = [np.zeros(N_SUITS, dtype=int) for _ in range(n)]
        self._sample_particles(n)

    # -------------------------------------------------------------- updates

    def observe(self, state) -> None:
        for trade in state.trades[self._seen:]:
            self.update(trade)
        self._seen = len(state.trades)

    def update(self, trade) -> None:
        self._trades.append(trade)
        buyer, seller, suit = trade.buyer, trade.seller, trade.suit
        self._flows[buyer][suit] += 1
        self._flows[seller][suit] -= 1
        self._needed[seller][suit] = max(self._needed[seller][suit], -self._flows[seller][suit])

        h = self.hands[: self.count]
        h[:, buyer, suit] += 1
        h[:, seller, suit] -= 1
        alive = np.nonzero(h[:, seller, suit] >= 0)[0]
        if alive.size == 0:
            self._reseed()
            return
        self.hands[: alive.size] = self.hands[alive]
        self.assign_idx[: alive.size] = self.assign_idx[alive]
        self.count = alive.size
        if self.count < self.n_particles * 0.5:
            self._reseed()

    # ------------------------------------------------------------ posterior

    def set_assignment_prior(self, prior: Optional[np.ndarray]) -> None:
        """Override the hand-likelihood prior with a learned belief (P2). The
        exact flow-consistency factor is still applied in assignment_posterior
        and world sampling is reweighted by this prior. Pass None to clear."""
        self._prior = None if prior is None else np.asarray(prior, dtype=float)

    def assignment_posterior(self) -> np.ndarray:
        opponents = [self._needed[j] for j in range(self._n_players) if j != self.player]
        return _exact_posterior(self.own_hand.tolist(), opponents, self._n_players, prior=self._prior)

    def goal_marginals(self) -> np.ndarray:
        post = self.assignment_posterior()
        gm = np.zeros(N_SUITS)
        for ai, p in enumerate(post):
            gm[ALL_ASSIGNMENTS[ai].goal_suit] += p
        return gm

    def sample_world(self) -> tuple:
        if self.count == 0:
            raise RuntimeError("no consistent worlds remain")
        if self._prior is None:
            i = int(self.rng.integers(0, self.count))
        else:
            w = self._prior[self.assign_idx[: self.count]]
            total = w.sum()
            if total > 0:
                i = int(self.rng.choice(self.count, p=w / total))
            else:
                i = int(self.rng.integers(0, self.count))
        assignment = ALL_ASSIGNMENTS[self.assign_idx[i]]
        hands = self.hands[i].tolist()
        return assignment, hands

    @property
    def consistent(self) -> bool:
        return self.count > 0

    # --------------------------------------------------------------- internals

    def _sample_particles(self, n_players: int) -> None:
        per = cards_per_player(n_players)
        prior = np.array([hand_likelihood(self.own_hand.tolist(), a.counts) for a in ALL_ASSIGNMENTS], dtype=float)
        prior /= prior.sum()
        self._n_players = n_players
        hands: list[np.ndarray] = []
        assign_idx: list[int] = []
        budget = self.n_particles * 4
        while len(hands) < self.n_particles and budget > 0:
            ai = int(self.rng.choice(len(ALL_ASSIGNMENTS), p=prior))
            h = self._sample_initial(ai, n_players, per)
            if h is not None:
                hands.append(h)
                assign_idx.append(ai)
            budget -= 1
        self.hands = np.stack(hands) if hands else np.zeros((0, n_players, N_SUITS), dtype=int)
        self.assign_idx = np.array(assign_idx, dtype=int)
        self.count = len(hands)

    def _reseed(self) -> None:
        n = self._n_players
        self._sample_particles(n)
        for trade in self._trades:
            self.hands[:, trade.buyer, trade.suit] += 1
            self.hands[:, trade.seller, trade.suit] -= 1

    def _sample_initial(self, ai: int, n_players: int, per: int) -> Optional[np.ndarray]:
        counts = ALL_ASSIGNMENTS[ai].counts
        remaining = np.array(counts) - self.own_hand
        if (remaining < 0).any():
            return None
        opps = [j for j in range(n_players) if j != self.player]
        needs = np.stack([self._needed[j] for j in opps])
        if (needs.sum(axis=1) > per).any() or (needs.sum(axis=0) > remaining).any():
            return None
        for _attempt in range(8):
            order = list(opps)
            self.rng.shuffle(order)
            rem = remaining.copy()
            chosen: dict[int, np.ndarray] = {}
            ok = True
            for j in order:
                need = self._needed[j]
                if (need > rem).any():
                    ok = False
                    break
                rem = rem - need
                fill = per - int(need.sum())
                deck = np.repeat(np.arange(N_SUITS), rem)
                self.rng.shuffle(deck)
                extra = np.bincount(deck[:fill], minlength=N_SUITS)
                chosen[j] = need + extra
                rem = rem - extra
            if ok:
                hands = np.zeros((n_players, N_SUITS), dtype=int)
                hands[self.player] = self.own_hand
                for j in opps:
                    hands[j] = chosen[j]
                return hands
        return None
