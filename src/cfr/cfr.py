"""Chance-sampled Counterfactual Regret Minimization (CFR) for Figgie-Lite.

The game is the two-player zero-sum reduction in figgie_lite.py. Each tick is
sequentialized (player 0 then player 1), and player 1's info set spans all of
player 0's simultaneous actions at that tick. The public history records, per
tick, both actions AND both buy outcomes (in the real game the card exchange
is public), so info sets reflect everything a player observes.

CFR traverses the full game tree with chance (world x deal) sampled; regrets
are updated with the opponent reach weight and the average strategy with the
player's own reach weight. A Strategy is any object exposing
    probs(player, hand, history) -> list[float]
so reference (hand-crafted) and learned (dictionary) strategies are treated
uniformly by the exact evaluators.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Optional

from .figgie_lite import (
    ACTION_COUNT,
    HAND_INDEX,
    REACHABLE_HANDS,
    SPLITS,
    info_key,
    apply_buy,
    encode_joint,
    payoff,
)


def uniform(n: int) -> list[float]:
    return [1.0 / n] * n


class Strategy:
    def probs(self, player: int, hand: tuple, history: tuple) -> list[float]:
        raise NotImplementedError


class DictStrategy(Strategy):
    """CFR's learned average strategy, stored as a sparse table."""

    def __init__(self, table: dict[tuple, list[float]]):
        self.table = table

    def probs(self, player: int, hand: tuple, history: tuple) -> list[float]:
        return self.table.get((player, HAND_INDEX[hand], history),
                              uniform(ACTION_COUNT))


class FuncStrategy(Strategy):
    """A hand-crafted reference strategy defined by a pure function."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def probs(self, player: int, hand: tuple, history: tuple) -> list[float]:
        return self.fn(player, hand, history)


class CFR:
    def __init__(self, ticks: int = 3, seed: Optional[int] = None):
        self.ticks = ticks
        self.rng = random.Random(seed)
        self.reg: dict[tuple, list[float]] = {}
        self.avg: dict[tuple, list[float]] = {}

    # ------------------------------------------------------------ strategy

    def _regret_match(self, key: tuple) -> list[float]:
        r = self.reg.get(key)
        if r is None or sum(max(x, 0.0) for x in r) <= 0:
            return uniform(ACTION_COUNT)
        s = [max(x, 0.0) for x in r]
        tot = sum(s)
        return [x / tot for x in s]

    # -------------------------------------------------------------- iterate

    def _walk(self, hands, cash, goal: int, hist: tuple, tick: int,
              pi0: float, pi1: float) -> float:
        """One chance-sampled CFR traversal; both players updated in place.

        Returns player 0's expected payoff for this chance outcome. The tree
        is small enough to enumerate both players' actions exactly at every
        node (the only sampling is the chance node).
        """
        if tick >= self.ticks:
            return payoff(hands, cash, goal)

        h0, h1 = tuple(hands[0]), tuple(hands[1])
        k0 = info_key(0, h0, hist)
        k1 = info_key(1, h1, hist)
        s0 = self._regret_match(k0)
        s1 = self._regret_match(k1)

        # child[a0][a1] = player-0 payoff of the continuation after (a0, a1).
        child = [[0.0] * ACTION_COUNT for _ in range(ACTION_COUNT)]
        for a0 in range(ACTION_COUNT):
            h_, c_, suc0 = apply_buy(hands, cash, 0, a0)
            for a1 in range(ACTION_COUNT):
                h2_, c2_, suc1 = apply_buy(h_, c_, 1, a1)
                joint = encode_joint(a0, a1, suc0, suc1)
                child[a0][a1] = self._walk(
                    h2_, c2_, goal, hist + (joint,), tick + 1,
                    pi0 * s0[a0], pi1 * s1[a1])

        # player 0's node (opponent reach pi1 weights the regret update).
        vals0 = [sum(s1[a1] * child[a0][a1] for a1 in range(ACTION_COUNT))
                 for a0 in range(ACTION_COUNT)]
        v0 = sum(s0[a0] * vals0[a0] for a0 in range(ACTION_COUNT))
        r0 = self.reg.setdefault(k0, [0.0] * ACTION_COUNT)
        a0m = self.avg.setdefault(k0, [0.0] * ACTION_COUNT)
        for a in range(ACTION_COUNT):
            r0[a] += pi1 * (vals0[a] - v0)
            a0m[a] += pi0 * s0[a]

        # player 1's node (opponent reach pi0 weights the regret update).
        # vals1 / v1 are in player 0's payoff units; player 1 maximizes
        # -payoff, so its regret is (v1 - vals1[a]).
        vals1 = [sum(s0[a0] * child[a0][a1] for a0 in range(ACTION_COUNT))
                 for a1 in range(ACTION_COUNT)]
        v1 = sum(s1[a1] * vals1[a1] for a1 in range(ACTION_COUNT))
        r1 = self.reg.setdefault(k1, [0.0] * ACTION_COUNT)
        a1m = self.avg.setdefault(k1, [0.0] * ACTION_COUNT)
        for a in range(ACTION_COUNT):
            r1[a] += pi0 * (v1 - vals1[a])
            a1m[a] += pi1 * s1[a]

        return v0

    def iterate(self, iterations: int, progress_every: Optional[int] = None,
                report: Optional[Callable[[int, dict], None]] = None) -> None:
        for it in range(iterations):
            goal = self.rng.randrange(4)
            h0, h1 = SPLITS[goal][self.rng.randrange(len(SPLITS[goal]))]
            hands = [list(h0), list(h1)]
            cash = [100.0, 100.0]
            self._walk(hands, cash, goal, (), 0, 1.0, 1.0)
            if progress_every and report and (it + 1) % progress_every == 0:
                report(it + 1, self.exploitability(DictStrategy(self.average_strategy())))

    # ----------------------------------------------------- average strategy

    def average_strategy(self) -> dict[tuple, list[float]]:
        out: dict[tuple, list[float]] = {}
        for key, a in self.avg.items():
            tot = sum(a)
            out[key] = [x / tot for x in a] if tot > 0 else uniform(ACTION_COUNT)
        return out

    # ------------------------------------------------------------- queries

    def strategy_value(self, strat: Strategy) -> float:
        """Exact expected payoff of the strategy pair (for player 0)."""
        total = 0.0
        for goal in range(4):
            for h0, h1 in SPLITS[goal]:
                v = self._value(strat, [list(h0), list(h1)], [100.0, 100.0],
                                goal, (), 0)
                total += v / (4 * len(SPLITS[goal]))
        return total

    def _value(self, strat: Strategy, hands, cash, goal, hist, tick) -> float:
        if tick >= self.ticks:
            return payoff(hands, cash, goal)
        h0, h1 = tuple(hands[0]), tuple(hands[1])
        s0 = strat.probs(0, h0, hist)
        s1 = strat.probs(1, h1, hist)
        v = 0.0
        for a0 in range(ACTION_COUNT):
            h_, c_, suc0 = apply_buy(hands, cash, 0, a0)
            inner = 0.0
            for a1 in range(ACTION_COUNT):
                h2_, c2_, suc1 = apply_buy(h_, c_, 1, a1)
                inner += s1[a1] * self._value(
                    strat, h2_, c2_, goal,
                    hist + (encode_joint(a0, a1, suc0, suc1),), tick + 1)
            v += s0[a0] * inner
        return v

    def best_response(self, u: int, strat_opp: Strategy) -> float:
        """Exact best-response value for player u against strat_opp.

        The opponent plays strat_opp; player u best-responds with ONE pure
        action per info set. Because an info set spans several chance outcomes
        (same private hand, different worlds), the value of each action is the
        expected continuation payoff under u's belief at that info set, which
        must weight each deal by the opponent's reach along the *observed*
        public history (Bayesian update on the opponent's past actions). The
        optimal pure strategy is built by a backward pass, deepest decision
        level first, so the continuation at every info set is already fixed.
        """
        ticks = self.ticks
        sigma: dict[tuple, int] = {}

        for t in range(ticks - 1, -1, -1):
            V: dict[tuple, list[float]] = {}

            def walk(hands, cash, goal, hist, tick, reach) -> float:
                # `reach` is the opponent's reach along `hist`: the product of
                # the opponent's action probabilities for the actions recorded
                # in the observed history. Returns u's payoff expectation from
                # this point under (sigma at deeper levels, strat_opp here).
                if tick >= ticks:
                    return payoff(hands, cash, goal) if u == 0 else -payoff(hands, cash, goal)
                h0, h1 = tuple(hands[0]), tuple(hands[1])
                k0 = info_key(0, h0, hist)
                k1 = info_key(1, h1, hist)
                s0 = strat_opp.probs(0, h0, hist)
                s1 = strat_opp.probs(1, h1, hist)
                child = [[0.0] * ACTION_COUNT for _ in range(ACTION_COUNT)]
                for a0 in range(ACTION_COUNT):
                    h_, c_, suc0 = apply_buy(hands, cash, 0, a0)
                    for a1 in range(ACTION_COUNT):
                        h2_, c2_, suc1 = apply_buy(h_, c_, 1, a1)
                        r = reach * (s1[a1] if u == 0 else s0[a0])
                        child[a0][a1] = walk(
                            h2_, c2_, goal, hist + (encode_joint(a0, a1, suc0, suc1),),
                            tick + 1, r)
                if tick == t:
                    if u == 0:
                        vals = [sum(s1[a1] * child[a0][a1] for a1 in range(ACTION_COUNT))
                                for a0 in range(ACTION_COUNT)]
                        k = k0
                    else:
                        vals = [sum(s0[a0] * child[a0][a1] for a0 in range(ACTION_COUNT))
                                for a1 in range(ACTION_COUNT)]
                        k = k1
                    V.setdefault(k, [0.0] * ACTION_COUNT)
                    for a in range(ACTION_COUNT):
                        V[k][a] += reach * vals[a]
                    return vals[sigma.get(k, 0)]
                # Pass-through: follow the already-fixed deeper sigma.
                if u == 0:
                    a0 = sigma.get(k0, 0)
                    return sum(s1[a1] * child[a0][a1] for a1 in range(ACTION_COUNT))
                a1 = sigma.get(k1, 0)
                return sum(s0[a0] * child[a0][a1] for a0 in range(ACTION_COUNT))

            for goal in range(4):
                for h0, h1 in SPLITS[goal]:
                    walk([list(h0), list(h1)], [100.0, 100.0], goal, (), 0, 1.0)
            for k, v in V.items():
                sigma[k] = max(range(ACTION_COUNT), key=lambda a: v[a])

        # Expected payoff with u playing the built pure BR.
        def final_value(hands, cash, goal, hist, tick):
            if tick >= ticks:
                return payoff(hands, cash, goal) if u == 0 else -payoff(hands, cash, goal)
            h0, h1 = tuple(hands[0]), tuple(hands[1])
            s0 = strat_opp.probs(0, h0, hist)
            s1 = strat_opp.probs(1, h1, hist)
            if u == 0:
                a0 = sigma.get(info_key(0, h0, hist), 0)
                val = 0.0
                for a1 in range(ACTION_COUNT):
                    h_, c_, suc0 = apply_buy(hands, cash, 0, a0)
                    h2_, c2_, suc1 = apply_buy(h_, c_, 1, a1)
                    val += s1[a1] * final_value(
                        h2_, c2_, goal, hist + (encode_joint(a0, a1, suc0, suc1),), tick + 1)
                return val
            a1 = sigma.get(info_key(1, h1, hist), 0)
            val = 0.0
            for a0 in range(ACTION_COUNT):
                h_, c_, suc0 = apply_buy(hands, cash, 0, a0)
                h2_, c2_, suc1 = apply_buy(h_, c_, 1, a1)
                val += s0[a0] * final_value(
                    h2_, c2_, goal, hist + (encode_joint(a0, a1, suc0, suc1),), tick + 1)
            return val

        total = 0.0
        for goal in range(4):
            for h0, h1 in SPLITS[goal]:
                total += final_value([list(h0), list(h1)], [100.0, 100.0],
                                     goal, (), 0) / (4 * len(SPLITS[goal]))
        return total

    def exploitability(self, strat: Strategy) -> dict:
        """Total exploitability of the strategy pair plus per-player parts.

        For the two-player zero-sum reduction:
            eps0 = br0 - value_avg   (player 0's loss)
            eps1 = br1 + value_avg   (player 1's loss, br1 in player-1 units)
        """
        value = self.strategy_value(strat)
        br0 = self.best_response(0, strat)
        br1 = self.best_response(1, strat)
        eps0 = br0 - value
        eps1 = br1 + value
        return {"value": value, "br0": br0, "br1": br1, "eps0": eps0,
                "eps1": eps1, "eps": eps0 + eps1}


# --------------------------------------------------------- reference strategies

def reference_uniform_actions(p: int, hand: tuple, hist: tuple) -> list[float]:
    return uniform(ACTION_COUNT)


def reference_pass(p: int, hand: tuple, hist: tuple) -> list[float]:
    probs = [0.0] * ACTION_COUNT
    probs[0] = 1.0
    return probs


def reference_belief_buy(p: int, hand: tuple, hist: tuple) -> list[float]:
    """Buy the most-believed suit (argmax of the hand's per-suit counts).

    A unique max signals confidence in the goal suit (mirrors the heuristic's
    buy-if-confident rule); ties fall back to PASS.
    """
    counts = list(hand)
    mx = max(counts)
    top = [s for s in range(4) if counts[s] == mx]
    probs = [0.0] * ACTION_COUNT
    if len(top) == 1 and mx > 0:
        probs[top[0] + 1] = 1.0
    else:
        probs[0] = 1.0
    return probs


def reference_belief_with_signals(p: int, hand: tuple, hist: tuple) -> list[float]:
    """Belief-buy that also reacts to the opponent's past buy signals: buy a
    suit the opponent just tried to buy (public signal) if we do not hold it,
    else stick to the argmax-of-hand rule."""
    counts = list(hand)
    if len(hist) > 0:
        last_joint = hist[-1]
        a1 = (last_joint >> 2) // ACTION_COUNT
        if a1 != 0 and counts[a1 - 1] == 0:
            probs = [0.0] * ACTION_COUNT
            probs[a1] = 1.0
            return probs
    mx = max(counts)
    top = [s for s in range(4) if counts[s] == mx]
    probs = [0.0] * ACTION_COUNT
    if len(top) == 1 and mx > 0:
        probs[top[0] + 1] = 1.0
    else:
        probs[0] = 1.0
    return probs


# ----------------------------------------------------------- head-to-head

def play_match(strat0: Strategy, strat1: Strategy, ticks: int, n: int,
               seed: Optional[int] = None) -> float:
    """Mean player-0 payoff over `n` sampled games (P0 uses strat0)."""
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n):
        goal = rng.randrange(4)
        h0, h1 = SPLITS[goal][rng.randrange(len(SPLITS[goal]))]
        hands = [list(h0), list(h1)]
        cash = [100.0, 100.0]
        hist: tuple = ()
        for tick in range(ticks):
            a0 = _sample(strat0.probs(0, tuple(hands[0]), hist), rng)
            a1 = _sample(strat1.probs(1, tuple(hands[1]), hist), rng)
            hands, cash, s0 = apply_buy(hands, cash, 0, a0)
            hands, cash, s1 = apply_buy(hands, cash, 1, a1)
            hist += (encode_joint(a0, a1, s0, s1),)
        total += payoff(hands, cash, goal)
    return total / n


def _sample(probs: list[float], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for a, p in enumerate(probs):
        acc += p
        if r <= acc:
            return a
    return len(probs) - 1
