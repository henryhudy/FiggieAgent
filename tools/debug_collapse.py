"""Debug: compare tracker posterior vs exact enumeration at collapse ticks."""

from __future__ import annotations

import random
import sys

import numpy as np

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.agents.random import RandomAgent
from src.belief.tracker import BeliefTracker
from src.env.deck import ALL_ASSIGNMENTS, N_SUITS, cards_per_player
from src.env.engine import Engine
from src.env.state import RoundConfig


def exact_posterior(own_hand, needed, n_players):
    per = cards_per_player(n_players)
    opps = [j for j in range(n_players) if j != 0]
    counts_per_assignment = []
    for a in ALL_ASSIGNMENTS:
        remaining = np.array(a.counts) - np.array(own_hand)
        if (remaining < 0).any():
            counts_per_assignment.append(0)
            continue
        n = count_allocations(needed, remaining, per, opps, a)
        counts_per_assignment.append(n)
    counts_per_assignment = np.array(counts_per_assignment, dtype=float)
    post = counts_per_assignment / counts_per_assignment.sum()
    return post, counts_per_assignment


def count_allocations(needed, remaining, per, opps, a):
    total = 0

    def rec(i, rem, hands):
        nonlocal total
        if i == len(opps):
            total += 1
            return
        j = opps[i]
        need = needed[j]
        lo = need.copy()
        remaining_slots = per - lo.sum()
        if remaining_slots < 0:
            return
        # enumerate allocations of remaining_slots extra cards across suits
        # given rem caps, such that each opponent (incl. later) can still be met.
        for extra in gen_fills(rem, remaining_slots, i, needed, opps):
            hand = lo + extra
            if (hand > rem).any():
                continue
            new_rem = rem - hand
            if any(new_rem[s] < 0 for s in range(N_SUITS)):
                continue
            rec(i + 1, new_rem, hands + [hand])

    rec(0, remaining.copy(), [])
    return total


def gen_fills(rem, slots, i, needed, opps):
    # generate all 4-vectors 'extra' >=0 summing to slots with extra <= rem - need(hand... )
    out = []

    def rec(s, idx, vec):
        if idx == N_SUITS - 1:
            v = vec + [s]
            if all(v[t] <= rem[t] for t in range(N_SUITS)):
                out.append(np.array(v, dtype=int))
            return
        for k in range(s + 1):
            rec(s - k, idx + 1, vec + [k])

    rec(slots, 0, [])
    return out


def main():
    engine = Engine(RoundConfig(n_ticks=60), random.Random(1))
    rng = random.Random(2)
    agents = [RandomAgent(rng.randrange(1 << 30)) for _ in range(4)]
    found = 0
    for r in range(30):
        state = engine.new_round()
        for p, a in enumerate(agents):
            a.new_round(state, p)
        tr = BeliefTracker(seed=100 + r, n_particles=8192)
        tr.init(state, 0)
        true_ai = ALL_ASSIGNMENTS.index(state.assignment)
        for t in range(engine.config.n_ticks):
            actions = [a.choose_action(engine, state, p) for p, a in enumerate(agents)]
            engine.step(state, actions)
            for a in agents:
                a.observe(state)
            tr.observe(state)
            post = tr.assignment_posterior()
            if post.max() >= 0.9 and int(np.argmax(post)) != true_ai:
                exact, counts = exact_posterior(tr.own_hand.tolist(), tr._needed, 4)
                print(f"round {r} tick {t}: tracker P(argmax)={post.max():.3f} argmax={np.argmax(post)} true={true_ai}")
                print(f"  tracker posterior: {np.round(post, 3)}")
                print(f"  exact   counts  : {counts}")
                print(f"  exact   posterior: {np.round(exact, 3)}")
                print(f"  needed: {[[int(x) for x in row] for row in tr._needed]}")
                found += 1
                if found >= 5:
                    return


if __name__ == "__main__":
    main()
