"""Probe: how much goal-suit signal do opponents' buys carry?

The strategy-free belief tracker treats trades only as card-flow constraints and
is weakly informative on current opponents (M1 finding). This probe quantifies
the ceiling of the next lever - conditioning beliefs on opponent *behavior*.

It runs rounds with a neutral random-observer target and, by default, heuristic
opponents (a control uses all-random). For every round it records the exact
strategy-free goal marginal and an "oracle" goal marginal that additionally
assumes each opponent-initiated buy of suit s is evidence for goal = s:

    P(buy s | goal = g) = p_g if g == s else p_n,  p_n = (1 - p_g) / 3

The empirical buy-goal alignment rate p_g is measured from the data, so the
oracle is the best a behavior-conditioned belief could do given how much the
opponents' buys actually reveal.  If the oracle barely improves on the
strategy-free baseline, behavior conditioning is not worth building yet.
"""

from __future__ import annotations

import argparse
import random
from typing import Optional

import numpy as np

from ..agents.heuristic import HeuristicAgent
from ..agents.random import RandomAgent
from ..belief.tracker import BeliefTracker
from ..env.deck import ALL_ASSIGNMENTS, N_SUITS
from ..env.engine import Engine
from ..env.state import RoundConfig


def _goal_marginal(post, suits=N_SUITS):
    gm = np.zeros(suits)
    for ai, p in enumerate(post):
        gm[ALL_ASSIGNMENTS[ai].goal_suit] += p
    return gm


def _oracle_marginal(post, buys, p_g, suits=N_SUITS):
    """Reweight the exact assignment posterior by per-suit buy evidence."""
    if not buys:
        return _goal_marginal(post)
    p_n = (1.0 - p_g) / (suits - 1)
    counts = np.bincount(buys, minlength=suits)
    logw = np.zeros(len(ALL_ASSIGNMENTS))
    for ai, a in enumerate(ALL_ASSIGNMENTS):
        for s in range(suits):
            c = counts[s]
            logw[ai] += c * np.log(p_g if a.goal_suit == s else p_n)
    w = post * np.exp(logw - logw.max())
    w /= w.sum()
    return _goal_marginal(w)


def collect(engine, agents, n_rounds: int, seed: int, p_g: float) -> dict:
    gen = np.random.default_rng(seed)
    target = agents[0]
    rec = {
        "base_pairs": [],
        "orc_pairs": [],
        "base_arg": [],
        "orc_arg": [],
        "base_p_true": [],
        "orc_p_true": [],
        "buys": 0,
        "goal_buys": 0,
    }
    for r in range(n_rounds):
        rot = r % len(agents)
        seated = agents[rot:] + agents[:rot]
        state = engine.new_round()
        for p, a in enumerate(seated):
            a.new_round(state, p)
        seat = next(p for p, a in enumerate(seated) if a is target)
        true_goal = state.assignment.goal_suit
        tracker = BeliefTracker(gen, n_particles=256)
        tracker.init(state, seat)
        buys: list[int] = []

        for _ in range(engine.config.n_ticks):
            actions = [a.choose_action(engine, state, p) for p, a in enumerate(seated)]
            engine.step(state, actions)
            for a in seated:
                a.observe(state)
            buys = [
                t.suit
                for t in state.trades
                if t.buyer != seat and isinstance(seated[t.buyer], HeuristicAgent)
            ]
            tracker.observe(state)
            post = tracker.assignment_posterior()
            base = _goal_marginal(post)
            orc = _oracle_marginal(post, buys, p_g)
            rec["base_arg"].append(float(np.argmax(base) == true_goal))
            rec["orc_arg"].append(float(np.argmax(orc) == true_goal))
            rec["base_p_true"].append(float(base[true_goal]))
            rec["orc_p_true"].append(float(orc[true_goal]))
            for s in range(N_SUITS):
                rec["base_pairs"].append((float(base[s]), 1.0 if s == true_goal else 0.0))
                rec["orc_pairs"].append((float(orc[s]), 1.0 if s == true_goal else 0.0))

        rec["buys"] += len(buys)
        rec["goal_buys"] += sum(1 for s in buys if s == true_goal)
    return rec


def _ece_and_curve(pairs):
    pairs = np.array(pairs)
    preds, actuals = pairs[:, 0], pairs[:, 1]
    bins = np.linspace(0.0, 1.0, 11)
    bin_idx = np.clip(np.digitize(preds, bins[1:-1]), 0, 9)
    ece = 0.0
    curve = []
    for b in range(10):
        mask = bin_idx == b
        if not mask.any():
            continue
        p = preds[mask].mean()
        f = actuals[mask].mean()
        ece += (mask.sum() / preds.size) * abs(f - p)
        curve.append((bins[b], bins[b + 1], p, f, mask.sum()))
    return ece, curve


def summarize(rec: dict, n_rounds: int, p_g: float) -> dict:
    base_ece, base_curve = _ece_and_curve(rec["base_pairs"])
    orc_ece, orc_curve = _ece_and_curve(rec["orc_pairs"])
    measured = rec["goal_buys"] / rec["buys"] if rec["buys"] else float("nan")
    return {
        "buy_goal_rate": measured,
        "buy_count": rec["buys"],
        "base_argmax": float(np.mean(rec["base_arg"])),
        "orc_argmax": float(np.mean(rec["orc_arg"])),
        "base_p_true": float(np.mean(rec["base_p_true"])),
        "orc_p_true": float(np.mean(rec["orc_p_true"])),
        "base_ece": float(base_ece),
        "orc_ece": float(orc_ece),
        "base_curve": base_curve,
        "orc_curve": orc_curve,
    }


def print_report(s: dict, n_rounds: int) -> None:
    print(f"rounds: {n_rounds}   heuristic buys: {s['buy_count']}")
    print(f"empirical buy-goal alignment rate : {s['buy_goal_rate']:.3f}")
    print(f"{'':24}{'strategy-free':>14}{'oracle':>14}")
    print(f"{'goal argmax accuracy':24}{s['base_argmax']:>14.3f}{s['orc_argmax']:>14.3f}")
    print(f"{'mean P(true goal)':24}{s['base_p_true']:>14.3f}{s['orc_p_true']:>14.3f}")
    print(f"{'goal-marginal ECE':24}{s['base_ece']:>14.3f}{s['orc_ece']:>14.3f}")
    print("baseline calibration curve (pred -> actual):")
    for lo, hi, p, f, n in s["base_curve"]:
        print(f"  [{lo:.1f},{hi:.1f}) {p:.3f} -> {f:.3f}  n={n}")
    print("oracle calibration curve (pred -> actual):")
    for lo, hi, p, f, n in s["orc_curve"]:
        print(f"  [{lo:.1f},{hi:.1f}) {p:.3f} -> {f:.3f}  n={n}")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Probe behavioral goal signal")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-goal", type=float, default=0.8, help="assumed P(buy s | goal = s)")
    ap.add_argument("--mode", choices=["heuristic", "random"], default="heuristic")
    args = ap.parse_args(argv)

    engine = Engine(RoundConfig(n_players=args.players, n_ticks=args.ticks), random.Random(args.seed))
    if args.mode == "heuristic":
        agents = [RandomAgent(args.seed + 1)] + [HeuristicAgent(args.seed + 2 + i) for i in range(args.players - 1)]
    else:
        agents = [RandomAgent(args.seed + i) for i in range(args.players)]

    rec = collect(engine, agents, args.rounds, args.seed + 100, args.p_goal)
    print_report(summarize(rec, args.rounds, args.p_goal), args.rounds)


if __name__ == "__main__":
    main()
