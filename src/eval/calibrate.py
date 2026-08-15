"""Belief-tracker calibration harness: posterior vs simulator ground truth.

Runs rounds with a live BeliefTracker attached to a designated player and
records, every tick, the tracker's posterior probability on the true
assignment. Reports how the posterior concentrates, its argmax accuracy, and a
calibration curve (predicted P(true) vs realized frequency) with ECE, compared
against the hand-only prior baseline.
"""

from __future__ import annotations

import argparse
import random
from typing import Optional

import numpy as np

from ..agents.heuristic import HeuristicAgent
from ..agents.random import RandomAgent
from ..belief.tracker import BeliefTracker
from ..env.deck import ALL_ASSIGNMENTS
from ..env.engine import Engine
from ..env.state import RoundConfig


def collect(engine, agents, n_rounds: int, n_particles: int, seed: int, posterior_every: int = 4) -> dict:
    gen = np.random.default_rng(seed)
    target = agents[0]
    rec = {
        "init_p_true": [],
        "end_p_true": [],
        "argmax_correct": [],
        "goal_argmax_correct": [],
        "prior_argmax_correct": [],
        "pairs": [],  # (pred, actual) for calibration curve, every tick
        "goal_pairs": [],
    }
    for r in range(n_rounds):
        rot = r % len(agents)
        seated = agents[rot:] + agents[:rot]
        state = engine.new_round()
        for p, a in enumerate(seated):
            a.new_round(state, p)
        seat = next(p for p, a in enumerate(seated) if a is target)
        true_ai = ALL_ASSIGNMENTS.index(state.assignment)

        tracker = BeliefTracker(gen, n_particles=n_particles)
        tracker.init(state, seat)
        rec["init_p_true"].append(float(tracker.assignment_posterior()[true_ai]))
        init_post = tracker.assignment_posterior()
        rec["prior_argmax_correct"].append(float(np.argmax(init_post) == true_ai))

        for tick in range(engine.config.n_ticks):
            actions = [a.choose_action(engine, state, p) for p, a in enumerate(seated)]
            engine.step(state, actions)
            for a in seated:
                a.observe(state)
            tracker.observe(state)
            if tick % posterior_every:
                continue
            post = tracker.assignment_posterior()
            for ai in range(len(ALL_ASSIGNMENTS)):
                rec["pairs"].append((float(post[ai]), 1.0 if ai == true_ai else 0.0))
            gm = tracker.goal_marginals()
            for s in range(4):
                rec["goal_pairs"].append((float(gm[s]), 1.0 if s == state.assignment.goal_suit else 0.0))

        post = tracker.assignment_posterior()
        rec["end_p_true"].append(float(post[true_ai]))
        rec["argmax_correct"].append(float(np.argmax(post) == true_ai))
        gm = tracker.goal_marginals()
        rec["goal_argmax_correct"].append(float(np.argmax(gm) == state.assignment.goal_suit))
    return rec


def calibrate(rec: dict, n_rounds: int) -> dict:
    def ece_and_curve(pairs):
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
            w = mask.sum() / preds.size
            ece += w * abs(f - p)
            curve.append((bins[b], bins[b + 1], p, f, mask.sum()))
        return ece, curve

    ece, curve = ece_and_curve(rec["pairs"])
    goal_ece, goal_curve = ece_and_curve(rec["goal_pairs"])
    return {
        "mean_init_p_true": float(np.mean(rec["init_p_true"])),
        "mean_end_p_true": float(np.mean(rec["end_p_true"])),
        "argmax_accuracy": float(np.mean(rec["argmax_correct"])),
        "prior_argmax_accuracy": float(np.mean(rec["prior_argmax_correct"])),
        "goal_argmax_accuracy": float(np.mean(rec["goal_argmax_correct"])),
        "ece": float(ece),
        "goal_ece": float(goal_ece),
        "curve": curve,
        "goal_curve": goal_curve,
        "samples": int(np.array(rec["pairs"]).shape[0]),
        "goal_samples": int(np.array(rec["goal_pairs"]).shape[0]),
    }


def print_report(stats: dict, n_rounds: int) -> None:
    print(f"rounds: {n_rounds}   calibration samples: {stats['samples']}")
    print(f"mean P(true assignment) at deal   : {stats['mean_init_p_true']:.3f}  (hand prior baseline)")
    print(f"mean P(true assignment) at reveal : {stats['mean_end_p_true']:.3f}")
    print(f"assignment argmax accuracy        : {stats['argmax_accuracy']:.3f}  (prior: {stats['prior_argmax_accuracy']:.3f})")
    print(f"goal-suit argmax accuracy         : {stats['goal_argmax_accuracy']:.3f}  (chance: 0.250)")
    print(f"expected calibration error (ECE)  : {stats['ece']:.3f}")
    print(f"goal-marginal ECE                 : {stats['goal_ece']:.3f}")
    print(f"{'bin':<12}{'predicted':>10}{'actual':>9}{'n':>8}")
    for lo, hi, p, f, n in stats["curve"]:
        print(f"[{lo:.1f},{hi:.1f})   {p:>10.3f}{f:>9.3f}{n:>8}")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Calibrate the belief tracker")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--particles", type=int, default=512)
    ap.add_argument("--posterior-every", type=int, default=4)
    ap.add_argument("--mode", choices=["heuristic", "random"], default="heuristic")
    args = ap.parse_args(argv)

    engine = Engine(RoundConfig(n_players=args.players, n_ticks=args.ticks), random.Random(args.seed))
    if args.mode == "heuristic":
        agents = [HeuristicAgent(args.seed + 1)] + [RandomAgent(args.seed + 2 + i) for i in range(args.players - 1)]
    else:
        agents = [RandomAgent(args.seed + i) for i in range(args.players)]

    rec = collect(engine, agents, args.rounds, args.particles, args.seed + 100, args.posterior_every)
    print_report(calibrate(rec, args.rounds), args.rounds)


if __name__ == "__main__":
    main()
