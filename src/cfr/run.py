"""Train CFR on Figgie-Lite and report exploitability and head-to-heads.

Usage:
    python -m src.cfr.run --iterations 20000 --ticks 3 --seed 0
    python -m src.cfr.run --save policy.json --iterations 40000

Outputs a per-iteration exploitability curve, the exploitability of several
reference strategies, and head-to-head match results between the CFR average
strategy and the references.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

from .cfr import (
    CFR,
    DictStrategy,
    FuncStrategy,
    play_match,
    reference_belief_buy,
    reference_belief_with_signals,
    reference_pass,
    reference_uniform_actions,
)


def build_policy(cfr: CFR, ticks: int) -> dict:
    """Aggregate the learned average strategy for the full-game CFRAgent:
    per (goal, tick, hand) a 5-vector, marginalized over the public
    histories actually reached during training (the table is sparse).

    A hand is included under goal g only if it is a feasible subset of world
    g's deck; a player who believes g plays the average of the CFR buy
    policy over the histories reached with a goal-feasible hand.
    """
    from .figgie_lite import REACHABLE_HANDS, deck_counts
    avg = cfr.average_strategy()
    n_hands = len(REACHABLE_HANDS)
    # counts[goal][tick][hand][action]
    acc = [[[[0.0] * 5 for _ in range(n_hands)] for _ in range(ticks)] for _ in range(4)]
    cnt = [[[0 for _ in range(n_hands)] for _ in range(ticks)] for _ in range(4)]
    for key, probs in avg.items():
        player, hi, hist = key
        tick = len(hist)
        if tick >= ticks:
            continue
        hand = REACHABLE_HANDS[hi]
        for g in range(4):
            dc = deck_counts(g)
            if all(hand[s] <= dc[s] for s in range(4)):
                for a in range(5):
                    acc[g][tick][hi][a] += probs[a]
                cnt[g][tick][hi] += 1
    pol: dict[str, list[list[list[float]]]] = {}
    for g in range(4):
        rows: list[list[list[float]]] = []
        for tick in range(ticks):
            hand_rows: list[list[float]] = []
            for hi in range(n_hands):
                if cnt[g][tick][hi] > 0:
                    hand_rows.append([x / cnt[g][tick][hi] for x in acc[g][tick][hi]])
                else:
                    hand_rows.append([1.0 / 5.0] * 5)
            rows.append(hand_rows)
        pol[str(g)] = rows
    return {"ticks": ticks, "hands": n_hands, "policy": pol}


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="CFR on Figgie-Lite")
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--ticks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--matches", type=int, default=4000)
    ap.add_argument("--save", default=None, help="save aggregated policy JSON")
    args = ap.parse_args(argv)

    cfr = CFR(ticks=args.ticks, seed=args.seed)

    print(f"CFR on Figgie-Lite: ticks={args.ticks}, iterations={args.iterations}, "
          f"seed={args.seed}")
    print(f"{'iter':>8} {'value':>10} {'br0':>10} {'br1':>10} {'exploitability':>15} "
          f"{'time/s':>8}")
    t0 = time.time()
    curve = []

    def report(it: int, exp: dict) -> None:
        print(f"{it:>8} {exp['value']:>10.4f} {exp['br0']:>10.4f} {exp['br1']:>10.4f} "
              f"{exp['eps']:>15.6f} {time.time()-t0:>8.1f}", flush=True)
        curve.append([it, exp["value"], exp["eps"]])

    cfr.iterate(args.iterations, progress_every=args.eval_every, report=report)

    avg = DictStrategy(cfr.average_strategy())
    final = cfr.exploitability(avg)
    print(f"\nFinal: value={final['value']:.4f} eps0={final['eps0']:.5f} "
          f"eps1={final['eps1']:.5f} exploitability={final['eps']:.6f}")

    refs = {
        "pass": FuncStrategy(reference_pass),
        "uniform": FuncStrategy(reference_uniform_actions),
        "belief-buy": FuncStrategy(reference_belief_buy),
        "belief-signals": FuncStrategy(reference_belief_with_signals),
    }
    print(f"\nReference-strategy exploitability (vs exact best response):")
    print(f"{'strategy':>15} {'value':>10} {'br0':>10} {'br1':>10} {'exploitability':>15}")
    ref_stats = {}
    for name, strat in refs.items():
        exp = cfr.exploitability(strat)
        ref_stats[name] = exp
        print(f"{name:>15} {exp['value']:>10.4f} {exp['br0']:>10.4f} "
              f"{exp['br1']:>10.4f} {exp['eps']:>15.6f}")

    print(f"\nHead-to-head (mean P0 payoff over {args.matches} games):")
    for name, strat in refs.items():
        a = play_match(avg, strat, args.ticks, args.matches, seed=args.seed)
        b = play_match(strat, avg, args.ticks, args.matches, seed=args.seed + 1)
        print(f"  CFR-Nash vs {name:<15}: {a:+8.3f}   {name} vs CFR-Nash: {b:+8.3f}")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(build_policy(cfr, args.ticks), f)
        print(f"\nSaved aggregated policy -> {args.save}")
        curve_path = os.path.splitext(args.save)[0] + "_curve.json"
        with open(curve_path, "w") as f:
            json.dump(curve, f)
        print(f"Saved convergence curve -> {curve_path}")


if __name__ == "__main__":
    main()
