"""Multi-seed CFR benchmark runner.

Runs (a) the CFR agent against N-1 random bots and (b) the mixed full pool
(CFR + IS-MCTS + heuristic + random) over several seeds, saving per-round
records and printing bootstrap CIs, paired Wilcoxon tests, and Cohen's d.

Usage:
    PYTHONPATH=. .venv/bin/python tools/bench_cfr.py \
        --cfr experiments/cfr_ticks2.json \
        --random --seeds 7 8 9 --rounds 600 --out experiments/cfr_random_pool.json
    PYTHONPATH=. .venv/bin/python tools/bench_cfr.py \
        --cfr experiments/cfr_ticks2.json \
        --mixed --seeds 12345 12346 12347 --rounds 80 --sims 120 --depth 5 \
        --buy-only --top-k 2 --out experiments/cfr_mixed_pool_seeds.json
"""

from __future__ import annotations

import argparse
import json
import random

from src.agents.cfr_agent import CFRAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.is_mcts import ISMCTSAgent
from src.agents.random import RandomAgent
from src.env.engine import Engine
from src.env.state import RoundConfig

Record = tuple[int, str, float]


def run(engine: Engine, factories: list, n_rounds: int, base_seed: int) -> list[Record]:
    agents = [f() for f in factories]
    records: list[Record] = []
    for r in range(n_rounds):
        rot = r % len(agents)
        seated = agents[rot:] + agents[:rot]
        state = engine.new_round()
        for p, agent in enumerate(seated):
            agent.new_round(state, p)
        for _ in range(engine.config.n_ticks):
            actions = [a.choose_action(engine, state, p) for p, a in enumerate(seated)]
            engine.step(state, actions)
            for agent in seated:
                agent.observe(state)
        deltas = engine.settle(state)
        for agent, delta in zip(seated, deltas):
            records.append((r, agent.name, delta))
    return records


def build_pools(args) -> dict:
    cfr = [lambda: CFRAgent(args.cfr, seed=args.seed)]
    pools = {}
    if args.random:
        pools["random"] = cfr + [lambda i=i: RandomAgent(args.seed + 2 + i)
                                 for i in range(3)]
    if args.mixed:
        pools["mixed"] = cfr + [
            lambda: ISMCTSAgent(
                args.seed + 2, n_sims=args.sims, depth=args.depth,
                n_particles=args.particles, buy_only=args.buy_only,
                top_k=args.top_k, value_mode="hold",
            ),
            lambda: HeuristicAgent(args.seed + 3),
            lambda: RandomAgent(args.seed + 4),
        ]
    return pools


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfr", required=True, help="CFR policy JSON")
    ap.add_argument("--random", action="store_true", help="CFR vs 3 random bots")
    ap.add_argument("--mixed", action="store_true", help="CFR + IS-MCTS + heuristic + random")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--particles", type=int, default=512)
    ap.add_argument("--buy-only", action="store_true")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--out", default=None, help="JSON output path (else stdout)")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)

    from src.eval.stats import report_stats

    all_records: dict[str, list[Record]] = {}
    per_seed: dict[str, list] = {}
    for seed in args.seeds:
        args.seed = seed
        rng = random.Random(seed)
        engine = Engine(RoundConfig(n_players=4, n_ticks=args.ticks), rng)
        pools = build_pools(args)
        for pool_name, factories in pools.items():
            rec = run(engine, factories, args.rounds, seed)
            all_records.setdefault(pool_name, []).extend(
                [(r, name, d) for r, name, d in rec]
            )
            per_seed.setdefault(pool_name, []).append((seed, rec))
            print(f"== {pool_name} pool, seed {seed} ==")
            print(report_stats(rec, n_boot=args.boot, seed=seed))
            print()

    for pool_name, rec in all_records.items():
        pooled: list[Record] = []
        for seed_i, (seed, rec_s) in enumerate(per_seed[pool_name]):
            offset = seed_i * args.rounds
            pooled.extend((r + offset, name, d) for r, name, d in rec_s)
        print(f"== {pool_name} pool, ALL seeds pooled (n={len(pooled)} rounds) ==")
        print(report_stats(pooled, n_boot=args.boot, seed=args.seeds[0]))
        print()

    if args.out:
        pooled_out: dict[str, list[Record]] = {}
        for pool_name in per_seed:
            pooled_out[pool_name] = []
            for seed_i, (_seed, rec_s) in enumerate(per_seed[pool_name]):
                offset = seed_i * args.rounds
                pooled_out[pool_name].extend((r + offset, name, d) for r, name, d in rec_s)
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "per_seed": per_seed,
                       "pooled": pooled_out}, f, indent=1)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
