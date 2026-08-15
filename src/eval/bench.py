"""Round-robin benchmark harness for Figgie agents."""

from __future__ import annotations

import argparse
import random
from typing import Callable, Optional

from ..agents.cfr_agent import CFRAgent
from ..agents.heuristic import HeuristicAgent
from ..agents.is_mcts import ISMCTSAgent
from ..agents.neural import NeuralAgent
from ..agents.random import RandomAgent
from ..env.engine import Engine
from ..env.state import RoundConfig
from ..learning.net import FiggieNet, load_partial

import torch

Factory = Callable[[], object]


def play_round(engine: Engine, agents: list) -> list[float]:
    state = engine.new_round()
    for p, agent in enumerate(agents):
        agent.new_round(state, p)
    for _ in range(engine.config.n_ticks):
        actions = [agent.choose_action(engine, state, p) for p, agent in enumerate(agents)]
        engine.step(state, actions)
        for agent in agents:
            agent.observe(state)
    return engine.settle(state)


def run(engine: Engine, factories: list[Factory], n_rounds: int) -> list[tuple[int, str, float]]:
    agents = [factory() for factory in factories]
    records: list[tuple[int, str, float]] = []
    for r in range(n_rounds):
        rot = r % len(agents)
        seated = agents[rot:] + agents[:rot]
        deltas = play_round(engine, seated)
        for agent, delta in zip(seated, deltas):
            records.append((r, agent.name, delta))
    return records


def summarize(records: list[tuple[int, str, float]], n_rounds: int) -> dict:
    names = sorted({name for _, name, _ in records})
    rows = []
    for name in names:
        ds = [d for _, n, d in records if n == name]
        rows.append((name, sum(ds) / len(ds), sum(ds), len(ds)))
    win_count = {name: 0.0 for name in names}
    for r in range(n_rounds):
        rds = [(name, d) for rn, name, d in records if rn == r]
        mx = max(d for _, d in rds)
        winners = [name for name, d in rds if d == mx]
        for w in winners:
            win_count[w] += 1.0 / len(winners)
    return {"rows": rows, "wins": win_count}


def print_stats(summary: dict, n_rounds: int) -> None:
    rows = sorted(summary["rows"], key=lambda r: -r[1])
    print(f"{'agent':<10} {'mean delta':>10} {'total':>10} {'rounds':>7} {'win share':>10}")
    for name, mean, total, n in rows:
        share = summary["wins"][name] / n_rounds
        print(f"{name:<10} {mean:>10.2f} {total:>10.2f} {n:>7} {share:>10.2f}")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Benchmark Figgie agents")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--mcts", action="store_true", help="use one IS-MCTS agent")
    ap.add_argument("--cfr", default=None, help="policy JSON path for one CFR agent")
    ap.add_argument("--neural", default=None, help="checkpoint path for one neural search agent")
    ap.add_argument("--sims", type=int, default=100, help="MCTS sims per tick")
    ap.add_argument("--depth", type=int, default=6, help="MCTS in-tree depth")
    ap.add_argument("--particles", type=int, default=512)
    ap.add_argument("--buy-only", action="store_true", help="MCTS: no sells/withdraws")
    ap.add_argument("--top-k", type=int, default=None, help="MCTS: buy only on top-k belief suits")
    ap.add_argument("--learned-belief", action="store_true", help="neural: use the net's learned belief head (P2)")
    ap.add_argument("--bayes-combine", action="store_true", help="neural: Bayes-combine exact + learned posterior into the obs")
    ap.add_argument("--value-mode", choices=["hold", "world", "rollout"], default="hold")
    ap.add_argument("--stats", action="store_true", help="report bootstrap CIs and paired Wilcoxon tests")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument(
        "--opponents",
        choices=["heuristic", "random"],
        default="heuristic" if "--mcts" in (argv or []) or "--neural" in (argv or []) else "random",
    )
    args = ap.parse_args(argv)

    engine = Engine(RoundConfig(n_players=args.players, n_ticks=args.ticks), random.Random(args.seed))
    if args.neural:
        ckpt = torch.load(args.neural, map_location="cpu")
        net = FiggieNet(history=ckpt.get("history", "mlp"))
        load_partial(net, args.neural)
        net.eval()
        factories: list[Factory] = [
            lambda: NeuralAgent(
                net, seed=args.seed + 1, n_sims=args.sims, depth=args.depth,
                use_learned_belief=args.learned_belief, bayes_combine=args.bayes_combine,
            )
        ]
        opp_cls = HeuristicAgent if args.opponents == "heuristic" else RandomAgent
        factories += [lambda i=i: opp_cls(args.seed + 2 + i) for i in range(args.players - 1)]
    elif args.mcts:
        factories: list[Factory] = [
            lambda: ISMCTSAgent(
                args.seed + 1, n_sims=args.sims, depth=args.depth, n_particles=args.particles,
                buy_only=args.buy_only, top_k=args.top_k, value_mode=args.value_mode,
            )
        ]
        opp_cls = HeuristicAgent if args.opponents == "heuristic" else RandomAgent
        factories += [lambda i=i: opp_cls(args.seed + 2 + i) for i in range(args.players - 1)]
    elif args.cfr:
        factories: list[Factory] = [
            lambda: CFRAgent(args.cfr, seed=args.seed + 1)
        ]
        opp_cls = HeuristicAgent if args.opponents == "heuristic" else RandomAgent
        factories += [lambda i=i: opp_cls(args.seed + 2 + i) for i in range(args.players - 1)]
    else:
        factories = [lambda: HeuristicAgent(args.seed + 1)]
        factories += [lambda i=i: RandomAgent(args.seed + 2 + i) for i in range(args.players - 1)]

    records = run(engine, factories, args.rounds)
    print_stats(summarize(records, args.rounds), args.rounds)
    if args.stats:
        from .stats import report_stats
        print()
        print(report_stats(records, n_boot=args.boot, seed=args.seed))


if __name__ == "__main__":
    main()
