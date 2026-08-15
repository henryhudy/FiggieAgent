"""Playable single-round Figgie CLI (P3): you vs a trained neural net + bots.

Runs one 4-player round at 120 ticks. Seat 0 is a human; seats 1..3 default to
the trained neural agent (search), the heuristic bot, and a random bot, in that
order. At settlement the round deltas and winner are printed.

Example:
    python -m src.play_cli --checkpoint experiments/checkpoints/figgie_net_p2.pt
"""

from __future__ import annotations

import argparse
import random
from typing import Optional

import torch

from .agents.heuristic import HeuristicAgent
from .agents.human import HumanAgent
from .agents.neural import NeuralAgent
from .agents.random import RandomAgent
from .env.engine import Engine
from .env.state import RoundConfig
from .learning.net import FiggieNet, load_partial


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Play a Figgie round against the trained agent")
    ap.add_argument("--checkpoint", default="experiments/checkpoints/figgie_net_p2.pt")
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--learned-belief", action="store_true", help="use the P2 learned belief head")
    ap.add_argument("--bayes-combine", action="store_true", help="Bayes-combine exact + learned posterior into the obs")
    ap.add_argument("--seat", type=int, default=0, help="which seat the human plays")
    ap.add_argument("--no-belief", action="store_true", help="hide the belief goal-suit display")
    args = ap.parse_args(argv)

    engine = Engine(RoundConfig(n_ticks=args.ticks), random.Random(args.seed))
    if args.seat != 0:
        print("note: only seat 0 is supported; using seat 0")
    human = HumanAgent(args.seed, show_belief=not args.no_belief)
    agents = [human]

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    net = FiggieNet(history=ckpt.get("history", "mlp"))
    load_partial(net, args.checkpoint)
    net.eval()
    agents += [
        NeuralAgent(net, seed=args.seed + 1, n_sims=args.sims, depth=args.depth, use_learned_belief=args.learned_belief, bayes_combine=args.bayes_combine),
        HeuristicAgent(args.seed + 2),
        RandomAgent(args.seed + 3),
    ]

    state = engine.new_round()
    for p, a in enumerate(agents):
        a.new_round(state, p)
    for _ in range(engine.config.n_ticks):
        actions = [a.choose_action(engine, state, p) for p, a in enumerate(agents)]
        engine.step(state, actions)
        for a in agents:
            a.observe(state)

    deltas = engine.settle(state)
    print("\n" + "=" * 64)
    print("ROUND OVER")
    for p, (a, d) in enumerate(zip(agents, deltas)):
        print(f"  P{p} {a.name:<10} delta ${d:+.2f}")
    mx = max(deltas)
    winners = [p for p, d in enumerate(deltas) if d == mx]
    print("winner(s): " + ", ".join(f"P{p} ({agents[p].name})" for p in winners))


if __name__ == "__main__":
    main()
