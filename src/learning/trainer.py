"""AlphaZero-style training loop for the M4 population-based neural agent.

Iteration:
  1. Generate self-play games: the neural seat searches with the current net
     against a pool mixture (greedy self, past selves, heuristic, random);
     record (obs, MCTS policy pi, terminal z).
  2. Train: policy cross-entropy (soft targets from pi) + value MSE on z.
  3. Evaluate: vs random opponents (continuity with M3), and a no-collapse
     check vs each past self (greedy) + heuristic + random. After the last
     iteration, a full-pool eval runs the current net against IS-MCTS +
     heuristic + random.

Value targets and predictions are z = settle_delta / VALUE_SCALE.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
from torch import nn, optim

from ..agents.heuristic import HeuristicAgent
from ..agents.is_mcts import ISMCTSAgent
from ..agents.neural import NeuralAgent
from ..agents.pool import SnapshotPool
from ..agents.random import RandomAgent
from ..env.engine import Engine
from ..env.state import RoundConfig
from ..eval.bench import run, summarize
from ..learning.net import FiggieNet, VALUE_SCALE, load_partial
from ..learning.obs import encode
from .replay import ReplayBuffer
from .selfplay import generate_games


def train_step(net: nn.Module, opt, obs, hist, pi, zz, assign, value_w: float = 1.0, belief_w: float = 0.0) -> tuple[float, float, float]:
    net.train()
    logits, v, b = net(obs, hist)
    logp = torch.log_softmax(logits, dim=1)
    loss_pi = -(pi * logp).sum(dim=1).mean()
    loss_v = nn.functional.mse_loss(v, zz)
    loss = loss_pi + value_w * loss_v
    if belief_w > 0:
        loss_b = nn.functional.cross_entropy(b, assign)
        loss = loss + belief_w * loss_b
    else:
        loss_b = torch.tensor(0.0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss_pi.item()), float(loss_v.item()), float(loss_b.item())


def evaluate(net, n_rounds: int, ticks: int, sims: int, depth: int, seed: int, learned_belief: bool = False, bayes_combine: bool = False) -> tuple[float, float]:
    engine = Engine(RoundConfig(n_ticks=ticks), random.Random(seed))
    factories = [lambda: NeuralAgent(net, seed=seed + 1, n_sims=sims, depth=depth, use_learned_belief=learned_belief)]
    factories += [lambda i=i: RandomAgent(seed + 2 + i) for i in range(3)]
    records = run(engine, factories, n_rounds)
    summary = summarize(records, n_rounds)
    rows = dict((name, mean) for name, mean, _, _ in summary["rows"])
    return rows["neural"], summary["wins"]["neural"] / n_rounds


def evaluate_pool(net, pool: SnapshotPool, n_rounds: int, ticks: int, sims: int, depth: int, seed: int, learned_belief: bool = False, bayes_combine: bool = False) -> dict:
    """No-collapse check: current net (search) vs each past self (greedy) +
    heuristic + random. Returns {snapshot: (neural mean delta, win share)}."""
    results: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(pool.names):
        engine = Engine(RoundConfig(n_ticks=ticks), random.Random(seed + 10 * i))
        factories = [
            lambda: NeuralAgent(net, seed=seed + 1, n_sims=sims, depth=depth, use_learned_belief=learned_belief, bayes_combine=bayes_combine),
            lambda name=name: pool.policy_agent(name, seed=seed + 2, temperature=0.0),
            lambda: HeuristicAgent(seed + 3),
            lambda: RandomAgent(seed + 4),
        ]
        records = run(engine, factories, n_rounds)
        summary = summarize(records, n_rounds)
        rows = dict((n, mean) for n, mean, _, _ in summary["rows"])
        results[name] = (rows["neural"], summary["wins"]["neural"] / n_rounds)
    return results


def evaluate_full_pool(net, n_rounds: int, ticks: int, sims: int, depth: int, seed: int, learned_belief: bool = False, bayes_combine: bool = False) -> dict:
    """M4 exit test: current net (search) vs the full bot pool: IS-MCTS
    (M2 config), heuristic, and one random liquidity seat."""
    engine = Engine(RoundConfig(n_ticks=ticks), random.Random(seed))
    factories = [
        lambda: NeuralAgent(net, seed=seed + 1, n_sims=sims, depth=depth, use_learned_belief=learned_belief, bayes_combine=bayes_combine),
        lambda: ISMCTSAgent(seed + 2, n_sims=sims, depth=depth, buy_only=True, top_k=2, value_mode="hold"),
        lambda: HeuristicAgent(seed + 3),
        lambda: RandomAgent(seed + 4),
    ]
    records = run(engine, factories, n_rounds)
    summary = summarize(records, n_rounds)
    rows = dict((n, (mean, summary["wins"][n] / n_rounds)) for n, mean, _, _ in summary["rows"])
    return rows


def calibrate_value(net, n_rounds: int, ticks: int, sims: int, seed: int) -> dict:
    """Bin the net's value prediction against actual settle z on eval games."""
    engine = Engine(RoundConfig(n_ticks=ticks), random.Random(seed + 77))
    neural = NeuralAgent(net, seed=seed + 78, n_sims=sims, depth=3)
    preds: list[float] = []
    actuals: list[float] = []
    for r in range(n_rounds):
        opps = [RandomAgent(random.Random(seed + 100 + r + i).randrange(1 << 30)) for i in range(3)]
        agents = [neural] + opps
        state = engine.new_round()
        for p, a in enumerate(agents):
            a.new_round(state, p)
        round_preds: list[float] = []
        for _ in range(ticks):
            action, pi, v = neural.search(engine, state, 0)
            round_preds.append(float(v))
            actions = [action] + [a.choose_action(engine, state, p) for p, a in enumerate(opps)]
            engine.step(state, actions)
            for a in agents:
                a.observe(state)
        z = engine.settle(state)[0] / VALUE_SCALE
        preds.extend(round_preds)
        actuals.extend([z] * len(round_preds))
    preds = np.array(preds)
    actuals = np.array(actuals)
    mse = float(np.mean((preds - actuals) ** 2))
    bins = np.linspace(-1.0, 1.0, 11)
    bin_idx = np.clip(np.digitize(preds, bins[1:-1]), 0, 9)
    ece = 0.0
    curve = []
    for b in range(10):
        mask = bin_idx == b
        if not mask.any():
            continue
        p, f = preds[mask].mean(), actuals[mask].mean()
        ece += (mask.sum() / preds.size) * abs(f - p)
        curve.append((bins[b], bins[b + 1], p, f, mask.sum()))
    slope, intercept = np.polyfit(preds, actuals, 1)
    return {
        "mse": mse,
        "ece": ece,
        "curve": curve,
        "corr": float(np.corrcoef(preds, actuals)[0, 1]),
        "affine_slope": float(slope),
        "affine_intercept": float(intercept),
    }


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Train the M4 population-based neural search agent")
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--train-steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4, help="Adam weight decay")
    ap.add_argument("--value-w", type=float, default=1.0, help="weight of the value MSE loss")
    ap.add_argument("--belief-w", type=float, default=0.0, help="weight of the learned-belief cross-entropy loss (P2)")
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--eval-rounds", type=int, default=40)
    ap.add_argument("--pool-eval-rounds", type=int, default=20)
    ap.add_argument("--full-eval-rounds", type=int, default=20)
    ap.add_argument("--pool-size", type=int, default=4, help="max past selves kept")
    ap.add_argument("--self-w", type=float, default=0.25, help="greedy self-policy opponent weight")
    ap.add_argument("--past-w", type=float, default=0.35, help="past-self opponent weight")
    ap.add_argument("--heuristic-w", type=float, default=0.25)
    ap.add_argument("--random-w", type=float, default=0.15)
    ap.add_argument("--full-eval", action="store_true", help="run the full-pool eval after the last iteration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="checkpoint to warm-start the net from (M3 or previous M4 run)")
    ap.add_argument("--out", default="experiments/checkpoints/figgie_net.pt")
    ap.add_argument("--history", default="mlp", choices=["mlp", "attn"], help="history pathway: P1 flat MLP or P2 self-attention")
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    net = FiggieNet(history=args.history)
    if args.init:
        load_partial(net, args.init)
    net.eval()
    buffer = ReplayBuffer()
    rng = np.random.default_rng(args.seed)
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    pool = SnapshotPool(args.seed, args.pool_size)
    mix = {"self": args.self_w, "past": args.past_w, "heuristic": args.heuristic_w, "random": args.random_w}

    for it in range(args.iterations):
        if len(pool) > 0:
            pool_eval = evaluate_pool(net, pool, args.pool_eval_rounds, args.ticks, args.sims, args.depth, args.seed + it * 3, args.belief_w > 0)
            for name, (md, ws) in pool_eval.items():
                print(f"    no-collapse vs {name}: mean {md:+.2f}/round, win share {ws:.2f}")

        print(f"--- iteration {it}: generating {args.games} self-play games")
        obs, hist, pi, z, assign = generate_games(
            net, args.games, n_ticks=args.ticks, n_sims=args.sims, depth=args.depth,
            seed=args.seed + it, pool=pool, mix=mix, learned_belief=args.belief_w > 0,
        )
        buffer.extend(obs, hist, pi, z, assign)
        print(f"    buffer size {len(buffer)}, mean z {np.mean([a[0] for a in z]):.3f}")
        for step in range(args.train_steps):
            o, h, p, zz, a = buffer.sample(args.batch, rng)
            lp, lv, lb = train_step(
                net, opt, torch.from_numpy(o), torch.from_numpy(h), torch.from_numpy(p), torch.from_numpy(zz),
                torch.from_numpy(a), value_w=args.value_w, belief_w=args.belief_w,
            )
            if step % 100 == 0:
                print(f"    step {step:4d}  loss_pi {lp:.4f}  loss_v {lv:.4f}  loss_b {lb:.4f}")
        net.eval()
        mdelta, wshare = evaluate(net, args.eval_rounds, args.ticks, args.sims, args.depth, args.seed + it * 7, args.belief_w > 0)
        print(f"    eval vs random: mean {mdelta:+.2f}/round, win share {wshare:.2f}")
        torch.save({"model": net.state_dict(), "iteration": it, "history": args.history}, args.out)
        pool.add(f"v{it}", net)
        print(f"    pool: {pool.names}")

    if args.full_eval:
        print("--- full-pool eval (neural vs ismcts + heuristic + random)")
        rows = evaluate_full_pool(net, args.full_eval_rounds, args.ticks, args.sims, args.depth, args.seed + 900, args.belief_w > 0)
        for name, (md, ws) in rows.items():
            print(f"    {name:<10} mean {md:+.2f}/round, win share {ws:.2f}")

    print("--- final value calibration")
    cal = calibrate_value(net, 20, args.ticks, args.sims, args.seed + 500)
    print(f"    MSE {cal['mse']:.4f}  ECE {cal['ece']:.4f}  corr {cal['corr']:.3f}")
    print(f"    affine actual = {cal['affine_slope']:.3f} * pred + {cal['affine_intercept']:+.3f}")
    for lo, hi, p, f, n in cal["curve"]:
        print(f"    [{lo:.2f},{hi:.2f}) pred {p:+.3f} -> actual {f:+.3f}  n={n}")
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
