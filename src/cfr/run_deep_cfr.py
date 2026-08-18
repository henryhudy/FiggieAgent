"""Reproduce a fixed-seed Figgie-Lite tabular-CFR versus Deep-CFR comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from .cfr import CFR, DictStrategy
from .deep_cfr import DeepCFR, DeepCFRConfig


def run_comparison(
    seeds: list[int],
    iterations: int,
    config: DeepCFRConfig,
) -> dict:
    """Train both methods and evaluate every final policy exactly."""
    rows = []
    evaluator = CFR(ticks=config.ticks, seed=0)
    for seed in seeds:
        deep = DeepCFR(config, seed=seed)
        deep.iterate(iterations)
        deep_eps = evaluator.exploitability(deep.average_strategy())["eps"]

        tabular = CFR(ticks=config.ticks, seed=seed)
        tabular.iterate(iterations)
        tabular_eps = evaluator.exploitability(
            DictStrategy(tabular.average_strategy())
        )["eps"]
        rows.append(
            {
                "seed": seed,
                "deep_cfr_exploitability": deep_eps,
                "tabular_cfr_exploitability": tabular_eps,
            }
        )

    def summary(field: str) -> dict[str, float]:
        values = [row[field] for row in rows]
        return {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    return {
        "config": {
            "seeds": seeds,
            "iterations": iterations,
            "ticks": config.ticks,
            "hidden_size": config.hidden_size,
            "memory_capacity": config.memory_capacity,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "advantage_steps": config.advantage_steps,
            "strategy_steps": config.strategy_steps,
            "train_every": config.train_every,
        },
        "rows": rows,
        "deep_cfr": summary("deep_cfr_exploitability"),
        "tabular_cfr": summary("tabular_cfr_exploitability"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact Figgie-Lite Deep CFR versus tabular CFR benchmark"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 7, 11, 19, 23])
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--memory-capacity", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--advantage-steps", type=int, default=25)
    parser.add_argument("--strategy-steps", type=int, default=240)
    parser.add_argument("--train-every", type=int, default=25)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = DeepCFRConfig(
        ticks=args.ticks,
        hidden_size=args.hidden_size,
        memory_capacity=args.memory_capacity,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        advantage_steps=args.advantage_steps,
        strategy_steps=args.strategy_steps,
        train_every=args.train_every,
    )
    result = run_comparison(args.seeds, args.iterations, config)
    payload = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
