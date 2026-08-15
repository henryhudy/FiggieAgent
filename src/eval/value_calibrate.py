"""Value-calibration harness: net value prediction vs actual settle z.

Loads a trained checkpoint and plays eval rounds with the neural seat versus
random opponents, recording the network's value-head prediction at every tick
against the round's realized settle z = delta / VALUE_SCALE. Reports MSE, ECE,
correlation, and the best-fit affine map (actual ~= slope * pred + intercept) so
the value head can be recalibrated post-hoc.

The affine map is decision-invariant for argmax choices, so this is a
diagnostic/artifact harness, not part of search.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import torch

from ..learning.net import FiggieNet, load_partial
from ..learning.trainer import calibrate_value


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Calibrate a trained net's value head")
    ap.add_argument("--checkpoint", default="experiments/checkpoints/figgie_net_p2.pt")
    ap.add_argument("--history", default=None, choices=["mlp", "attn"],
                    help="history pathway; defaults to the checkpoint's recorded value")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--out", default=None, help="write the affine artifact JSON here")
    args = ap.parse_args(argv)

    ck = torch.load(args.checkpoint, map_location="cpu")
    history = args.history or ck.get("history", "mlp")
    net = FiggieNet(history=history)
    load_partial(net, args.checkpoint)
    net.eval()

    cal = calibrate_value(net, args.rounds, args.ticks, args.sims, args.seed)
    print(f"checkpoint: {args.checkpoint} (history={history})")
    print(f"value MSE: {cal['mse']:.4f}   ECE: {cal['ece']:.4f}   corr: {cal['corr']:.3f}")
    print(f"affine recalibration: actual = {cal['affine_slope']:.3f} * pred + {cal['affine_intercept']:+.3f}")
    print(f"{'bin':<12}{'predicted':>10}{'actual':>9}{'n':>8}")
    for lo, hi, p, f, n in cal["curve"]:
        print(f"[{lo:.2f},{hi:.2f})   {p:>10.3f}{f:>9.3f}{n:>8}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        artifact = {
            "checkpoint": args.checkpoint,
            "history": history,
            "mse": cal["mse"],
            "ece": cal["ece"],
            "corr": cal["corr"],
            "affine_slope": cal["affine_slope"],
            "affine_intercept": cal["affine_intercept"],
        }
        with open(args.out, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"wrote artifact to {args.out}")


if __name__ == "__main__":
    main()
