"""Generate the data files and preview renders for the Figgie paper figures.

The values mirror the fixed simulator results reported in the manuscript and
DESIGN.md. They make the paper figures reproducible from tracked source files;
the original experiment logs are not distributed with this repository.

Usage:
    python make_figures.py            # write data/*.dat + preview/*.png
    python make_figures.py --dat-only # only write the pgfplots .dat files
"""

from __future__ import annotations

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PREVIEW = os.path.join(HERE, "preview")

# ---------------------------------------------------------------------------
# 1. Random-pool benchmark: mean round delta / win share vs. 3 random bots.
#    Config: 4 players, 120 ticks, 120 search sims, depth 5, seed 7.
#    Source: DESIGN.md M2/M3/M4 tables. Original run logs are not included.
# ---------------------------------------------------------------------------
RANDOM_POOL = [
    # (agent, mean_delta_usd, win_share)
    ("CFR-policy", 46.6, 0.46),   # search-free external baseline (sec:cfr)
    ("Heuristic", 76.0, 0.65),
    ("IS-MCTS (M2)", 109.8, 0.85),
    ("M3 (neural)", 123.3, 0.95),
    ("M4 (pop.)", 128.8, 0.97),
    ("P1 (+hist.)", 125.8, 0.94),
    ("P2 (+belief)", 127.75, 0.95),
]

# ---------------------------------------------------------------------------
# 2. Mixed full-pool benchmark: mean round delta per seat.
#    Lineup: neural + IS-MCTS + heuristic + random.
#    Config: 40 rounds, 100 sims, depth 4, seed 12345.
#    Source: DESIGN.md M4/P2/P2b/P1b notes. Original run logs are not included.
#    Columns: run, neural, ismcts, heuristic, random.
# ---------------------------------------------------------------------------
MIXED_POOL = [
    ("M4", +49.30, +9.5, -8.80, -50.00),
    ("P2", +43.25, +7.0, -1.50, -48.75),
    ("P2+combine", +49.00, +0.58, -2.42, -47.17),
    ("P1b", +28.50, +16.0, -1.00, -43.50),
]

# ---------------------------------------------------------------------------
# 3. Value-head reliability bins (P1v, dedicated harness).
#    Source: recorded P1v calibration summary in DESIGN.md.
#    Columns: bin_lo, bin_hi, mean_predicted, mean_actual, n.
# ---------------------------------------------------------------------------
CALIB = [
    (-1.00, -0.80, -1.095, 2.700, 4),
    (-0.80, -0.60, -0.660, 2.067, 6),
    (-0.40, -0.20, -0.287, 1.667, 3),
    (-0.20, 0.00, -0.049, 0.080, 5),
    (0.00, 0.20, 0.147, -0.158, 24),
    (0.20, 0.40, 0.317, 0.818, 11),
    (0.40, 0.60, 0.509, 0.442, 24),
    (0.60, 0.80, 0.681, 1.587, 47),
    (0.80, 1.00, 2.268, 2.310, 2276),
]
AFFINE_SLOPE, AFFINE_INTERCEPT = 0.22698374031257595, 1.7479006285818928

# ---------------------------------------------------------------------------
# 4. Learned-belief head cross-entropy vs. training step (P2 run).
#    Source: recorded P2 training summary in DESIGN.md.
# ---------------------------------------------------------------------------
BELIEF_CE = [  # (global_step, cross_entropy)
    (0, 2.4849), (100, 1.0119), (200, 0.3254), (300, 0.1303),
    (400, 5.0339), (500, 0.3888), (600, 0.1641), (700, 0.0732),
    (800, 2.3647), (900, 0.2458), (1000, 0.1375), (1100, 0.1200),
    (1200, 2.3138), (1300, 0.2618), (1400, 0.1777), (1500, 0.1843),
    (1600, 2.1236), (1700, 0.3607), (1800, 0.2818), (1900, 0.2343),
]

# ---------------------------------------------------------------------------
# 5. Bundle-quote usage probe (P1b). Source: DESIGN.md P1b note.
# ---------------------------------------------------------------------------
BUNDLE = [
    ("bundle bids posted\n(452 / 720 decisions)", 452),
    ("bundle trades filled\n(2 / 138 trades)", 2),
]


def write_dat(name: str, header: str, rows: list[list]) -> str:
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    with open(path, "w") as f:
        f.write(header.strip() + "\n")
        for row in rows:
            f.write(" ".join(str(x) for x in row) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dat-only", action="store_true")
    args = ap.parse_args()

    files = {
        "random_pool.dat": write_dat(
            "random_pool.dat",
            "agent mean_delta win_share",
            [list(r) for r in RANDOM_POOL],
        ),
        "mixed_pool.dat": write_dat(
            "mixed_pool.dat",
            "run neural ismcts heuristic random",
            [list(r) for r in MIXED_POOL],
        ),
        "calib.dat": write_dat(
            "calib.dat",
            "bin_lo bin_hi pred actual n",
            [list(r) for r in CALIB],
        ),
        "belief_ce.dat": write_dat(
            "belief_ce.dat",
            "step ce",
            [list(r) for r in BELIEF_CE],
        ),
        "bundle.dat": write_dat(
            "bundle.dat",
            "label value",
            [list(r) for r in BUNDLE],
        ),
    }
    for name, path in files.items():
        print(f"wrote {path}")

    if args.dat_only:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "figure.dpi": 150,
        "savefig.dpi": 200,
    })
    C = {
        "navy": "#1f3b73",
        "teal": "#2a9d8f",
        "orange": "#e76f51",
        "gray": "#8d99ae",
        "gold": "#e9c46a",
    }

    # --- preview 1: random-pool progression ---------------------------------
    labels = [r[0] for r in RANDOM_POOL]
    delta = [r[1] for r in RANDOM_POOL]
    share = [r[2] for r in RANDOM_POOL]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    cols = [C["gold"], C["navy"], *([C["teal"]] * (len(delta) - 2))]
    bars = ax.bar(labels, delta, color=cols, width=0.62)
    for b, d, s in zip(bars, delta, share):
        ax.text(b.get_x() + b.get_width() / 2, d + 2.5, f"${d:+.0f}\n({s:.2f})",
                ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhspan(122, 132, color=C["gold"], alpha=0.15)
    ax.set_ylim(0, 150)
    ax.set_ylabel("mean round $\\Delta$ vs. 3 random bots ($\\$$)")
    ax.set_title("Random-pool benchmark saturates near $+\\sim126$ after M3",
                 fontsize=10)
    ax.tick_params(axis="x", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PREVIEW, "fig_random_pool.png"))
    plt.close(fig)

    # --- preview 2: mixed full-pool -----------------------------------------
    runs = [r[0] for r in MIXED_POOL]
    seats = ["neural", "ismcts", "heuristic", "random"]
    seat_color = {"neural": C["navy"], "ismcts": C["teal"],
                  "heuristic": C["orange"], "random": C["gray"]}
    x = np.arange(len(runs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, seat in enumerate(seats):
        vals = [r[1 + i] for r in MIXED_POOL]
        ax.bar(x + (i - 1.5) * w, vals, w, label=seat, color=seat_color[seat])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.set_ylabel("mean round $\\Delta$ ($\\$$)")
    ax.set_title("Mixed full-pool: behavior conditioning only helps vs. "
                 "behavior-driven opponents", fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left",
              bbox_to_anchor=(0.0, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(PREVIEW, "fig_mixed_pool.png"))
    plt.close(fig)

    # --- preview 3: reliability diagram -------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    lo = np.array([c[0] for c in CALIB])
    hi = np.array([c[1] for c in CALIB])
    pred = np.array([c[2] for c in CALIB])
    act = np.array([c[3] for c in CALIB])
    n = np.array([c[4] for c in CALIB])
    size = 20 + 60 * n / n.max()
    ax.scatter(pred, act, s=size, color=C["navy"], alpha=0.85, zorder=3,
               edgecolor="white", linewidth=0.5)
    xgrid = np.linspace(-1.2, 2.5, 100)
    ax.plot(xgrid, xgrid, "--", color=C["gray"], lw=1.2,
            label="perfect calibration")
    ax.plot(xgrid, AFFINE_SLOPE * xgrid + AFFINE_INTERCEPT, color=C["orange"],
            lw=1.4, label="affine fit $y = 0.227x + 1.75$")
    ax.annotate("$n=2276$ (92%)", xy=(2.268, 2.310), xytext=(1.15, 2.05),
                fontsize=8, arrowprops=dict(arrowstyle="-", color=C["gray"]))
    ax.set_xlabel("mean predicted value $\\hat{v}$")
    ax.set_ylabel("mean actual value $v$")
    ax.set_title("Value-head reliability (P1v): MSE $=0.96$, ECE $=0.08$",
                 fontsize=10)
    ax.set_xlim(-1.3, 2.6)
    ax.set_ylim(-0.6, 3.0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PREVIEW, "fig_calibration.png"))
    plt.close(fig)

    # --- preview 4: belief CE -------------------------------------------------
    steps = [r[0] for r in BELIEF_CE]
    ce = [r[1] for r in BELIEF_CE]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.plot(steps, ce, color=C["teal"], lw=1.5, marker="o", ms=3)
    ax.axhline(0.1, color=C["gray"], ls=":", lw=1)
    ax.annotate("plateau $\\approx 0.1$", xy=(900, 0.1), xytext=(900, 0.32),
                fontsize=8, color=C["gray"])
    for t in range(1, 5):
        ax.axvline(t * 300, color=C["gray"], lw=0.5, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("global training step")
    ax.set_ylabel("belief CE $\\mathcal{L}_b$")
    ax.set_title("Learned-belief head collapses from CE $\\approx 2.5$ to "
                 "$\\approx 0.1$ (P2)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PREVIEW, "fig_belief_ce.png"))
    plt.close(fig)

    # --- preview 5: bundle over-posting ----------------------------------------
    # Plot rates, rather than raw counts, because the two quantities have
    # different denominators (decisions vs. completed trades).
    labels_b = ["bundle bids posted\n(452 of 720 decisions)", "bundle trades\n"
                "filled (2 of 138)"]
    vals_b = [452 / 720 * 100, 2 / 138 * 100]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    bars = ax.bar(labels_b, vals_b, color=[C["navy"], C["orange"]], width=0.5)
    for b, v in zip(bars, vals_b):
        denominator = 720 if v > 10 else 138
        numerator = 452 if v > 10 else 2
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%\n({numerator}/{denominator})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("rate (%)")
    ax.set_title("Bundle quotes are posted but almost never filled (P1b)",
                 fontsize=10)
    ax.set_ylim(0, 78)
    fig.tight_layout()
    fig.savefig(os.path.join(PREVIEW, "fig_bundle.png"))
    plt.close(fig)

    print(f"wrote previews to {PREVIEW}/")


if __name__ == "__main__":
    main()
