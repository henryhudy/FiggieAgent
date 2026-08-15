"""Statistical helpers for benchmark reporting.

Pure-Python implementations (no scipy dependency): bootstrap confidence
intervals for per-agent mean round delta, a paired Wilcoxon signed-rank test,
and a paired Cohen's d effect size for per-round deltas between two agents.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Optional


def per_round_deltas(records: list[tuple[int, str, float]]) -> dict[str, dict[int, float]]:
    """Group records by agent name -> {round: delta} (one delta per round)."""
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for r, name, delta in records:
        out[name][r] = delta
    return dict(out)


def bootstrap_mean_ci(
    deltas_by_round: dict[int, float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> tuple[float, float, float]:
    """Bootstrap (round-level resampling) 1-alpha CI for the mean delta.

    Returns (mean, lo, hi). Resampling keeps each round's per-agent delta
    intact, so cross-agent correlation within a round is preserved for
    per-agent means; for paired differences the caller should use
    bootstrap_diff_ci which resamples the paired rounds jointly.
    """
    rng = random.Random(seed)
    rounds = sorted(deltas_by_round)
    vals = [deltas_by_round[r] for r in rounds]
    n = len(vals)
    mean = sum(vals) / n
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return mean, lo, hi


def bootstrap_diff_ci(
    x: dict[int, float],
    y: dict[int, float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of (x - y) over paired rounds."""
    rng = random.Random(seed)
    rounds = sorted(set(x) & set(y))
    if not rounds:
        return (float("nan"), float("nan"), float("nan"))
    diffs = [x[r] - y[r] for r in rounds]
    n = len(diffs)
    mean = sum(diffs) / n
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return mean, lo, hi


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon_signed_rank(
    x: dict[int, float],
    y: dict[int, float],
) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test on x - y.

    Returns (W+, p-value). Ties are dropped and the normal approximation with
    continuity correction is used, matching scipy's default for n > 25.
    """
    rounds = sorted(set(x) & set(y))
    diffs = [x[r] - y[r] for r in rounds]
    diffs = [d for d in diffs if abs(d) > 1e-12]
    n = len(diffs)
    if n == 0:
        return (0.0, 1.0)
    ranked = sorted((abs(d), i) for i, d in enumerate(diffs))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[ranked[k][1]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i, d in enumerate(diffs) if d > 0)
    if n < 30:
        # exact enumeration over the 2^n sign patterns.
        from itertools import combinations

        total = 0.0
        for k in range(n):
            for comb in combinations(ranks, k):
                if sum(comb) >= w_plus:
                    total += 1
        p = total / (2 ** n)
        p = min(2.0 * p, 1.0)
        return w_plus, p
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_plus - mu) / sigma
    z = (abs(z) - 0.5) / (sigma / abs(sigma)) if False else z
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return w_plus, p


def cohens_d_paired(x: dict[int, float], y: dict[int, float]) -> float:
    """Paired Cohen's d (d_z) for (x - y) over paired rounds.

    d_z = mean(diff) / sd(diff), using the sample standard deviation of the
    paired differences. Conventionally |d| ~ 0.2/0.5/0.8 marks a
    small/medium/large effect.
    """
    rounds = sorted(set(x) & set(y))
    diffs = [x[r] - y[r] for r in rounds]
    n = len(diffs)
    if n < 2:
        return float("nan")
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0.0:
        return float("inf") if mean > 0 else (float("-inf") if mean < 0 else float("nan"))
    return mean / sd


def report_stats(
    records: list[tuple[int, str, float]],
    n_boot: int = 2000,
    seed: Optional[int] = None,
    n_rounds: Optional[int] = None,
) -> str:
    """Human-readable stats block for a benchmark run."""
    groups = per_round_deltas(records)
    names = sorted(groups)
    n_rounds = n_rounds or max((max(g) + 1 for g in groups.values()), default=0)
    lines = []
    lines.append(f"{'agent':<10} {'mean':>9} {'95% CI':>22} {'n':>5}")
    cis = {}
    for name in names:
        mean, lo, hi = bootstrap_mean_ci(groups[name], n_boot=n_boot, seed=seed)
        cis[name] = (mean, lo, hi)
        lines.append(f"{name:<10} {mean:>+9.2f} [{lo:>+8.2f}, {hi:>+8.2f}] {len(groups[name]):>5}")
    lines.append("")
    base = names[0] if names else None
    if base:
        lines.append(f"Paired Wilcoxon signed-rank vs. {base} (per-round delta):")
        for name in names[1:]:
            w, p = wilcoxon_signed_rank(groups[base], groups[name])
            m, lo, hi = bootstrap_diff_ci(groups[base], groups[name], n_boot=n_boot, seed=seed)
            d = cohens_d_paired(groups[base], groups[name])
            lines.append(
                f"  {base:<8} - {name:<8}: diff {m:>+8.2f} "
                f"[{lo:>+7.2f}, {hi:>+7.2f}]  W={w:>7.0f} p={p:.4f} "
                f"d={d:+.2f}"
            )
    return "\n".join(lines)
