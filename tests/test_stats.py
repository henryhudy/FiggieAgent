"""Tests for the benchmark statistics helpers."""

from __future__ import annotations

import random

from src.eval.stats import (
    bootstrap_diff_ci,
    bootstrap_mean_ci,
    cohens_d_paired,
    per_round_deltas,
    report_stats,
    wilcoxon_signed_rank,
)


def _synthetic_records(seed=0, n=120):
    rng = random.Random(seed)
    records = []
    for r in range(n):
        # heuristic gets +1 with noise, cfr gets +0.3, random gets 0
        records.append((r, "heuristic", rng.gauss(1.0, 3.0)))
        records.append((r, "cfr", rng.gauss(0.3, 3.0)))
        records.append((r, "random", rng.gauss(0.0, 3.0)))
    return records


def test_per_round_deltas_groups_by_round():
    groups = per_round_deltas(_synthetic_records())
    assert set(groups) == {"heuristic", "cfr", "random"}
    assert len(groups["heuristic"]) == 120
    assert all(r in groups["cfr"] for r in range(120))


def test_bootstrap_ci_covers_true_mean():
    rng = random.Random(1)
    data = {r: rng.gauss(5.0, 2.0) for r in range(200)}
    mean, lo, hi = bootstrap_mean_ci(data, n_boot=500, seed=7)
    assert abs(mean - 5.0) < 0.5
    assert lo <= mean <= hi


def test_bootstrap_diff_ci():
    groups = per_round_deltas(_synthetic_records())
    mean, lo, hi = bootstrap_diff_ci(groups["heuristic"], groups["random"], n_boot=500, seed=7)
    assert 0.5 < mean < 1.5
    assert lo < mean < hi


def test_wilcoxon_detects_offset():
    groups = per_round_deltas(_synthetic_records())
    w, p = wilcoxon_signed_rank(groups["heuristic"], groups["random"])
    assert p < 0.05  # clear positive offset
    w2, p2 = wilcoxon_signed_rank(groups["random"], groups["heuristic"])
    assert abs(w + w2 - 120 * 121 / 2) < 1e-6  # W+ and W- sum to n(n+1)/2


def test_wilcoxon_null_is_insignificant():
    rng = random.Random(3)
    x = {r: rng.gauss(0.0, 2.0) for r in range(200)}
    y = {r: rng.gauss(0.0, 2.0) for r in range(200)}
    _, p = wilcoxon_signed_rank(x, y)
    assert p > 0.05


def test_cohens_d_positive_for_offset():
    groups = per_round_deltas(_synthetic_records())
    d = cohens_d_paired(groups["heuristic"], groups["random"])
    assert 0.2 < d < 0.6  # ~1.0 offset / ~3.0 sd of diffs
    d_rev = cohens_d_paired(groups["random"], groups["heuristic"])
    assert abs(d + d_rev) < 1e-9


def test_cohens_d_nan_for_empty():
    assert cohens_d_paired({}, {}) != cohens_d_paired({}, {})  # nan != nan



def test_report_stats_runs():
    out = report_stats(_synthetic_records(), n_boot=200, seed=5)
    assert "95% CI" in out
    assert "Wilcoxon" in out
    assert "d=" in out
