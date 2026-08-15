"""Baseline comparison: the heuristic bot should beat random bots."""

from __future__ import annotations

import random

from src.agents.heuristic import HeuristicAgent
from src.agents.random import RandomAgent
from src.env.engine import Engine
from src.env.state import RoundConfig
from src.eval.bench import run, summarize


def test_heuristic_beats_random():
    engine = Engine(RoundConfig(), random.Random(42))
    factories = [lambda: HeuristicAgent(1), lambda: RandomAgent(2), lambda: RandomAgent(3), lambda: RandomAgent(4)]
    records = run(engine, factories, n_rounds=800)
    summary = summarize(records, 800)
    rows = dict((name, (mean, total)) for name, mean, total, _ in summary["rows"])
    h_mean, h_total = rows["heuristic"]
    r_mean, r_total = rows["random"]
    assert h_mean > r_mean
    assert h_total > r_total
    assert summary["wins"]["heuristic"] > 0.5 * 800
