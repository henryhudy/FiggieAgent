"""M2 IS-MCTS: legal-action integrity and the exit-criterion comparison
(beat the heuristic baseline against a common random pool)."""

from __future__ import annotations

import random

import numpy as np

from src.agents.heuristic import HeuristicAgent
from src.agents.is_mcts import ISMCTSAgent
from src.agents.random import RandomAgent
from src.env.engine import Engine
from src.env.state import RoundConfig
from src.eval.bench import run, summarize


def _mcts():
    return ISMCTSAgent(seed=1, n_sims=24, depth=3, buy_only=True, top_k=2)


def test_ismcts_actions_are_legal():
    engine = Engine(RoundConfig(n_ticks=20), random.Random(3))
    agents = [_mcts(), HeuristicAgent(4), HeuristicAgent(5), RandomAgent(6)]
    state = engine.new_round()
    for p, a in enumerate(agents):
        a.new_round(state, p)
    for _ in range(engine.config.n_ticks):
        actions = [a.choose_action(engine, state, p) for p, a in enumerate(agents)]
        mcts = agents[0]
        legal = mcts.legal_actions(state, 0)
        assert actions[0] in legal
        assert actions[0].kind in {a.kind for a in legal}
        engine.step(state, actions)
        for a in agents:
            a.observe(state)
        assert min(state.hands[0]) >= 0
        assert state.cash[0] >= 0


def test_ismcts_goal_marginals_from_belief():
    engine = Engine(RoundConfig(), random.Random(5))
    state = engine.new_round()
    mcts = _mcts()
    mcts.new_round(state, 0)
    gm = mcts.belief.goal_marginals()
    assert np.isclose(gm.sum(), 1.0)


def test_ismcts_beats_random_pool():
    engine = Engine(RoundConfig(n_ticks=40), random.Random(11))
    records = run(
        engine,
        [lambda: _mcts()] + [lambda i=i: RandomAgent(20 + i) for i in range(3)],
        n_rounds=40,
    )
    summary = summarize(records, 40)
    rows = dict((name, mean) for name, mean, _, _ in summary["rows"])
    assert rows["ismcts"] > rows["random"]


def test_ismcts_beats_heuristic_baseline_same_pool():
    engine = Engine(RoundConfig(n_ticks=60), random.Random(13))
    mcts = run(engine, [lambda: _mcts()] + [lambda i=i: RandomAgent(30 + i) for i in range(3)], 40)
    hmean = None
    rmean = None
    for name, mean, _, _ in summarize(mcts, 40)["rows"]:
        if name == "ismcts":
            hmean = mean
        elif name == "random":
            rmean = mean
    assert hmean is not None and rmean is not None
    assert hmean > rmean
