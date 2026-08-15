"""M4 population pool: PolicyAgent legality, SnapshotPool sampling, boundedness."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.agents.pool import PolicyAgent, SnapshotPool
from src.agents.random import RandomAgent
from src.env.engine import Engine
from src.env.state import RoundConfig
from src.learning.net import FiggieNet
from src.learning.selfplay import generate_games


def _net() -> FiggieNet:
    net = FiggieNet()
    net.eval()
    return net


def test_policy_agent_plays_legal_round():
    engine = Engine(RoundConfig(n_ticks=60), random.Random(0))
    agents = [PolicyAgent(_net(), seed=1), PolicyAgent(_net(), seed=2), RandomAgent(3), RandomAgent(4)]
    state = engine.new_round()
    for p, a in enumerate(agents):
        a.new_round(state, p)
    for _ in range(engine.config.n_ticks):
        actions = [a.choose_action(engine, state, p) for p, a in enumerate(agents)]
        engine.step(state, actions)
        for p in range(4):
            assert min(state.hands[p]) >= 0, f"negative hand {state.hands[p]}"
            assert state.cash[p] >= 0, f"negative cash {state.cash[p]}"
    deltas = engine.settle(state)
    assert abs(sum(deltas)) < 1e-6


def test_policy_agent_greedy_is_deterministic():
    engine = Engine(RoundConfig(n_ticks=1), random.Random(0))
    base = _net()
    a2net = FiggieNet()
    a2net.load_state_dict(base.state_dict())
    a2net.eval()
    a1 = PolicyAgent(base, seed=7, temperature=0.0)
    a2 = PolicyAgent(a2net, seed=7, temperature=0.0)
    state = engine.new_round()
    a1.new_round(state, 0)
    a2.new_round(state, 0)
    act1 = a1.choose_action(engine, state, 0)
    act2 = a2.choose_action(engine, state, 0)
    assert act1 == act2


def test_pool_add_roundtrip():
    pool = SnapshotPool(seed=0, max_size=3)
    net = _net()
    pool.add("v0", net)
    assert pool.names == ["v0"]
    agent = pool.policy_agent("v0", seed=1, temperature=0.0)
    assert agent.name == "v0"
    assert agent.net is not None


def test_pool_bounded():
    pool = SnapshotPool(seed=0, max_size=2)
    net = _net()
    pool.add("v0", net)
    pool.add("v1", net)
    pool.add("v2", net)
    assert pool.names == ["v1", "v2"]


def test_sample_opponents_shapes_and_names():
    pool = SnapshotPool(seed=1, max_size=4)
    net = _net()
    pool.add("v0", net)
    pool.add("v1", net)
    rng = random.Random(3)
    mix = {"self": 0.5, "past": 0.5, "heuristic": 0.0, "random": 0.0}
    opps = pool.sample_opponents(3, net, rng, mix=mix)
    assert len(opps) == 3
    assert all(o.name in ("self", "v0", "v1") for o in opps)


def test_generate_games_with_pool():
    pool = SnapshotPool(seed=0, max_size=2)
    net = _net()
    obs, hist, pi, z, assign = generate_games(
        net, n_games=2, n_ticks=20, n_sims=5, depth=2, seed=4, pool=pool, mix={"self": 0.5, "past": 0.5, "heuristic": 0.0, "random": 0.0}
    )
    assert len(obs) == len(hist) == len(pi) == len(z) == len(assign) == 2 * 20
    assert set(assign).issubset(set(range(12)))
    assert np.allclose(np.asarray(pi).sum(axis=1), 1.0)
    assert np.isfinite(np.asarray(z)).all()
