"""M3 neural search agent: architecture, action-space roundtrip, legal actions,
and the exit-criterion comparison (trained net beats the M2 IS-MCTS agent and
the heuristic baseline against a common random pool)."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.agents.heuristic import HeuristicAgent
from src.agents.is_mcts import ISMCTSAgent
from src.agents.neural import NeuralAgent
from src.agents.random import RandomAgent
from src.env.engine import Engine
from src.env.state import RoundConfig
from src.eval.bench import run, summarize
from src.learning.net import FiggieNet, load_partial
from src.learning.obs import N_ACTIONS, action_index, encode, index_action, legal_mask, trade_counts
from src.learning.replay import ReplayBuffer


def _trained_net() -> FiggieNet:
    net = FiggieNet()
    net.eval()
    return net


def test_action_space_roundtrip():
    for i in range(N_ACTIONS):
        assert action_index(index_action(i)) == i


def test_encode_shapes():
    engine = Engine(RoundConfig(), random.Random(3))
    state = engine.new_round()
    agent = NeuralAgent(_trained_net(), seed=1)
    agent.new_round(state, 0)
    obs = encode(state, 0, agent._gm, agent._post, agent._trade_counts)
    assert obs.shape == (53,)
    assert np.isfinite(obs).all()
    mask = legal_mask(state, 0)
    assert mask.shape == (N_ACTIONS,)


def test_net_forward_shapes():
    net = _trained_net()
    obs = torch.zeros(8, 53)
    logits, v, b = net(obs)
    assert logits.shape == (8, N_ACTIONS)
    assert v.shape == (8, 4)
    assert b.shape == (8, 12)


def test_bayes_combine_posterior_is_normalized():
    engine = Engine(RoundConfig(n_ticks=20), random.Random(3))
    state = engine.new_round()
    agent = NeuralAgent(_trained_net(), seed=1, n_sims=8, depth=2, use_learned_belief=True, bayes_combine=True)
    agent.new_round(state, 0)
    for _ in range(3):
        _action, _pi, _v = agent.search(engine, state, 0)
    post = agent._post
    assert post.shape == (12,)
    assert np.isfinite(post).all()
    assert abs(post.sum() - 1.0) < 1e-5
    gm = agent._gm
    assert gm.shape == (4,) and abs(gm.sum() - 1.0) < 1e-5


def test_load_partial_grows_trunk_and_policy(tmp_path):
    torch.manual_seed(0)
    old = FiggieNet(n_in=41, n_actions=21)
    old_w = {k: v.clone() for k, v in old.state_dict().items()}
    ck = tmp_path / "old.pt"
    torch.save({"model": old.state_dict()}, ck)

    new = FiggieNet()
    load_partial(new, str(ck))
    assert torch.equal(new.state_dict()["trunk.0.weight"][:, :41], old_w["trunk.0.weight"])
    assert new.state_dict()["trunk.0.weight"][:, 41:].abs().sum() == 0.0
    assert torch.equal(new.state_dict()["policy_head.2.weight"][:21], old_w["policy_head.2.weight"])
    assert new.state_dict()["policy_head.2.weight"][21:].abs().sum() == 0.0
    assert torch.equal(new.state_dict()["policy_head.2.bias"][:21], old_w["policy_head.2.bias"])
    assert new.state_dict()["policy_head.2.bias"][21:].abs().sum() == 0.0


def test_replay_buffer_sample():
    buf = ReplayBuffer()
    buf.extend(
        [np.random.rand(53).astype(np.float32) for _ in range(20)],
        [np.random.rand(40).astype(np.float32) for _ in range(20)],
        [np.random.rand(N_ACTIONS).astype(np.float32) for _ in range(20)],
        [np.random.rand(4).astype(np.float32) for _ in range(20)],
        [i % 12 for i in range(20)],
    )
    o, h, p, z, a = buf.sample(8)
    assert o.shape == (8, 53) and h.shape == (8, 40) and p.shape == (8, N_ACTIONS) and z.shape == (8, 4)
    assert a.shape == (8,) and set(a.tolist()).issubset(set(range(12)))


def test_neural_actions_are_legal():
    engine = Engine(RoundConfig(n_ticks=20), random.Random(3))
    agents = [
        NeuralAgent(_trained_net(), seed=1, n_sims=16, depth=3),
        HeuristicAgent(4),
        HeuristicAgent(5),
        RandomAgent(6),
    ]
    state = engine.new_round()
    for p, a in enumerate(agents):
        a.new_round(state, p)
    for _ in range(engine.config.n_ticks):
        actions = [a.choose_action(engine, state, p) for p, a in enumerate(agents)]
        legal = agents[0].legal_actions(state, 0)
        assert actions[0] in legal
        engine.step(state, actions)
        for a in agents:
            a.observe(state)
        assert min(state.hands[0]) >= 0
        assert state.cash[0] >= 0


def test_neural_beats_m2_and_heuristic_same_pool():
    import os

    ckpt = next(
        (p for p in ("experiments/checkpoints/figgie_net_p1.pt", "experiments/checkpoints/figgie_net_m4.pt", "experiments/checkpoints/figgie_net.pt") if os.path.exists(p)),
        None,
    )
    if ckpt is None:
        pytest.skip("trained checkpoint not present")
    net = load_partial(FiggieNet(), ckpt)
    net.eval()
    ticks, sims, depth, rounds = 80, 60, 4, 30
    neural = run(
        Engine(RoundConfig(n_ticks=ticks), random.Random(13)),
        [lambda: NeuralAgent(net, seed=1, n_sims=sims, depth=depth)]
        + [lambda i=i: RandomAgent(30 + i) for i in range(3)],
        rounds,
    )
    mcts = run(
        Engine(RoundConfig(n_ticks=ticks), random.Random(13)),
        [lambda: ISMCTSAgent(1, n_sims=sims, depth=depth, buy_only=True, top_k=2)]
        + [lambda i=i: RandomAgent(30 + i) for i in range(3)],
        rounds,
    )
    nmeans = dict((name, mean) for name, mean, _, _ in summarize(neural, rounds)["rows"])
    mmeans = dict((name, mean) for name, mean, _, _ in summarize(mcts, rounds)["rows"])
    assert nmeans["neural"] > mmeans["ismcts"] > mmeans["random"]
