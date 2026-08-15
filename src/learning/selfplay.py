"""Self-play game generation for the neural seat.

The neural agent plays via IS-MCTS search against a population pool of
opponents (M4): a greedy copy of the current net, past selves, heuristic, and
random. With no pool supplied it falls back to random opponents (the liquid
common pool used for M2/M3 evaluation). At every tick we record the
observation, the search's visit distribution pi, and (once the round ends) the
terminal delta z = settle/50 for the neural seat.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from ..agents.neural import NeuralAgent
from ..agents.random import RandomAgent
from ..env.deck import ALL_ASSIGNMENTS
from ..env.engine import Engine
from ..env.state import RoundConfig
from ..learning.net import VALUE_SCALE
from ..learning.obs import encode, history_features


def play_one(engine: Engine, neural: NeuralAgent, opponents: list, seat: int, rng: random.Random) -> tuple[list, list, list, list, list, int]:
    agents = opponents[:]
    state = engine.new_round()
    agents.insert(seat, neural)
    for p, a in enumerate(agents):
        a.new_round(state, p)
    obs_list: list[np.ndarray] = []
    hist_list: list[np.ndarray] = []
    pi_list: list[np.ndarray] = []
    assign_idx = ALL_ASSIGNMENTS.index(state.assignment)
    for _ in range(engine.config.n_ticks):
        action, pi, _v = neural.search(engine, state, seat)
        obs_list.append(
            encode(state, seat, neural._gm, neural._post, neural._trade_counts)
        )
        hist_list.append(neural._hist)
        pi_list.append(pi)
        actions = [None] * len(agents)
        for p, a in enumerate(agents):
            actions[p] = action if p == seat else a.choose_action(engine, state, p)
        engine.step(state, actions)
        for a in agents:
            a.observe(state)
    deltas = engine.settle(state)
    z_all = np.asarray(deltas[seat:] + deltas[:seat], dtype=np.float32) / VALUE_SCALE
    return obs_list, hist_list, pi_list, z_all, assign_idx


def generate_games(
    net,
    n_games: int,
    n_players: int = 4,
    n_ticks: int = 120,
    n_sims: int = 60,
    depth: int = 4,
    seed: int = 0,
    pool=None,
    mix=None,
    learned_belief: bool = False,
) -> tuple[list, list, list, list, list]:
    rng = random.Random(seed)
    engine = Engine(RoundConfig(n_players=n_players, n_ticks=n_ticks), rng)
    neural = NeuralAgent(net, seed=seed, n_sims=n_sims, depth=depth, use_learned_belief=learned_belief)
    all_obs: list[np.ndarray] = []
    all_hist: list[np.ndarray] = []
    all_pi: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    all_assign: list[int] = []
    for g in range(n_games):
        neural.rng = random.Random(seed + 1000 * (g + 1))
        if pool is not None:
            opponents = pool.sample_opponents(n_players - 1, net, rng, mix=mix)
        else:
            opponents = [RandomAgent(rng.randrange(1 << 30)) for _ in range(n_players - 1)]
        seat = g % n_players
        obs, hist, pi, z_all, assign_idx = play_one(engine, neural, opponents, seat, rng)
        all_obs.extend(obs)
        all_hist.extend(hist)
        all_pi.extend(pi)
        all_z.extend([z_all] * len(obs))
        all_assign.extend([assign_idx] * len(obs))
    return all_obs, all_hist, all_pi, all_z, all_assign
