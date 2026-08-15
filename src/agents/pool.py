"""Population training pool and fast greedy opponents for M4 (DESIGN.md §8).

General-sum games cannot rely on pure self-play (it can cycle), so training
opponents are drawn from a mixture: a greedy copy of the current net, recent
past selves (cheap, no search), the heuristic bot, and random liquidity.
`SnapshotPool` stores bounded snapshots of past networks and provides:

- `sample_opponents(...)`: draw `n` opponent agent instances for self-play.
- `policy_agent(name, ...)`: build a fast `PolicyAgent` from a stored snapshot
  (used in the no-collapse eval field).

`PolicyAgent` plays from the net's policy head only (softmax sampling or
argmax), no tree search, so a pool of them costs ~one forward pass per tick.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch

from ..belief.tracker import BeliefTracker
from ..env.deck import ALL_ASSIGNMENTS, N_SUITS
from ..env.state import Action, KIND_PASS
from ..learning.net import FiggieNet
from ..learning.obs import N_ACTIONS, encode, history_features, index_action, legal_mask, trade_counts
from .base import Agent
from .heuristic import HeuristicAgent
from .random import RandomAgent

DEFAULT_MIX = {"self": 0.25, "past": 0.35, "heuristic": 0.25, "random": 0.15}


class PolicyAgent(Agent):
    """Fast opponent that samples/argmaxes the net's policy head (no search)."""

    name = "policy"

    def __init__(
        self,
        net,
        seed: Optional[int] = None,
        n_particles: int = 256,
        temperature: float = 1.0,
    ):
        self.net = net
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature
        self.n_particles = n_particles
        self.player = -1
        self.belief = BeliefTracker(seed, n_particles)
        self._gm = np.ones(N_SUITS) / N_SUITS
        self._post = np.full(len(ALL_ASSIGNMENTS), 1.0 / len(ALL_ASSIGNMENTS))

    def new_round(self, state, player: int) -> None:
        self.player = player
        self.belief = BeliefTracker(int(self.rng.integers(1 << 30)), self.n_particles)
        self.belief.init(state, player)

    def observe(self, state) -> None:
        self.belief.observe(state)

    def choose_action(self, engine, state, player: int) -> Action:
        self.player = player
        self._gm = self.belief.goal_marginals()
        self._post = self.belief.assignment_posterior()
        obs = encode(state, player, self._gm, self._post, trade_counts(state))
        hist = history_features(state, player)
        logits, _v, _b = self.net.evaluate(
            torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(hist).unsqueeze(0)
        )
        if self.temperature > 0:
            probs = torch.softmax(logits / self.temperature, dim=1).squeeze(0).numpy()
        else:
            probs = np.zeros(N_ACTIONS, dtype=float)
            probs[int(logits.squeeze(0).argmax())] = 1.0
        mask = legal_mask(state, player)
        probs = probs * mask
        total = probs.sum()
        if total <= 1e-12:
            return Action(KIND_PASS)
        probs = probs / total
        i = int(self.rng.choice(N_ACTIONS, p=probs))
        return index_action(i)


class SnapshotPool:
    """Bounded store of past network snapshots + mixture opponent sampling."""

    def __init__(self, seed: Optional[int] = None, max_size: int = 4):
        self._rng = random.Random(seed)
        self.max_size = max_size
        self.snapshots: list[tuple[str, dict]] = []

    def __len__(self) -> int:
        return len(self.snapshots)

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.snapshots]

    def add(self, name: str, net) -> None:
        sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        self.snapshots.append((name, sd))
        while len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)

    def make_net(self, state_dict: dict) -> FiggieNet:
        history = "attn" if any(k.startswith("history.proj") for k in state_dict) else "mlp"
        net = FiggieNet(history=history)
        net.load_state_dict(state_dict)
        net.eval()
        return net

    def policy_agent(self, name: str, seed: Optional[int] = None, temperature: float = 1.0) -> PolicyAgent:
        sd = dict(self._state_of(name))
        agent = PolicyAgent(self.make_net(sd), seed=seed, temperature=temperature)
        agent.name = name
        return agent

    def _state_of(self, name: str) -> dict:
        for n, sd in self.snapshots:
            if n == name:
                return sd
        raise KeyError(f"snapshot {name!r} not in pool ({self.names})")

    def sample_opponents(
        self,
        n: int,
        base_net,
        rng: random.Random,
        mix: Optional[dict] = None,
    ) -> list[Agent]:
        """Draw `n` opponent instances from the mixture for a self-play game."""
        mix = dict(DEFAULT_MIX) if mix is None else mix
        kinds = ["self", "past", "heuristic", "random"]
        weights = [mix.get(k, 0.0) for k in kinds]
        opps: list[Agent] = []
        for _ in range(n):
            kind = rng.choices(kinds, weights=weights)[0]
            seed = rng.randrange(1 << 30)
            if kind == "self":
                agent: Agent = PolicyAgent(base_net, seed=seed, temperature=1.0)
                agent.name = "self"
            elif kind == "past" and self.snapshots:
                agent = self.policy_agent(rng.choice(self.names), seed=seed, temperature=1.0)
            else:
                agent = HeuristicAgent(seed) if kind == "heuristic" else RandomAgent(seed)
            opps.append(agent)
        return opps
