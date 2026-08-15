"""M3 neural search agent: IS-MCTS with neural value/policy (DESIGN.md §7).

Same determinized tree as M2 (`agents/is_mcts.py`): each simulation samples a
consistent world from the belief tracker, opponents follow the heuristic policy
inside the world, and the tree branches on our buy-only actions. Differences:

- Leaf values come from the network's value head (trained on terminal delta/50)
  instead of the closed-form "hold" formula or full rollouts.
- In-tree selection is PUCT using the network's policy head as the action prior
  (masked to legal actions), replacing the uniform UCB1 expansion.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import torch

from ..belief.tracker import BeliefTracker
from ..env.deck import ALL_ASSIGNMENTS, N_SUITS
from ..env.state import (
    Action,
    KIND_ACCEPT_ASK,
    KIND_BUNDLE_BID,
    KIND_PASS,
    KIND_POST_BID,
    RoundState,
)
from ..learning.obs import (
    N_ACTIONS,
    PRICE_GRID,
    action_index,
    encode,
    history_features,
    trade_counts,
)
from ..env.state import BUNDLE_PAIRS
from .base import Agent
from .heuristic import HeuristicAgent
from .is_mcts import _Node, _copy_state

_PASS = Action(KIND_PASS)


class NeuralAgent(Agent):
    name = "neural"

    def __init__(
        self,
        net,
        seed: Optional[int] = None,
        n_sims: int = 100,
        depth: int = 4,
        n_particles: int = 512,
        c_puct: float = 1.0,
        expand_visits: int = 6,
        top_k: Optional[int] = None,
        use_learned_belief: bool = False,
        bayes_combine: bool = False,
    ):
        self.net = net
        self.rng = random.Random(seed)
        self.n_sims = n_sims
        self.depth = depth
        self.n_particles = n_particles
        self.c_puct = c_puct
        self.expand_visits = expand_visits
        self.top_k = top_k
        self.use_learned_belief = use_learned_belief
        self.bayes_combine = bayes_combine
        self.player = -1
        self.belief = BeliefTracker(seed, n_particles)
        self._gm = np.ones(N_SUITS) / N_SUITS
        self._post = np.full(len(ALL_ASSIGNMENTS), 1.0 / len(ALL_ASSIGNMENTS))
        self._trade_counts = np.zeros(N_SUITS, dtype=np.float32)
        self._hist = np.zeros(0, dtype=np.float32)
        self._top_suits: Optional[list[int]] = None
        self._last_policy: Optional[np.ndarray] = None
        self._last_value: float = 0.0
        self.fallback = HeuristicAgent(seed)

    def new_round(self, state, player: int) -> None:
        self.player = player
        self.belief = BeliefTracker(self.rng.randrange(1 << 30), self.n_particles)
        self.belief.init(state, player)
        self._trade_counts = trade_counts(state)
        self._post = self.belief.assignment_posterior()
        self._gm = self.belief.goal_marginals()
        self.fallback.new_round(state, player)

    def observe(self, state) -> None:
        self.belief.observe(state)

    # ------------------------------------------------------------- abstraction

    def legal_actions(self, state, player: int) -> list[Action]:
        acts = [_PASS]
        cash = state.cash[player]
        suits = self._top_suits if self._top_suits is not None else range(N_SUITS)
        for s in suits:
            ba = state.best_ask(s)
            if ba is not None and ba[1] != player and cash >= ba[0]:
                acts.append(Action(KIND_ACCEPT_ASK, s))
            for price in PRICE_GRID:
                if cash >= price:
                    acts.append(Action(KIND_POST_BID, s, price))
        for s1, s2 in BUNDLE_PAIRS:
            for price in PRICE_GRID:
                if cash >= price:
                    acts.append(Action(KIND_BUNDLE_BID, suit=s1, suit2=s2, price=price))
        return acts

    # ---------------------------------------------------------------- search

    def choose_action(self, engine, state, player: int) -> Action:
        action, _policy, _value = self.search(engine, state, player)
        return action

    def search(self, engine, state, player: int) -> tuple[Action, np.ndarray, float]:
        """Run IS-MCTS. Returns (action, 21-dim visit distribution, root value)."""
        self.player = player
        self._trade_counts = trade_counts(state)
        self._hist = history_features(state, player)
        if self.use_learned_belief:
            # Decoupled learned belief (P2): the obs always encodes the exact
            # strategy-free posterior (same distribution the policy/value heads
            # were trained on), and the belief head refines it into a
            # behavior-conditioned prior. That prior only re-weights MCTS world
            # sampling inside the exact-consistent set (`set_assignment_prior`),
            # so the learned belief can never suggest impossible worlds.
            self.belief.set_assignment_prior(None)
            self._post = self.belief.assignment_posterior()
            self._gm = self.belief.goal_marginals()
            obs_b = encode(state, player, self._gm, self._post, self._trade_counts)
            _p, _v, b = self.net.evaluate(
                torch.from_numpy(obs_b).unsqueeze(0), torch.from_numpy(self._hist).unsqueeze(0)
            )
            learned = torch.softmax(b, dim=1).squeeze(0).numpy()
            if self.bayes_combine:
                # Push: Bayes-combine the exact flow-consistent posterior with
                # the learned behavior signal into one posterior used both for
                # the policy/value obs and for world sampling. The product
                # P_exact(a) * P_learned(a) reweights only the exact-consistent
                # support, so impossible worlds stay out.
                combined = self._post * learned
                s = combined.sum()
                if s > 1e-12:
                    combined = combined / s
                self._post = combined
                self._gm = np.zeros(N_SUITS)
                for ai, p in enumerate(combined):
                    self._gm[ALL_ASSIGNMENTS[ai].goal_suit] += p
            self.belief.set_assignment_prior(self._post)
        else:
            self._gm = self.belief.goal_marginals()
            self._post = self.belief.assignment_posterior()
        self._top_suits = None
        if self.top_k is not None:
            self._top_suits = list(np.argsort(self._gm)[::-1][: self.top_k])

        root = _Node()
        legal = self.legal_actions(state, player)
        for _ in range(self.n_sims):
            try:
                assignment, hands = self.belief.sample_world()
            except RuntimeError:
                return self.fallback.choose_action(engine, state, player), np.zeros(
                    N_ACTIONS
                ), 0.0
            self._simulate(engine, state, assignment, hands, root, legal)

        dist = np.zeros(N_ACTIONS, dtype=np.float32)
        total = 0
        for a in legal:
            c = root.children.get(a)
            if c is not None:
                dist[action_index(a)] = c.n
                total += c.n
        if total > 0:
            dist /= total
        self._last_policy = dist
        self._last_value = root.q / root.n if root.n > 0 else 0.0

        best = max(legal, key=lambda a: root.children[a].n if a in root.children else -1)
        if best not in root.children or root.children[best].n == 0:
            return self.fallback.choose_action(engine, state, player), dist, self._last_value
        return best, dist, self._last_value

    def _simulate(self, engine, state, assignment, hands, root, root_legal) -> None:
        sim = _copy_state(state, hands, assignment)
        n = state.config.n_players
        agents = [HeuristicAgent(self.rng.randrange(1 << 30)) for _ in range(n)]
        for j in range(n):
            agents[j].new_round(sim, j)
        node = root
        path = [root]
        depth = 0
        while depth < self.depth and sim.tick < state.config.n_ticks:
            legal = self.legal_actions(sim, self.player)
            action, node = self._select(node, legal, sim)
            acts = [None] * n
            for j in range(n):
                if j == self.player:
                    acts[j] = action
                else:
                    acts[j] = agents[j].choose_action(engine, sim, j)
            engine.step(sim, acts)
            for j in range(n):
                agents[j].observe(sim)
            path.append(node)
            depth += 1
        v = self._leaf_value(sim)
        for nd in path:
            nd.n += 1
            nd.q += v

    def _leaf_value(self, sim: RoundState) -> float:
        obs = encode(sim, self.player, self._gm, self._post, self._trade_counts)
        _p, v, _b = self.net.evaluate(
            torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(self._hist).unsqueeze(0)
        )
        return float(v[0, 0].item())

    def _select(self, node, legal: list[Action], state: RoundState) -> tuple[Action, object]:
        obs = encode(state, self.player, self._gm, self._post, self._trade_counts)
        logits, _v, _b = self.net.evaluate(
            torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(self._hist).unsqueeze(0)
        )
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        idx = [action_index(a) for a in legal]
        prior = probs[idx]
        s = prior.sum()
        prior = prior / s if s > 1e-12 else np.ones(len(idx)) / len(idx)

        if node.n < self.expand_visits:
            unexpanded = [a for a in legal if a not in node.children]
            if unexpanded:
                a = unexpanded[self.rng.randrange(len(unexpanded))]
                child = node.children.get(a)
                if child is None:
                    child = node.children[a] = _Node()
                return a, child

        best_i = max(
            range(len(legal)),
            key=lambda i: self._puct(node, legal[i], prior[i]),
        )
        a = legal[best_i]
        child = node.children.get(a)
        if child is None:
            child = node.children[a] = _Node()
        return a, child

    def _puct(self, node, a: Action, prior: float) -> float:
        child = node.children.get(a)
        n = child.n if child is not None else 0
        q = child.q / n if n > 0 else 0.0
        return q + self.c_puct * prior * math.sqrt(max(node.n, 1)) / (1 + n)
