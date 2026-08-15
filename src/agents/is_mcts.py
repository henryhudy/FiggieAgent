"""Information-Set MCTS (determinized) with heuristic rollouts: M2.

At each tick the agent samples consistent worlds from the exact belief tracker
and runs a tree over its own action sequence: opponents follow the heuristic
policy inside the sampled world, so a world fully determines how the game
advances given our actions. Each simulation fixes a world, plays `depth`
in-tree ticks (our action by UCB1, opponents by heuristic), then rolls the rest
out with the heuristic for everyone; the terminal settle delta is backed up.
The root action with the most visits is played.

This is the PIMC / determinized IS-MCTS design from DESIGN.md §7, with a
depth-limited tree and heuristic continuation. Opponent *policies* are the
heuristic bot itself (not searched); the M1 probe showed behavior-conditioning
their beliefs adds little, so the search's value comes from lookahead over the
strategy-free belief.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..belief.tracker import BeliefTracker
from ..env.deck import N_SUITS
from ..env.state import (
    Action,
    KIND_ACCEPT_ASK,
    KIND_ACCEPT_BID,
    KIND_PASS,
    KIND_POST_ASK,
    KIND_POST_BID,
    KIND_WITHDRAW,
    RoundState,
)
from .base import Agent
from .heuristic import HeuristicAgent

DEFAULT_PRICE_GRID = (6, 12, 18, 24)
CARD_VALUE = 10.0


@dataclass
class _Node:
    n: int = 0
    q: float = 0.0
    children: dict = field(default_factory=dict)


def _copy_state(state: RoundState, hands, assignment) -> RoundState:
    """Fresh sim state: world hands/assignment, real quotes/cash, no trades."""
    return RoundState(
        config=state.config,
        assignment=assignment,
        hands=[h[:] for h in hands],
        cash=list(state.cash),
        quotes={p: dict(qs) for p, qs in state.quotes.items()},
        tick=state.tick,
        trades=[],
    )


class ISMCTSAgent(Agent):
    name = "ismcts"

    def __init__(
        self,
        seed: Optional[int] = None,
        n_sims: int = 100,
        depth: int = 6,
        n_particles: int = 512,
        price_grid=DEFAULT_PRICE_GRID,
        c: float = 2.0,
        expand_visits: int = 4,
        buy_only: bool = False,
        top_k: Optional[int] = None,
        value_mode: str = "hold",
    ):
        self.rng = random.Random(seed)
        self.n_sims = n_sims
        self.depth = depth
        self.n_particles = n_particles
        self.price_grid = price_grid
        self.c = c
        self.expand_visits = expand_visits
        self.buy_only = buy_only
        self.top_k = top_k
        self.value_mode = value_mode
        self.player = -1
        self.belief = BeliefTracker(seed, n_particles)
        self.fallback = HeuristicAgent(seed)
        self._gm = np.ones(N_SUITS) / N_SUITS

    def new_round(self, state, player: int) -> None:
        self.player = player
        self.belief = BeliefTracker(self.rng.randrange(1 << 30), self.n_particles)
        self.belief.init(state, player)
        self.fallback.new_round(state, player)

    def observe(self, state) -> None:
        self.belief.observe(state)

    # ------------------------------------------------------------- abstraction

    def legal_actions(self, state, player: int) -> list[Action]:
        """Masked action abstraction: pass, accepts, withdraws, coarse bid/ask
        price buckets (DESIGN.md §6.3). When top_k is set, buy actions are
        limited to the belief's top-k goal suits."""
        acts = [Action(KIND_PASS)]
        cash = state.cash[player]
        hand = state.hands[player]
        own = state.quotes.get(player, {})
        top = set()
        if self.top_k is not None:
            top = {s for s in np.argsort(self._gm)[::-1][: self.top_k].tolist()}
        for s in range(N_SUITS):
            if s in top or self.top_k is None:
                ba = state.best_ask(s)
                if ba is not None and ba[1] != player and cash >= ba[0]:
                    acts.append(Action(KIND_ACCEPT_ASK, s))
            if not self.buy_only:
                bb = state.best_bid(s)
                if bb is not None and bb[1] != player and hand[s] >= 1:
                    acts.append(Action(KIND_ACCEPT_BID, s))
                if s in own:
                    acts.append(Action(KIND_WITHDRAW, s))
            if s in top or self.top_k is None:
                for price in self.price_grid:
                    if cash >= price:
                        acts.append(Action(KIND_POST_BID, s, price))
                    if not self.buy_only and hand[s] >= 1:
                        acts.append(Action(KIND_POST_ASK, s, price))
        return acts

    # ---------------------------------------------------------------- search

    def choose_action(self, engine, state, player: int) -> Action:
        self.player = player
        self._gm = self.belief.goal_marginals()
        root = _Node()
        for _ in range(self.n_sims):
            try:
                assignment, hands = self.belief.sample_world()
            except RuntimeError:
                return self.fallback.choose_action(engine, state, player)
            self._simulate(engine, state, assignment, hands, root)
        legal = self.legal_actions(state, player)
        best = max(
            legal,
            key=lambda a: root.children[a].n if a in root.children else -1,
        )
        if best not in root.children or root.children[best].n == 0:
            return self.fallback.choose_action(engine, state, player)
        return best

    def _simulate(self, engine, state, assignment, hands, root: _Node) -> None:
        sim = _copy_state(state, hands, assignment)
        n = state.config.n_players
        agents = [HeuristicAgent(self.rng.randrange(1 << 30)) for _ in range(n)]
        for j in range(n):
            agents[j].new_round(sim, j)
        node = root
        path = [root]
        depth = 0
        while depth < self.depth and sim.tick < state.config.n_ticks:
            action, node = self._select(node, self.legal_actions(sim, self.player))
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
        if self.value_mode == "world":
            hand = sim.hands[self.player]
            leaf = (sim.cash[self.player] - sim.ante) + CARD_VALUE * hand[assignment.goal_suit]
        elif self.value_mode == "hold":
            hand = sim.hands[self.player]
            leaf = (sim.cash[self.player] - sim.ante) + CARD_VALUE * sum(
                hand[s] * self._gm[s] for s in range(N_SUITS)
            )
        else:
            while sim.tick < state.config.n_ticks:
                acts = [agents[j].choose_action(engine, sim, j) for j in range(n)]
                engine.step(sim, acts)
                for j in range(n):
                    agents[j].observe(sim)
            leaf = engine.settle(sim)[self.player]
        for nd in path:
            nd.n += 1
            nd.q += leaf

    def _select(self, node: _Node, legal: list[Action]) -> tuple[Action, _Node]:
        unexpanded = [a for a in legal if a not in node.children]
        if unexpanded and node.n < self.expand_visits + len(node.children):
            a = unexpanded[self.rng.randrange(len(unexpanded))]
        else:
            a = max(legal, key=lambda a: self._ucb(node, a))
        child = node.children.get(a)
        if child is None:
            child = node.children[a] = _Node()
        return a, child

    def _ucb(self, node: _Node, a: Action) -> float:
        child = node.children.get(a)
        if child is None or child.n == 0:
            return -math.inf
        return child.q / child.n + self.c * math.sqrt(math.log(max(node.n, 1)) / child.n)
