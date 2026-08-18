"""A compact Deep CFR implementation for the Figgie-Lite reduction.

The full Figgie environment is multi-player and general-sum, so the usual
two-player zero-sum CFR guarantees do not apply there.  This module instead
uses the explicit Figgie-Lite game from :mod:`figgie_lite`: an exact chance
model, two players, and a zero-sum settlement payoff.  It follows the Deep
CFR recipe with one neural advantage model per player, reservoir memories for
advantage and average-strategy samples, and a final neural average strategy.

It is intentionally small and inspectable.  The exact evaluator in
``cfr.py`` is used for reporting exploitability; no full-Figgie equilibrium
claim is implied by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .cfr import Strategy, uniform
from .figgie_lite import (
    ACTION_COUNT,
    SPLITS,
    apply_buy,
    encode_joint,
    outcome_conditioned_posterior,
    payoff,
)


def _sample(probs: list[float], rng: random.Random) -> int:
    threshold = rng.random()
    total = 0.0
    for action, probability in enumerate(probs):
        total += probability
        if threshold <= total:
            return action
    return len(probs) - 1


def _regret_matching(advantages: torch.Tensor) -> list[float]:
    positive = torch.clamp(advantages, min=0.0)
    total = float(positive.sum())
    if total <= 1e-12:
        return uniform(ACTION_COUNT)
    return [float(value / total) for value in positive]


class _Network(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ACTION_COUNT),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class _Reservoir:
    """Fixed-size uniform reservoir used by Deep CFR's replay memories."""

    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self.rng = rng
        self.items: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        self.seen = 0

    def add(self, features: torch.Tensor, target: torch.Tensor, weight: float = 1.0) -> None:
        self.seen += 1
        item = (features.detach().cpu(), target.detach().cpu(), float(weight))
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        slot = self.rng.randrange(self.seen)
        if slot < self.capacity:
            self.items[slot] = item

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.items:
            raise ValueError("cannot sample an empty reservoir")
        batch = [self.items[self.rng.randrange(len(self.items))] for _ in range(batch_size)]
        return (
            torch.stack([row[0] for row in batch]),
            torch.stack([row[1] for row in batch]),
            torch.tensor([row[2] for row in batch], dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True)
class DeepCFRConfig:
    ticks: int = 2
    hidden_size: int = 64
    memory_capacity: int = 20_000
    learning_rate: float = 2e-3
    batch_size: int = 128
    advantage_steps: int = 40
    strategy_steps: int = 160
    train_every: int = 25
    include_posterior: bool = True


class DeepCFRStrategy(Strategy):
    """The learned average strategy, exposed to the exact CFR evaluator."""

    def __init__(self, model: _Network, encoder: "InfoSetEncoder"):
        self.model = model.eval()
        self.encoder = encoder

    def probs(self, player: int, hand: tuple, history: tuple) -> list[float]:
        with torch.no_grad():
            logits = self.model(self.encoder.encode(player, hand, history).unsqueeze(0))[0]
            return [float(value) for value in torch.softmax(logits, dim=-1)]


class InfoSetEncoder:
    """Fixed-width encoding of a Figgie-Lite information set."""

    def __init__(self, ticks: int, include_posterior: bool = True):
        self.ticks = ticks
        self.include_posterior = include_posterior
        # Hand counts (4), optional exact card-consistency posterior (4),
        # acting player (2), and each public history item: two one-hot actions
        # plus the two public buy-success flags.
        self.size = 6 + (4 if include_posterior else 0) + ticks * (2 * ACTION_COUNT + 2)

    def encode(self, player: int, hand: tuple, history: tuple) -> torch.Tensor:
        if len(history) > self.ticks:
            raise ValueError("history is longer than the configured horizon")
        values = [count / 3.0 for count in hand]
        if self.include_posterior:
            values.extend(outcome_conditioned_posterior(player, hand, history))
        values.extend([1.0 if player == 0 else 0.0, 1.0 if player == 1 else 0.0])
        for index in range(self.ticks):
            row = [0.0] * (2 * ACTION_COUNT + 2)
            if index < len(history):
                joint = history[index]
                a0 = (joint >> 2) // ACTION_COUNT
                a1 = (joint >> 2) % ACTION_COUNT
                row[a0] = 1.0
                row[ACTION_COUNT + a1] = 1.0
                row[2 * ACTION_COUNT] = float(bool((joint >> 1) & 1))
                row[2 * ACTION_COUNT + 1] = float(bool(joint & 1))
            values.extend(row)
        return torch.tensor(values, dtype=torch.float32)


class DeepCFR:
    """External-sampling Deep CFR for Figgie-Lite.

    One traversal is performed for each player per iteration.  The traversing
    player enumerates actions while the opponent action is sampled from its
    current advantage policy.  This is the standard external-sampling shape;
    sampled advantage targets and behaviour strategies are kept in uniform
    reservoir memories and fitted by small neural networks.
    """

    def __init__(self, config: DeepCFRConfig = DeepCFRConfig(), seed: Optional[int] = None):
        self.config = config
        self.rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(seed)
        self.encoder = InfoSetEncoder(config.ticks, config.include_posterior)
        self.advantage_models = [_Network(self.encoder.size, config.hidden_size) for _ in range(2)]
        self.strategy_model = _Network(self.encoder.size, config.hidden_size)
        self.advantage_memory = [
            _Reservoir(config.memory_capacity, random.Random(self.rng.randrange(2**31)))
            for _ in range(2)
        ]
        self.strategy_memory = _Reservoir(
            config.memory_capacity, random.Random(self.rng.randrange(2**31))
        )
        self.iterations = 0

    def _policy(self, player: int, hand: tuple, history: tuple) -> list[float]:
        with torch.no_grad():
            values = self.advantage_models[player](
                self.encoder.encode(player, hand, history).unsqueeze(0)
            )[0]
        return _regret_matching(values)

    def _record_strategy(self, player: int, hand: tuple, history: tuple, policy: list[float], reach: float) -> None:
        # The average-strategy memory is weighted by the traverser's own
        # reach, matching the reach-weighted averaging used by CFR.
        self.strategy_memory.add(
            self.encoder.encode(player, hand, history),
            torch.tensor(policy, dtype=torch.float32),
            max(reach, 1e-8),
        )

    def _traverse(self, hands, cash, goal: int, history: tuple, tick: int,
                  traverser: int, traverser_reach: float) -> float:
        if tick >= self.config.ticks:
            value = payoff(hands, cash, goal)
            return value if traverser == 0 else -value

        h0, h1 = tuple(hands[0]), tuple(hands[1])
        s0 = self._policy(0, h0, history)
        s1 = self._policy(1, h1, history)

        if traverser == 0:
            self._record_strategy(0, h0, history, s0, traverser_reach)
            sampled_a1 = _sample(s1, self.rng)
            values: list[float] = []
            for a0 in range(ACTION_COUNT):
                h_after_0, c_after_0, success_0 = apply_buy(hands, cash, 0, a0)
                h_after_1, c_after_1, success_1 = apply_buy(
                    h_after_0, c_after_0, 1, sampled_a1
                )
                joint = encode_joint(a0, sampled_a1, success_0, success_1)
                values.append(self._traverse(
                    h_after_1, c_after_1, goal, history + (joint,), tick + 1,
                    traverser, traverser_reach * s0[a0]
                ))
            expected = sum(probability * value for probability, value in zip(s0, values))
            regrets = torch.tensor([value - expected for value in values], dtype=torch.float32)
            self.advantage_memory[0].add(self.encoder.encode(0, h0, history), regrets)
            return expected

        self._record_strategy(1, h1, history, s1, traverser_reach)
        sampled_a0 = _sample(s0, self.rng)
        values = []
        for a1 in range(ACTION_COUNT):
            h_after_0, c_after_0, success_0 = apply_buy(hands, cash, 0, sampled_a0)
            h_after_1, c_after_1, success_1 = apply_buy(
                h_after_0, c_after_0, 1, a1
            )
            joint = encode_joint(sampled_a0, a1, success_0, success_1)
            values.append(self._traverse(
                h_after_1, c_after_1, goal, history + (joint,), tick + 1,
                traverser, traverser_reach * s1[a1]
            ))
        expected = sum(probability * value for probability, value in zip(s1, values))
        regrets = torch.tensor([value - expected for value in values], dtype=torch.float32)
        self.advantage_memory[1].add(self.encoder.encode(1, h1, history), regrets)
        return expected

    @staticmethod
    def _fit_advantage(model: _Network, memory: _Reservoir, steps: int, batch_size: int, learning_rate: float) -> None:
        if not memory:
            return
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.train()
        for _ in range(steps):
            features, targets, _ = memory.sample(min(batch_size, len(memory)))
            loss = F.mse_loss(model(features), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def iterate(self, iterations: int) -> None:
        for _ in range(iterations):
            goal = self.rng.randrange(4)
            h0, h1 = SPLITS[goal][self.rng.randrange(len(SPLITS[goal]))]
            for player in range(2):
                self._traverse([list(h0), list(h1)], [100.0, 100.0], goal, (), 0, player, 1.0)
            self.iterations += 1
            if self.iterations % self.config.train_every == 0:
                for player in range(2):
                    self._fit_advantage(
                        self.advantage_models[player], self.advantage_memory[player],
                        self.config.advantage_steps, self.config.batch_size, self.config.learning_rate,
                    )

        # Do not leave the final partial block untrained.
        for player in range(2):
            self._fit_advantage(
                self.advantage_models[player], self.advantage_memory[player],
                self.config.advantage_steps, self.config.batch_size, self.config.learning_rate,
            )

    def average_strategy(self) -> DeepCFRStrategy:
        if not self.strategy_memory:
            raise RuntimeError("call iterate before fitting an average strategy")
        optimizer = torch.optim.Adam(self.strategy_model.parameters(), lr=self.config.learning_rate)
        self.strategy_model.train()
        for _ in range(self.config.strategy_steps):
            features, targets, weights = self.strategy_memory.sample(
                min(self.config.batch_size, len(self.strategy_memory))
            )
            log_probs = F.log_softmax(self.strategy_model(features), dim=-1)
            cross_entropy = -(targets * log_probs).sum(dim=-1)
            loss = (cross_entropy * weights).sum() / weights.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return DeepCFRStrategy(self.strategy_model, self.encoder)

    def memory_sizes(self) -> tuple[int, int, int]:
        return len(self.advantage_memory[0]), len(self.advantage_memory[1]), len(self.strategy_memory)
