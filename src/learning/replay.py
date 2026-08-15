"""Replay buffer of (obs, trade-history, MCTS policy pi, per-player value z,
true assignment index for the learned-belief head)."""

from __future__ import annotations

from typing import Optional

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.obs: list[np.ndarray] = []
        self.hist: list[np.ndarray] = []
        self.pi: list[np.ndarray] = []
        self.z: list[np.ndarray] = []
        self.assign: list[int] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.obs)

    def add(self, obs: np.ndarray, hist: np.ndarray, pi: np.ndarray, z: np.ndarray, assign: int) -> None:
        if self.capacity and len(self.obs) >= self.capacity:
            i = self.pos % self.capacity
            self.obs[i] = obs
            self.hist[i] = hist
            self.pi[i] = pi
            self.z[i] = z
            self.assign[i] = assign
            self.pos += 1
        else:
            self.obs.append(obs)
            self.hist.append(hist)
            self.pi.append(pi)
            self.z.append(z)
            self.assign.append(assign)

    def extend(self, obs, hist, pi, z, assign) -> None:
        for o, h, p, zz, a in zip(obs, hist, pi, z, assign):
            self.add(o, h, p, zz, a)

    def sample(self, batch: int, rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, len(self.obs), size=batch)
        return (
            np.stack([self.obs[i] for i in idx]),
            np.stack([self.hist[i] for i in idx]),
            np.stack([self.pi[i] for i in idx]),
            np.stack([self.z[i] for i in idx]),
            np.asarray([self.assign[i] for i in idx], dtype=np.int64),
        )
