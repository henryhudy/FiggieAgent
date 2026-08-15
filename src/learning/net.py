"""Policy/value network for the M4/P1/P2 search agent (DESIGN.md §6.2).

The 41-float observation (`learning.obs.encode`) flows through the trunk; the
trade-history window (`learning.obs.history_features`) flows through a parallel
history pathway (an MLP in P1, single-head self-attention in P2) whose output is
zero-initialized and added as a *residual* to the trunk output. This makes a
warm start from an older checkpoint exact (history contributes nothing until it
learns), and the policy/value heads see exactly the M4 representation at init.

The value head predicts a per-player expected round payout (self first, then
opponents in player order) ; the auxiliary "opponent value head". Its final
layer is also zero-initialized, so search starts policy-prior-driven and the
value becomes informative only as it learns. `evaluate(obs, history)` is the
no-grad search entry point; `history` may be omitted (zeros) for compatibility.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..env.deck import ALL_ASSIGNMENTS
from .obs import HIST_DIM, HIST_FEATS, N_ACTIONS, N_HISTORY

N_IN = 53
VALUE_SCALE = 50.0
N_ASSIGNMENTS = len(ALL_ASSIGNMENTS)


class _HistoryMLP(nn.Module):
    """P1: flat MLP over the flattened history window; zero-init output."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIST_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(hist))


class _HistoryAttention(nn.Module):
    """P2: single-head self-attention over the last N_HISTORY trade tokens."""

    def __init__(self, hist_feats: int = HIST_FEATS, n_history: int = N_HISTORY, d: int = 32):
        super().__init__()
        self.n_history = n_history
        self.d = d
        self.proj = nn.Linear(hist_feats, d)
        self.out = nn.Linear(d, 128)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        x = hist.view(-1, self.n_history, self.proj.in_features)
        t = self.proj(x)  # (B, N_HISTORY, d)
        qk = torch.bmm(t, t.transpose(1, 2)) / math.sqrt(self.d)
        attn = torch.softmax(qk, dim=-1)
        pooled = torch.bmm(attn, t).mean(dim=1)  # (B, d)
        return self.out(pooled)


class FiggieNet(nn.Module):
    def __init__(
        self,
        n_in: int = N_IN,
        n_actions: int = N_ACTIONS,
        hidden: int = 128,
        n_players: int = 4,
        history: str = "mlp",
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.history = _HistoryAttention() if history == "attn" else _HistoryMLP()
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_players),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
        self.belief_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, N_ASSIGNMENTS),
        )
        nn.init.zeros_(self.value_head[2].weight)
        nn.init.zeros_(self.value_head[2].bias)
        nn.init.zeros_(self.belief_head[2].weight)
        nn.init.zeros_(self.belief_head[2].bias)

    def forward(self, x: torch.Tensor, hist: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hist is None:
            hist = torch.zeros(x.shape[0], HIST_DIM, device=x.device)
        h = self.trunk(x)
        hh = self.history(hist)
        c = torch.relu(h + hh)  # residual: at init hh == 0, so c == trunk output
        p = self.policy_head(c)
        v = self.value_head(c)
        b = self.belief_head(c)
        return p, v, b

    def evaluate(self, obs: torch.Tensor, hist: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eval-mode, no-grad forward for search: returns (logits, values, belief)."""
        with torch.no_grad():
            p, v, b = self.forward(obs, hist)
        return p, v, b


def load_partial(net: FiggieNet, path: str) -> FiggieNet:
    """Warm-start: load checkpoint weights only where shapes match the new net.

    Lets a P1/P2 net (new history pathway, multi-player value head) start from
    an older checkpoint whose trunk and policy head transfer directly; with the
    residual design the transfer is exact (history contributes zero at init).
    The first linear layer also transfers when the input dimension has grown
    (the bundle-obs extension): the old input slice is copied and the new
    columns are zero-initialized, preserving the trained trunk.
    """
    sd = torch.load(path, map_location="cpu")["model"]
    own = net.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k not in own:
            continue
        ov = own[k]
        if ov.shape == v.shape:
            ov.copy_(v)
            loaded += 1
        elif k == "trunk.0.weight" and v.dim() == 2 and v.shape[0] == ov.shape[0] and v.shape[1] < ov.shape[1]:
            ov[:, : v.shape[1]].copy_(v)
            ov[:, v.shape[1] :].zero_()
            loaded += 1
        elif k == "policy_head.2.weight" and v.dim() == 2 and v.shape[0] < ov.shape[0] and v.shape[1] == ov.shape[1]:
            ov[: v.shape[0], :].copy_(v)
            ov[v.shape[0] :, :].zero_()
            loaded += 1
        elif k == "policy_head.2.bias" and v.shape[0] < ov.shape[0]:
            ov[: v.shape[0]].copy_(v)
            ov[v.shape[0] :].zero_()
            loaded += 1
    net.load_state_dict(own)
    print(f"load_partial: loaded {loaded}/{len(sd)} keys from {path}")
    return net
