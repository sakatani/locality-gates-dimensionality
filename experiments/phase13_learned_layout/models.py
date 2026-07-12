"""Phase 13 — the four arms for the LGRC cheap gate.

layout-2D    : encode → predict per-cell position → **Sinkhorn** soft-assign to a
               W×W lattice → local CA → soft-gather S/T → MLP.  (the thesis arm)
set-transformer : full self-attention over the set (Phase-11 strong baseline).
kNN-GNN      : recurrent GIN-E on a k-NN graph built from (u,v).
random-layout: random permutation placement + CA (control — isolates *learned*
               placement from *any* lattice placement).

All arms consume the same (B, N, 5) set features; differ only in how they use
structure. Batches are per-W (uniform N = W²).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .task import N_FEAT


def sinkhorn(log_alpha: torch.Tensor, n_iter: int = 12) -> torch.Tensor:
    """(B,N,N) scores → approx doubly-stochastic soft permutation (log-domain)."""
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return log_alpha.exp()


def _lattice_coords(W: int, device) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(W, device=device),
                            torch.arange(W, device=device), indexing="ij")
    xy = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1).float()
    return xy / max(W - 1, 1) - 0.5                       # (W*W, 2) in [-0.5,0.5]


class _CA(nn.Module):
    """Stabilised local 2D update (Phase-12 style), iterated ``iters`` times."""

    def __init__(self, hidden: int, groups: int = 4):
        super().__init__()
        self.msg = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.upd = nn.Conv2d(2 * hidden, hidden, 1)
        self.norm = nn.GroupNorm(groups, hidden)

    def forward(self, h: torch.Tensor, iters: int) -> torch.Tensor:
        for _ in range(iters):
            m = self.msg(h)
            u = torch.tanh(self.upd(torch.cat([h, m], dim=1)))
            h = self.norm(h + 0.5 * u)
        return h


def _gather_rows(P: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """P: (B,N,N); idx: (B,) → (B,N) the idx-th row per batch."""
    return P[torch.arange(P.shape[0], device=P.device), idx]


class LayoutCA(nn.Module):
    def __init__(self, hidden: int = 32, temp: float = 0.05, ca_iters_per_w: int = 2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.pos_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                      nn.Linear(hidden, 2))
        self.ca = _CA(hidden)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))
        self.temp = temp
        self.ca_iters_per_w = ca_iters_per_w
        self.hidden = hidden

    def forward(self, X, s_idx, t_idx, W, learned: bool = True,
                rand_perm: torch.Tensor | None = None):
        B, N, _ = X.shape
        enc = self.enc(X)                                     # (B,N,d)
        if learned:
            pos = self.pos_head(enc)                          # (B,N,2)
            lat = _lattice_coords(W, X.device)                # (N,2)
            d2 = ((pos[:, :, None, :] - lat[None, None]) ** 2).sum(-1)  # (B,N,N)
            P = sinkhorn(-d2 / self.temp)                     # (B,N,N)
        else:                                                 # random-layout control
            P = torch.zeros(B, N, N, device=X.device)
            P[torch.arange(B)[:, None], torch.arange(N)[None], rand_perm] = 1.0
        lattice = torch.bmm(P.transpose(1, 2), enc)           # (B,N,d) on lattice
        h = lattice.transpose(1, 2).reshape(B, self.hidden, W, W)
        h = self.ca(h, iters=self.ca_iters_per_w * W)
        emb = h.reshape(B, self.hidden, N).transpose(1, 2)    # (B,N,d)
        emb_s = torch.bmm(_gather_rows(P, s_idx).unsqueeze(1), emb).squeeze(1)
        emb_t = torch.bmm(_gather_rows(P, t_idx).unsqueeze(1), emb).squeeze(1)
        return self.head(torch.cat([emb_s, emb_t], dim=-1)).squeeze(-1)


class SetTransformer(nn.Module):
    def __init__(self, hidden: int = 32, layers: int = 4, heads: int = 4):
        super().__init__()
        self.enc = nn.Linear(N_FEAT, hidden)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True,
                                       dropout=0.0, norm_first=True)
            for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, X, s_idx, t_idx, W):
        h = self.enc(X)
        for layer in self.layers:
            h = layer(h)
        B = X.shape[0]
        ar = torch.arange(B, device=X.device)
        return self.head(torch.cat([h[ar, s_idx], h[ar, t_idx]], dim=-1)).squeeze(-1)


class KNNGNN(nn.Module):
    """Recurrent GIN-E (skip-to-input + edge MLP; Grötschla-style for size-gen)
    on a k-NN graph built from (u,v)."""

    def __init__(self, hidden: int = 32, k: int = 4, iters_per_w: int = 2):
        super().__init__()
        self.enc = nn.Linear(N_FEAT, hidden)
        self.skip = nn.Linear(N_FEAT, hidden)
        self.edge = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden))
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))
        self.k = k
        self.iters_per_w = iters_per_w

    def _knn(self, uv: torch.Tensor) -> torch.Tensor:
        d2 = (uv[:, :, None, :] - uv[:, None, :, :]).pow(2).sum(-1)  # (B,N,N)
        d2 = d2 + torch.eye(uv.shape[1], device=uv.device)[None] * 1e9
        return d2.topk(self.k, dim=-1, largest=False).indices        # (B,N,k)

    def forward(self, X, s_idx, t_idx, W):
        B, N, _ = X.shape
        nbr = self._knn(X[:, :, 0:2])                                # (B,N,k)
        skip = self.skip(X)
        h = self.enc(X)
        bi = torch.arange(B, device=X.device)[:, None, None]
        for _ in range(self.iters_per_w * W):
            hn = h[bi, nbr]                                          # (B,N,k,d)
            msg = self.edge(torch.cat([h[:, :, None, :].expand_as(hn), hn], dim=-1))
            agg = msg.mean(dim=2)                                    # (B,N,d)
            h = h + self.upd(torch.cat([h, agg], dim=-1)) + skip
            h = F.layer_norm(h, (h.shape[-1],))
        ar = torch.arange(B, device=X.device)
        return self.head(torch.cat([h[ar, s_idx], h[ar, t_idx]], dim=-1)).squeeze(-1)
