"""Phase 12 — model arms (PLAN §3).

Four arms consume the SAME (C, H, W) feature map (substrate H1) and read out the
SAME cell; they differ ONLY in the locality structure of the operator:

- ``Local2D``  conv over a 3×3 (= 9-cell) window in (x, y).
- ``Local1D``  conv over a 9-cell window on the row-major serialisation.
              kernel-9 conv1d has the SAME params + window-cell-count as 3×3
              conv2d → the comparison isolates the *dimensionality of the
              layout* (PLAN H2), nothing else.
- ``GlobalAttn`` full self-attention (Phase 11's regime; O(n²)).
- ``GNNAdj``   message passing on the true 4-adjacency (open cells only).

Matched depth + width across arms; readout = linear(gathered cell embedding).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .families import N_CHANNELS


def _gather_cells(feat_bnc: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
    """feat_bnc (B, N, C), idx_b (B,) → (B, C) the readout cell embedding."""
    b = torch.arange(feat_bnc.shape[0], device=feat_bnc.device)
    return feat_bnc[b, idx_b]


class Local2D(nn.Module):
    """Residual 3×3 conv stack on the grid."""

    def __init__(self, hidden: int = 32, layers: int = 10):
        super().__init__()
        self.inp = nn.Conv2d(N_CHANNELS, hidden, 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(hidden, hidden, 3, padding=1) for _ in range(layers)]
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x_bchw: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.inp(x_bchw))
        for conv in self.convs:
            h = h + F.relu(conv(h))
        B, C, H, W = h.shape
        h = h.reshape(B, C, H * W).transpose(1, 2)  # (B, N, C)
        return self.head(_gather_cells(h, idx_b)).squeeze(-1)


class Local1D(nn.Module):
    """Residual kernel-9 conv stack on the row-major serialisation."""

    def __init__(self, hidden: int = 32, layers: int = 10, kernel: int = 9):
        super().__init__()
        self.inp = nn.Conv1d(N_CHANNELS, hidden, 1)
        pad = kernel // 2
        self.convs = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, kernel, padding=pad) for _ in range(layers)]
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x_bchw: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_bchw.shape
        h = x_bchw.reshape(B, C, H * W)               # row-major sequence
        h = F.relu(self.inp(h))
        for conv in self.convs:
            h = h + F.relu(conv(h))
        h = h.transpose(1, 2)                          # (B, N, C)
        return self.head(_gather_cells(h, idx_b)).squeeze(-1)


class Local1DCurve(nn.Module):
    """kernel-9 conv stack on a SPACE-FILLING-CURVE serialisation (A.0.6 Hilbert
    control). Identical operator/params to ``Local1D``; only the linearisation
    order differs (the locality-preserving one). ``order`` (B-independent, (N,))
    maps curve-position → row-major index; ``inv_order`` is its inverse.
    """

    def __init__(self, hidden: int = 32, layers: int = 10, kernel: int = 9):
        super().__init__()
        self.inp = nn.Conv1d(N_CHANNELS, hidden, 1)
        pad = kernel // 2
        self.convs = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, kernel, padding=pad) for _ in range(layers)]
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x_bchw, idx_b, order, inv_order) -> torch.Tensor:
        B, C, H, W = x_bchw.shape
        seq = x_bchw.reshape(B, C, H * W)[:, :, order]   # reorder to curve
        h = F.relu(self.inp(seq))
        for conv in self.convs:
            h = h + F.relu(conv(h))
        h = h.transpose(1, 2)                            # (B, N, C) in curve order
        return self.head(_gather_cells(h, inv_order[idx_b])).squeeze(-1)


class GlobalLooped(nn.Module):
    """Capable *iterative* global baseline (A.0.6): ONE weight-shared Transformer
    encoder layer applied ``iters`` times → full attention that can propagate
    over many steps (unlike the shallow ``GlobalAttn``). Tests whether the
    advantage is '2D' vs merely 'iteration'.
    """

    def __init__(self, hidden: int = 32, iters: int = 16, heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(N_CHANNELS, hidden)
        self.layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2,
                                                batch_first=True, dropout=0.0)
        self.iters = iters
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x_bchw: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_bchw.shape
        h = self.embed(x_bchw.reshape(B, C, H * W).transpose(1, 2))
        for _ in range(self.iters):
            h = self.layer(h)
        return self.head(_gather_cells(h, idx_b)).squeeze(-1)


class ConvCA2D(nn.Module):
    """Adaptive-depth 2D-local operator (Stage B.2): ONE weight-shared local
    conv update iterated ``iters`` times (a stabilised neural-CA). Because the
    update is shared across steps, a rule learned on small grids with few steps
    can be run for MORE steps at test to propagate across larger grids — breaking
    the fixed-depth propagation bound. Gated residual + GroupNorm keep it stable
    over variable iteration counts.
    """

    def __init__(self, hidden: int = 32, groups: int = 4):
        super().__init__()
        self.embed = nn.Conv2d(N_CHANNELS, hidden, 1)
        self.msg = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.upd = nn.Conv2d(2 * hidden, hidden, 1)
        self.norm = nn.GroupNorm(groups, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x_bchw: torch.Tensor, idx_b: torch.Tensor,
                iters: int) -> torch.Tensor:
        h = F.relu(self.embed(x_bchw))
        for _ in range(iters):
            m = self.msg(h)
            u = torch.tanh(self.upd(torch.cat([h, m], dim=1)))
            h = self.norm(h + 0.5 * u)
        B, C, H, W = h.shape
        hf = h.reshape(B, C, H * W).transpose(1, 2)
        return self.head(_gather_cells(hf, idx_b)).squeeze(-1)


class ConvCA1D(nn.Module):
    """Adaptive-depth 1D control: the same iterated gated update on the row-major
    serialisation (kernel-9 = 9-cell window, matched to ConvCA2D's 3×3). Tests
    whether iterating longer rescues the wrong (W-dependent) linearisation.
    """

    def __init__(self, hidden: int = 32, groups: int = 4, kernel: int = 9):
        super().__init__()
        pad = kernel // 2
        self.embed = nn.Conv1d(N_CHANNELS, hidden, 1)
        self.msg = nn.Conv1d(hidden, hidden, kernel, padding=pad)
        self.upd = nn.Conv1d(2 * hidden, hidden, 1)
        self.norm = nn.GroupNorm(groups, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x_bchw: torch.Tensor, idx_b: torch.Tensor,
                iters: int) -> torch.Tensor:
        B, C, H, W = x_bchw.shape
        h = F.relu(self.embed(x_bchw.reshape(B, C, H * W)))
        for _ in range(iters):
            m = self.msg(h)
            u = torch.tanh(self.upd(torch.cat([h, m], dim=1)))
            h = self.norm(h + 0.5 * u)
        h = h.transpose(1, 2)
        return self.head(_gather_cells(h, idx_b)).squeeze(-1)


class UniversalTransformer(nn.Module):
    """Capable iterative-global baseline (A.0.7) — a Universal Transformer done
    properly: ONE weight-shared encoder layer applied ``iters`` times with
    **pre-LN** (norm_first, stable for deep recurrence) + a **per-iteration step
    embedding** (lets each step do different work) + a final norm. This is the
    fair "is it 2D or just iteration?" test the naive ``GlobalLooped`` failed.
    """

    def __init__(self, hidden: int = 32, iters: int = 16, heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(N_CHANNELS, hidden)
        self.step_emb = nn.Parameter(torch.zeros(iters, hidden))
        self.layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 2, batch_first=True, dropout=0.0,
            norm_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.iters = iters
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x_bchw: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_bchw.shape
        h = self.embed(x_bchw.reshape(B, C, H * W).transpose(1, 2))
        for t in range(self.iters):
            h = self.layer(h + self.step_emb[t])
        h = self.norm(h)
        return self.head(_gather_cells(h, idx_b)).squeeze(-1)


class GlobalAttn(nn.Module):
    """Small Transformer encoder over all cells (full attention)."""

    def __init__(self, hidden: int = 32, layers: int = 4, heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(N_CHANNELS, hidden)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 2,
                                         batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x_bchw: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_bchw.shape
        tok = x_bchw.reshape(B, C, H * W).transpose(1, 2)  # (B, N, C)
        h = self.enc(self.embed(tok))
        return self.head(_gather_cells(h, idx_b)).squeeze(-1)


def build_neighbour_index(walls_bhw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(B, N, 4) neighbour cell indices + (B, N, 4) validity mask (open cells only)."""
    B, H, W = walls_bhw.shape
    N = H * W
    nbr = np.zeros((B, N, 4), dtype=np.int64)
    mask = np.zeros((B, N, 4), dtype=np.float32)
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for b in range(B):
        for r in range(H):
            for c in range(W):
                u = r * W + c
                if walls_bhw[b, r, c] == 1:
                    continue
                for k, (dr, dc) in enumerate(offs):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and walls_bhw[b, nr, nc] == 0:
                        nbr[b, u, k] = nr * W + nc
                        mask[b, u, k] = 1.0
    return nbr, mask


class RecGNN(nn.Module):
    """Stabilized recurrent GNN in the style of Grötschla et al. 2022
    (RecGRU-E-lite): ONE weight-shared round applied ``rounds`` times, with
    (i) a skip connection to the input features at every round, (ii) an edge
    MLP on (h_v, h_w) pairs, (iii) a GRU state update, and (iv) an L2 state
    regularizer returned as an auxiliary loss. ``rounds`` is set at call time,
    so inference can run more rounds on larger graphs — the paper §5.2 arm."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.embed = nn.Linear(N_CHANNELS, hidden)
        self.skip = nn.Linear(N_CHANNELS, hidden)
        self.edge = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden))
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))
        self.hidden = hidden

    def forward(self, x_bchw, idx_b, nbr_bn4, mask_bn4, rounds: int):
        B, C, H, W = x_bchw.shape
        N = H * W
        tok = x_bchw.reshape(B, C, N).transpose(1, 2)             # (B, N, C)
        s = self.skip(tok)
        h = self.embed(tok)
        deg = mask_bn4.sum(-1, keepdim=True).clamp(min=1.0)
        l2 = h.new_zeros(())
        for _ in range(rounds):
            z = h + s                                             # input skip
            gathered = torch.gather(
                z.unsqueeze(2).expand(B, N, 4, self.hidden), 1,
                nbr_bn4.unsqueeze(-1).expand(B, N, 4, self.hidden),
            )
            pair = torch.cat(
                [z.unsqueeze(2).expand(B, N, 4, self.hidden), gathered], dim=-1)
            msg = (self.edge(pair) * mask_bn4.unsqueeze(-1)).sum(2) / deg
            h = self.gru(msg.reshape(B * N, self.hidden),
                         h.reshape(B * N, self.hidden)).reshape(B, N, self.hidden)
            l2 = l2 + h.pow(2).mean()
        logits = self.head(_gather_cells(h, idx_b)).squeeze(-1)
        return logits, l2 / max(rounds, 1)


class GNNAdj(nn.Module):
    """Mean-aggregation message passing over the true 4-adjacency (open cells)."""

    def __init__(self, hidden: int = 32, layers: int = 10):
        super().__init__()
        self.embed = nn.Linear(N_CHANNELS, hidden)
        self.self_w = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.nbr_w = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x_bchw, idx_b, nbr_bn4, mask_bn4) -> torch.Tensor:
        B, C, H, W = x_bchw.shape
        N = H * W
        h = self.embed(x_bchw.reshape(B, C, N).transpose(1, 2))   # (B, N, hidden)
        deg = mask_bn4.sum(-1, keepdim=True).clamp(min=1.0)       # (B, N, 1)
        for sw, nw in zip(self.self_w, self.nbr_w):
            gathered = torch.gather(
                h.unsqueeze(2).expand(B, N, 4, h.shape[-1]), 1,
                nbr_bn4.unsqueeze(-1).expand(B, N, 4, h.shape[-1]),
            )
            agg = (gathered * mask_bn4.unsqueeze(-1)).sum(2) / deg
            h = h + F.relu(sw(h) + nw(agg))
        return self.head(_gather_cells(h, idx_b)).squeeze(-1)
