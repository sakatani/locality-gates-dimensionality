"""E3b (second-review response): OUT-OF-SAMPLE check of the decision
procedure's GIVEN-GRAPH branch (step 2) on the OFFICIAL CLRS-30 benchmark
(Veličković et al. 2022; pip `dm-clrs`, v2.0.3): BFS on Erdős–Rényi graphs,
predict the BFS predecessor pointer `pi` per node; metric = pointer accuracy
(the CLRS metric for pointer outputs). Train at the benchmark's standard
train size n=16; evaluate at n=16 (held-out), 32, and 64 (the benchmark's
canonical OOD size).

Decision-procedure prediction (made BEFORE running, from §8): the dependency
structure is GIVEN as an edge list (step 2: yes) ⇒ use a stabilized recurrent
GNN; no spatial layout is involved or needed. Concretely: the §5.2-style
RecGNN should reach high pointer accuracy at n=16 AND retain it at 4× nodes,
while a plain fixed-depth MPNN (no stabilizers) should degrade — mirroring
in-house §5/§5.2 on external data.

Arms (graph versions, dense adjacency):
- recgnn: weight-shared round, input-skip + edge-MLP + GRU + L2 state reg;
  variable training rounds U[4,12], 12 rounds at inference.
- plain:  per-layer weights, fixed 12 layers, no skip/GRU/L2.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import clrs

QUICK = os.environ.get("PHASE12_QUICK", "0") == "1"
TRAIN_N = 16
TEST_NS = [16, 32] if QUICK else [16, 32, 64]
SEEDS = [0] if QUICK else [0, 1, 2]
HIDDEN = 64
N_TRAIN = 400 if QUICK else 2000
N_TEST = 200 if QUICK else 500
EPOCHS = 4 if QUICK else 25
BS = 32
L2_COEF = 1e-2
TRAIN_ROUNDS = (4, 12)
TEST_ROUNDS = 12


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def sample_bfs(n_samples: int, length: int, seed: int):
    """Official CLRS BFS instances → tensors (feat (B,N,2), adj (B,N,N), pi (B,N))."""
    sampler, _ = clrs.build_sampler("bfs", num_samples=n_samples, length=length,
                                    seed=seed)
    fb = sampler.next(n_samples)
    inp = {dp.name: np.asarray(dp.data) for dp in fb.features.inputs}
    pi = np.asarray(fb.outputs[0].data)
    feat = np.stack([inp["pos"], inp["s"]], axis=-1).astype(np.float32)
    adj = inp["adj"].astype(np.float32)
    return (torch.from_numpy(feat), torch.from_numpy(adj),
            torch.from_numpy(pi).long())


class _PtrHead(nn.Module):
    """Pointer scores: score(v -> u) from (h_v, h_u), masked to adj ∪ self."""

    def __init__(self, hidden: int):
        super().__init__()
        self.src = nn.Linear(hidden, hidden)
        self.dst = nn.Linear(hidden, hidden)

    def forward(self, h, adj):
        logits = torch.einsum("bvd,bud->bvu", self.src(h), self.dst(h))
        eye = torch.eye(adj.shape[1], device=adj.device)[None]
        mask = ((adj + eye) > 0).float()
        return logits.masked_fill(mask == 0, -1e9)


class RecGNNPtr(nn.Module):
    """Stabilized recurrent GNN (Grötschla-style, §5.2) for pointer output."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.embed = nn.Linear(2, hidden)
        self.skip = nn.Linear(2, hidden)
        self.edge = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden))
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = _PtrHead(hidden)
        self.hidden = hidden

    def forward(self, feat, adj, rounds: int):
        B, N, _ = feat.shape
        s = self.skip(feat)
        h = self.embed(feat)
        deg = adj.sum(-1, keepdim=True).clamp(min=1.0)
        l2 = h.new_zeros(())
        for _ in range(rounds):
            z = h + s
            pair = torch.cat([z.unsqueeze(2).expand(B, N, N, self.hidden),
                              z.unsqueeze(1).expand(B, N, N, self.hidden)], -1)
            msg = (self.edge(pair) * adj.unsqueeze(-1)).sum(2) / deg
            h = self.gru(msg.reshape(B * N, -1), h.reshape(B * N, -1)).reshape(B, N, -1)
            l2 = l2 + h.pow(2).mean()
        return self.head(h, adj), l2 / max(rounds, 1)


class PlainGNNPtr(nn.Module):
    """Plain fixed-depth MPNN (per-layer weights, no stabilizers) — contrast."""

    def __init__(self, hidden: int = 64, layers: int = 12):
        super().__init__()
        self.embed = nn.Linear(2, hidden)
        self.self_w = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.nbr_w = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.head = _PtrHead(hidden)

    def forward(self, feat, adj, rounds: int = 0):
        h = self.embed(feat)
        deg = adj.sum(-1, keepdim=True).clamp(min=1.0)
        for sw, nw in zip(self.self_w, self.nbr_w):
            agg = torch.bmm(adj, h) / deg
            h = h + F.relu(sw(h) + nw(agg))
        return self.head(h, adj), h.new_zeros(())


def run_arm(arm: str, seed: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    model = {"recgnn": RecGNNPtr, "plain": PlainGNNPtr}[arm](HIDDEN).to(dev)
    feat, adj, pi = sample_bfs(N_TRAIN, TRAIN_N, 1000 + seed)
    feat, adj, pi = feat.to(dev), adj.to(dev), pi.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = feat.shape[0]
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, BS):
            b = perm[i:i + BS]
            rounds = int(rng.integers(*TRAIN_ROUNDS)) if arm == "recgnn" else 0
            opt.zero_grad()
            logits, l2 = model(feat[b], adj[b], rounds)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   pi[b].reshape(-1)) + L2_COEF * l2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for tn in TEST_NS:
            f2, a2, p2 = sample_bfs(N_TEST, tn, 9000 + seed)
            accs = []
            for i in range(0, f2.shape[0], BS):
                fb, ab, pb = (f2[i:i + BS].to(dev), a2[i:i + BS].to(dev),
                              p2[i:i + BS].to(dev))
                logits, _ = model(fb, ab, TEST_ROUNDS)
                accs.append((logits.argmax(-1) == pb).float().mean(dim=1))
            out[tn] = float(torch.cat(accs).mean())
    return out


def main() -> None:
    t0 = time.time()
    print(f"E3b official CLRS BFS  train_n={TRAIN_N} test_ns={TEST_NS} "
          f"seeds={SEEDS} n_train={N_TRAIN} epochs={EPOCHS}")
    results: dict = {}
    for arm in ("recgnn", "plain"):
        per_seed = [run_arm(arm, s) for s in SEEDS]
        results[arm] = {str(tn): [r[tn] for r in per_seed] for tn in TEST_NS}
        pretty = " ".join(f"{tn}:{np.mean(results[arm][str(tn)]):.2f}"
                          for tn in TEST_NS)
        print(f"  [{arm}] {pretty}")
    out = {"config": {"train_n": TRAIN_N, "test_ns": TEST_NS, "seeds": SEEDS,
                      "hidden": HIDDEN, "n_train": N_TRAIN, "epochs": EPOCHS,
                      "clrs_version": getattr(clrs, "__version__", "2.0.3"),
                      "quick": QUICK},
           "results": results, "runtime_sec": round(time.time() - t0, 1)}
    rep = Path(__file__).resolve().parents[2] / "runs" / "phase12_2d_compute_medium"
    rep.mkdir(parents=True, exist_ok=True)
    with open(rep / "external_clrs.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"runtime={out['runtime_sec']}s\nwrote {rep / 'external_clrs.json'}")


if __name__ == "__main__":
    main()
