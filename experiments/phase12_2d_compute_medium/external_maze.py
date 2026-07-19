"""E3a (second-review response): OUT-OF-SAMPLE check of the decision procedure
on an EXTERNAL benchmark — Schwarzschild et al.'s easy-to-hard maze dataset
(NeurIPS 2021; pip `easy-to-hard-data`). Task: given a maze image (3×H×W),
predict the optimal-path mask (per-pixel binary); their standard metric is
full-maze accuracy (every pixel correct).

Decision-procedure prediction (made BEFORE running, from §8): the task is
native-2D (step 3 yes), low-2D-bandwidth/high-1D-bandwidth (step 4 yes:
wall-following is neighbor propagation), and its required depth scales with
maze size (step 5: scaling) ⇒ 2D-local should learn it and transfer to larger
unseen mazes with graceful fading; the matched 1D serialization should fail
outright (near-zero full-maze accuracy) at every size.

Arms: dense per-pixel variants of the §3.2 local arms — identical parameter
count and window (H2): Conv2d(h,h,3×3) ≡ Conv1d(h,h,9) on the row-major
serialization. Train on their 9×9 training set; test on their held-out 9×9,
11×11, and 13×13 test sets (larger mazes never seen in training).
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

from easy_to_hard_data import MazeDataset

QUICK = os.environ.get("PHASE12_QUICK", "0") == "1"
BIG = os.environ.get("PHASE12_MAZE_BIG", "0") == "1"   # fair-shot scale-up (exploratory)
ROOT = str(Path(__file__).resolve().parents[2] / ".maze_data")
TRAIN_SIZE = 9
TEST_SIZES = [9, 11] if QUICK else [9, 11, 13]
SEEDS = [0] if (QUICK or BIG) else [0, 1, 2]
HIDDEN, DEPTH = (64, 48) if BIG else (32, 24)
N_TRAIN = 20000 if BIG else (1000 if QUICK else 8000)
N_TEST = 300 if QUICK else 1000
EPOCHS = 50 if BIG else (4 if QUICK else 30)
BS = 64


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Dense2D(nn.Module):
    """§3.2 2D-local arm with a per-pixel head (3×3 convs, depth L)."""

    def __init__(self, hidden: int = 32, layers: int = 16):
        super().__init__()
        self.inp = nn.Conv2d(3, hidden, 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(hidden, hidden, 3, padding=1) for _ in range(layers)])
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):                                   # (B,3,H,W)
        h = F.relu(self.inp(x))
        for c in self.convs:
            h = h + F.relu(c(h))
        return self.head(h).squeeze(1)                      # (B,H,W) logits


class Dense1D(nn.Module):
    """Matched 1D-local arm: kernel-9 convs on the row-major serialization,
    per-position head. Identical params/window per layer as Dense2D (H2)."""

    def __init__(self, hidden: int = 32, layers: int = 16, kernel: int = 9):
        super().__init__()
        pad = kernel // 2
        self.inp = nn.Conv1d(3, hidden, 1)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, kernel, padding=pad) for _ in range(layers)])
        self.head = nn.Conv1d(hidden, 1, 1)

    def forward(self, x):                                   # (B,3,H,W)
        B, C, H, W = x.shape
        h = F.relu(self.inp(x.reshape(B, C, H * W)))
        for c in self.convs:
            h = h + F.relu(c(h))
        return self.head(h).squeeze(1).reshape(B, H, W)     # (B,H,W) logits


def _tensors(train: bool, size: int, n: int, seed: int):
    ds = MazeDataset(ROOT, train=train, size=size, download=False)
    g = np.random.default_rng(seed)
    idx = g.permutation(len(ds))[:n]
    X = ds.inputs[idx].float()
    Y = ds.targets[idx].float()
    return X, Y


def run_arm(arm: str, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    model = {"2d": Dense2D, "1d": Dense1D}[arm](HIDDEN, DEPTH).to(dev)
    Xtr, Ytr = _tensors(True, TRAIN_SIZE, N_TRAIN, seed)
    Xtr, Ytr = Xtr.to(dev), Ytr.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3 if BIG else 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) if BIG else None
    lossf = nn.BCEWithLogitsLoss()
    n = Xtr.shape[0]
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, BS):
            b = perm[i:i + BS]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), Ytr[b])
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for size in TEST_SIZES:
            Xte, Yte = _tensors(False, size, N_TEST, seed)
            solves, pix = [], []
            for i in range(0, Xte.shape[0], BS):
                xb, yb = Xte[i:i + BS].to(dev), Yte[i:i + BS].to(dev)
                pred = (model(xb) > 0).float()
                solves.append((pred == yb).flatten(1).all(dim=1).float())
                pix.append((pred == yb).float().mean(dim=(1, 2)))
            out[size] = {"solve": float(torch.cat(solves).mean()),
                         "pixel": float(torch.cat(pix).mean())}
    return out


def main() -> None:
    t0 = time.time()
    print(f"E3a external mazes  train={TRAIN_SIZE} test={TEST_SIZES} "
          f"seeds={SEEDS} n_train={N_TRAIN} epochs={EPOCHS}")
    results: dict = {}
    for arm in ("2d", "1d"):
        per_seed = [run_arm(arm, s) for s in SEEDS]
        results[arm] = {
            str(size): {"solve": [r[size]["solve"] for r in per_seed],
                        "pixel": [r[size]["pixel"] for r in per_seed]}
            for size in TEST_SIZES}
        pretty = " ".join(
            f"{size}:solve={np.mean(results[arm][str(size)]['solve']):.2f}"
            f"/pix={np.mean(results[arm][str(size)]['pixel']):.2f}"
            for size in TEST_SIZES)
        print(f"  [{arm}] {pretty}")
    out = {"config": {"train_size": TRAIN_SIZE, "test_sizes": TEST_SIZES,
                      "seeds": SEEDS, "hidden": HIDDEN, "depth": DEPTH,
                      "n_train": N_TRAIN, "n_test": N_TEST, "epochs": EPOCHS,
                      "quick": QUICK, "big": BIG},
           "results": results, "runtime_sec": round(time.time() - t0, 1)}
    rep = Path(__file__).resolve().parents[2] / "runs" / "phase12_2d_compute_medium"
    rep.mkdir(parents=True, exist_ok=True)
    outname = "external_maze_big.json" if BIG else "external_maze.json"
    with open(rep / outname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"runtime={out['runtime_sec']}s\nwrote {rep / outname}")


if __name__ == "__main__":
    main()
