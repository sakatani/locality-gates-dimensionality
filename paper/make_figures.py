"""Generate the manuscript figures from harden_stats.json (5-seed data).

Fig 1  fixed-size threshold sweep (Mechanism I): F1 accuracy vs W for
       2D-local / 1D row-major / 1D space-filling, with tau=0.80 line and
       per-seed spread.
Fig 2  length-generalization suite (Mechanism II): one panel per family
       (F1, F4b, F5), accuracy vs unseen test W for 2D-local / 1D-local / GNN.

Usage: <venv-python> make_figures.py  (writes fig1_bandwidth.pdf,
fig2_lengthgen.pdf + .png previews alongside this script)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "runs" / "phase12_2d_compute_medium" / "harden_stats.json"

COLORS = {"2d": "#1a7f37", "1d": "#c4432b", "1d_hilbert": "#e08a00",
          "gnn": "#5a5a5a"}
LABELS = {"2d": "2D-local", "1d": "1D-local (row-major)",
          "1d_hilbert": "1D-local (space-filling)", "gnn": "GNN (control)"}


def _series(d: dict[str, list[float]]):
    ws = sorted(int(w) for w in d)
    acc = np.array([d[str(w)] for w in ws], dtype=float)  # (n_W, n_seeds)
    return ws, acc.mean(axis=1), acc.std(axis=1, ddof=1)


def fig1(stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for arm in ("2d", "1d", "1d_hilbert"):
        ws, mean, sd = _series(stats["fixed_f1"][arm])
        ax.errorbar(ws, mean, yerr=sd, color=COLORS[arm], label=LABELS[arm],
                    marker="o", ms=4, lw=1.6, capsize=2.5)
    ax.axhline(0.80, color="k", ls=":", lw=1, alpha=0.7)
    ax.annotate(r"$\tau = 0.80$", xy=(11.0, 0.805), fontsize=8, alpha=0.8)
    ax.annotate("", xy=(8, 0.545), xytext=(4, 0.545),
                arrowprops=dict(arrowstyle="<->", color=COLORS["2d"], lw=1))
    ax.annotate(r"$W^{*}$: 8 vs 4", xy=(4.6, 0.555), fontsize=8,
                color=COLORS["2d"])
    ax.set_xlabel("grid width $W$ (square grids)")
    ax.set_ylabel("accuracy (F1 reachability)")
    ax.set_xticks([4, 6, 8, 10, 12])
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig1_bandwidth.{ext}", dpi=200)
    plt.close(fig)


def fig2(stats: dict) -> None:
    fams = [("F1", "F1 reachability\n(depth $\\sim W$: fades)"),
            ("F4b", "F4b bounded metric\n(depth $\\sim K$: flat)"),
            ("F5", "F5 local dynamics\n(depth $\\sim T$: flat)")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
    for ax, (fam, title) in zip(axes, fams):
        for arm in ("2d", "1d", "gnn"):
            ws, mean, sd = _series(stats["suite"][fam][arm])
            ax.errorbar(ws, mean, yerr=sd, color=COLORS[arm], label=LABELS[arm],
                        marker="o", ms=4, lw=1.6, capsize=2.5)
        ax.axhline(0.5, color="k", ls=":", lw=1, alpha=0.5)
        ax.axvspan(3.5, 6.5, color="k", alpha=0.06, lw=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("test width $W$ (train $W \\leq 6$)")
        ax.set_xticks([8, 10, 12, 16, 20])
        ax.set_ylim(0.42, 1.03)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy")
    axes[0].annotate("chance", xy=(16.2, 0.512), fontsize=8, alpha=0.6)
    axes[2].legend(fontsize=8, frameon=False, loc="center right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig2_lengthgen.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    stats = json.loads(DATA.read_text())
    fig1(stats)
    fig2(stats)
    print(f"wrote fig1_bandwidth / fig2_lengthgen (.pdf/.png) in {HERE}")


if __name__ == "__main__":
    main()
