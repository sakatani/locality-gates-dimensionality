# Phase 12 — Stage B.3.1 report: F4 fixed-K redesign → suite 3/3

| Field | Value |
|---|---|
| Stage | B.3.1 (F4 scale-invariant redesign; pre-reg `STAGE_B31_PREREG.md`, committed before the run, G1) |
| Verdict | ✅ **F4b PASS** — the metric family is now gate-clean; the reasoning suite reaches **3/3** (connectivity + metric + dynamics). |
| Artifacts | `experiments/phase12_2d_compute_medium/{stageb3_1.py, families.build_f4b}`; `stageb3_1.json`; 3 seeds; runtime 87 s |

## Result — F4b (bounded-radius R=8, FIXED K=6), train W ∈ {4,5,6}, test unseen W

| arm | 8 | 10 | 12 | 16 | 20 |
|---|---|---|---|---|---|
| **2d-local** | 0.97 | 0.96 | 0.96 | 0.94 | 0.93 |
| **1d-local** | 0.68 | 0.62 | 0.62 | 0.58 | 0.57 |
| **gnn (control)** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**Gate:** mean gap(2D−1D) over W≤12 = **+0.32** ≥ 0.10 ✓; 2D ≥ 1D at every W ✓;
GNN@8 = **1.00** ≥ 0.70 ✓ → **PASS**.

## What the fix confirms

- **The diagnosis was correct.** F4's GNN control failed (0.56) purely because its
  threshold `K = 2(W−1)` scaled with W (not scale-invariant). With a **fixed K=6**
  and bounded-radius placement, the identical task becomes cleanly
  length-generalizable and the GNN control jumps **0.56 → 1.00**. The 2D-vs-1D
  signal is if anything stronger (gap +0.32 vs +0.22).
- **2D-local is nearly flat** (0.97→0.93 across W=8–20) — the bounded-depth
  pattern again (K=6 needs ~6 propagation steps, not W-scaling), matching F5 and
  reinforcing the B.3 clarifying finding: *bounded-intrinsic-depth tasks
  length-generalize flatly; only depth-scaling tasks (reachability) fade.*
- 1D-local sits at 0.57–0.68 (above chance — it extracts some coordinate-heuristic
  signal — but far below 2D's 0.93+), so 2D ≥ 1D holds decisively everywhere.

## Consolidated reasoning suite (B.3 + B.3.1)

| family | computation | 2D length-gen | gate |
|---|---|---|---|
| **F1** reachability | connectivity (depth ~W) | 0.82→0.73 (fades) | ✅ PASS |
| **F4b** bounded distance | metric (depth ~K=6) | 0.97→0.93 (flat) | ✅ PASS |
| **F5** CA-rollout | local dynamics (depth T=2) | 0.92→0.88 (flat) | ✅ PASS |

> **Suite = 3/3.** 2D-local length-generalization is a **general** property across
> connectivity, metric, and dynamics reasoning — **not** reachability-specific. In
> every family 1D-local fails to transfer (chance-to-weak) while a scale-invariant
> GNN control confirms each task is genuinely length-generalizable.

Honest note: F4 required a design fix (F4b) to be gate-clean; the original's
control confound (W-scaling threshold) is documented in `STAGE_B3_REPORT.md`. This
is a corrected experiment, pre-registered (B.3.1) before the rerun — not a
post-hoc gate change.

## Standing

Direction A (broaden the base) is **complete**: Phase 12's computational-medium
result now spans a **3-family reasoning suite** (connectivity + metric + dynamics)
plus the fixed-W bandwidth result (A.0.5/A.0.6). The base is materially stronger
for the flagship **learned-layout** bet (the deferred novelty). Caveats unchanged:
pilot scale, grid families (layout free), hand-engineered/supervised →
structure-exploitable, Goal 1 stays closed.
