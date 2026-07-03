# Phase 12 — Stage B.1 report: length-generalisation

| Field | Value |
|---|---|
| Stage | B.1 (train small W, test unseen large W; pre-reg `STAGE_B1_PREREG.md`, committed before the run, G1) |
| Verdict | ✅ **PASS** — the strongest, most FM-relevant result of the pilot. 2D-local generalises small→large; 1D-local is at chance on every unseen size. |
| Artifacts | `experiments/phase12_2d_compute_medium/lengthgen.py`; `lengthgen.json`; 3 seeds; runtime 104 s |

## Result — train on W ∈ {4,5,6}, test on unseen W

| arm | 7 | 8 | 9 | 10 | 12 |
|---|---|---|---|---|---|
| **2D-local** | **0.93** | **0.83** | **0.78** | **0.76** | **0.74** |
| **1D-local (row-major)** | 0.55 | 0.49 | 0.50 | 0.49 | 0.51 |
| **1D-local (Hilbert)** | 0.63 | 0.63 | 0.58 | 0.59 | 0.57 |
| **GNN (positive control)** | 0.94 | 0.84 | 0.72 | 0.61 | 0.51 |

**Pre-registered gate:** mean gap(2D−1D) over W≤10 = **+0.320** (≥ 0.10 ✓);
2D ≥ 1D at every W ✓; GNN control @ W=8 = **0.84** (≥ 0.70 ✓). → **PASS.**

## What it shows

- **2D-local generalises across scale.** Trained only on W ≤ 6, it holds
  0.74–0.93 on grid widths it never saw (7–12). A weight-shared 2D-local operator
  learns a *scale-equivariant* propagation rule — the same 4-neighbour computation
  applies at any width — so the learned reasoning transfers to larger problems.
- **1D-local does not generalise — at all.** Row-major 1D sits at **exact chance**
  on every unseen W (0.49–0.55). It did not learn a transferable rule: the
  row-major serialisation is **not scale-invariant** (vertical neighbours are W
  apart), so a kernel calibrated to gap-6 at train is meaningless at gap-8/10/12.
  Hilbert is marginally better but also essentially fails (0.57–0.63) — its order
  is W-dependent too. **This is the decisive contrast:** the 2D advantage is not
  just a fixed-W accuracy edge (A.0.5) but a *generalisation* property 1D lacks.
- **The task IS length-generalisable (control holds).** The GNN — scale-invariant
  on the graph — generalises strongly at W=7,8 (0.94, 0.84), confirming the 1D
  failure is the serialisation, not that nothing transfers. Both GNN and 2D are
  depth-bounded and fade at W≥10; notably **2D-local degrades more gracefully than
  the GNN** (W=12: 0.74 vs 0.51), i.e. the 2D conv generalises *better* than a GNN
  handed the true adjacency, at the largest sizes.

## Why this matters most

Length-generalisation is the core of the "2D as a computational medium" thesis
and the FM-relevant axis (the [[foundation_model_aspiration]]): a model whose
*layout carries the computation* should extend to larger instances at fixed
per-cell compute. A.0.5/A.0.6 showed 2D beats matched 1D at *fixed* W (bandwidth);
B.1 shows the learned 2D computation **transfers to unseen larger problems where
the 1D serialisation is at chance**. That is the qualitative behaviour the FM
aspiration needs to see, now demonstrated at pilot scale.

## Honest caveats

- **Depth-bounded, not unbounded.** Both scale-invariant arms (2D, GNN) fade at
  W≥10 because path length exceeds the fixed 16-layer propagation budget. The
  headline is 2D-vs-1D *retention* on unseen W, and 2D's graceful degradation —
  not unbounded generalisation. Adaptive-depth / iterate-to-fixed-point is a
  natural follow-up.
- **Pilot scale, hand-engineered substrate, single family (F1).** Structure-
  exploitable, not emergent (Goal 1 stays closed). Not yet an FM claim — a
  *positive signal* toward the axis the FM line would need.
- Single depth/hidden; multi-scale curriculum and a capable-global length-gen
  comparison remain open.

## Standing after B.1

Phase 12 now has two converging positives: **(A.0.5/A.0.6)** 2D-local beats the
best matched 1D layout at fixed W (bandwidth-attributed, Hilbert-robust,
F3-controlled), and **(B.1)** the learned 2D computation length-generalises to
unseen larger problems where 1D is at chance. Open: the capable-iterative-global
comparison (A.0.7 inconclusive), F2 depth-generalisation, statistical hardening
(Holm), and the engineering-application sketch.
