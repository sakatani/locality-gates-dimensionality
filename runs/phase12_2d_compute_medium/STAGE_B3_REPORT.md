# Phase 12 — Stage B.3 report: reasoning-task suite (PARTIAL, clarifying)

| Field | Value |
|---|---|
| Stage | B.3 (does 2D-local length-gen generalize across reasoning tasks? pre-reg `STAGE_B3_PREREG.md`, committed before the run, G1) |
| Verdict | 🟡 **PARTIAL** — F5 (dynamics) passes cleanly; F4 (metric) shows a strong 2D-vs-1D signal but fails its own GNN control (a W-scaling-threshold design wrinkle). Plus a clarifying finding on the depth bound. |
| Artifacts | `experiments/phase12_2d_compute_medium/{stageb3.py, families.build_f4/build_f5, substrate.bfs_distance/ca_step}`; `stageb3.json`; 3 seeds; runtime 325 s |

## Result — train W ∈ {4,5,6}, test unseen W, 3 seeds

| family | arm | 8 | 10 | 12 | 16 | 20 | gate |
|---|---|---|---|---|---|---|---|
| **F1** reachability | 2d | 0.82 | 0.75 | 0.73 | 0.74 | 0.73 | ✅ **PASS** |
| (connectivity) | 1d | 0.51 | 0.50 | 0.53 | 0.52 | 0.50 | gap +0.25 |
| | gnn | 0.88 | 0.59 | 0.50 | 0.50 | 0.50 | gnn@8 0.88 |
| **F4** bounded-dist | 2d | 0.85 | 0.74 | 0.70 | 0.69 | 0.69 | ❌ **FAIL** |
| (metric) | 1d | 0.57 | 0.53 | 0.53 | 0.54 | 0.53 | gap +0.22 (strong) |
| | gnn | 0.56 | 0.50 | 0.50 | 0.50 | 0.50 | **gnn@8 0.56 < 0.70** |
| **F5** CA-rollout | 2d | **0.92** | 0.91 | 0.90 | 0.89 | **0.88** | ✅ **PASS** |
| (dynamics) | 1d | 0.55 | 0.53 | 0.53 | 0.51 | 0.52 | gap +0.37 |
| | gnn | 0.97 | 0.96 | 0.94 | 0.93 | 0.92 | gnn@8 0.97 |

**Suite gate (BOTH new families must pass):** F5 ✅, F4 ❌ → **PARTIAL**.

## Honest reading

- **F5 (local dynamics) — the cleanest length-gen result in the project.** A
  2D-local operator trained on W ≤ 6 predicts a 2-step CA rollout at **0.88–0.92
  across W = 8–20 — nearly flat** — while 1D is at chance and the GNN control
  confirms learnability (0.92–0.97). The property extends beyond propagation to
  *learning to simulate a local dynamical rule*: a genuinely different computation.

- **F4 (metric) — strong 2D signal, but the family is uninformative by its own
  control.** 2D beats 1D by +0.22 (2D 0.69–0.85 vs 1D ~0.54), but the GNN positive
  control fails (0.56 ≈ chance), so per the pre-registered gate F4 cannot be
  counted. **Cause (owned):** F4's threshold `K = 2(W−1)` *scales with W*, so the
  target is not cleanly scale-invariant — even the GNN can't transfer the
  W-dependent threshold across sizes. The control did its job (caught a
  not-cleanly-length-generalizable task). **Fix for future:** a *fixed* K makes
  the distance task scale-invariant; that redesign is deferred, not chased here
  (no post-hoc gate change, G1/G2).

- **F1 reproduces B.1** (reachability, gap +0.25, PASS) as the reference.

## The clarifying finding (why this PARTIAL is still informative)

F5 has **fixed** intrinsic computation depth (T=2), and 2D-local length-generalizes
**flatly** (0.88 at W=20). F1/reachability needs **W-scaling** depth (path length
~W), and generalization *fades* with W (B.1/B.2's "depth bound"). So:

> **The depth bound is not a limitation of 2D-local per se — it bites only when the
> task's required propagation depth grows with problem size.** For bounded-depth
> computation, 2D-local length-generalizes essentially perfectly.

This reframes B.2's adaptive-depth NULL honestly: adaptive depth is needed *only*
for depth-scaling tasks (reachability); it is *irrelevant* for bounded-depth tasks
(dynamics), where fixed-depth 2D-local already generalizes flatly. F5 is the
existence proof.

## What this does for the "generality" claim

- **Established (clean, controlled):** 2D-local length-generalization holds on
  **connectivity (F1)** and **local dynamics (F5)** — two distinct computation
  types, both with passing GNN controls. The B.1 result is **not**
  reachability-specific.
- **Suggestive but not gate-clean:** **metric/bounded-distance (F4)** — strong
  2D-vs-1D, control-confounded by a W-scaling threshold (fixable).
- **Net:** the base is broadened from one task to a two-clean-family suite plus a
  suggestive third — a materially stronger foundation for the flagship
  learned-layout bet, without over-claiming a full 3/3 sweep.

## Caveats

Pilot scale, grid families (layout free — this tests *execution* generality, not
the learned-layout problem). Hand-engineered, supervised → structure-exploitable,
Goal 1 stays closed.

## Decision

**B.3 PARTIAL** (F1 + F5 clean; F4 control-confounded). The generality of 2D-local
length-generalization is established for connectivity and dynamics; the metric
family needs a fixed-K redesign to be gate-clean. Combined with A.0.5/A.0.6
(bandwidth) and B.1 (reachability length-gen), Phase 12's computational-medium
result now spans multiple reasoning types. Next per the roadmap: either a quick
F4 fixed-K rerun to close the third family, or move to the flagship learned-layout
problem (the deferred novelty).
