# Stage A.0.8b pre-registration — trained global on BOUNDED-DEPTH tasks (F4b, F5)

**Date committed: 2026-07-20 (before the run). No-escape-hatch: one run at
these settings; the outcome is reported regardless of direction.**

## Motivation (second reviewer, revision round 2 — their top-priority request)

The E1 curriculum-UT (a08) trains on F1 (0.84 @ W∈{4,5,6}) and transfers
nothing (chance at W∈{16,20}). But the UT's per-step embeddings fix its
iteration count at 16, and F1's required depth scales with W — so its
transfer failure at large W is partly an architectural necessity (the paper
flags this caveat). The decisive test of "global attention cedes the
transfer axis" must therefore use tasks whose intrinsic depth is BOUNDED and
well within 16 iterations at every evaluated size: F4b (≈K=6 propagation
steps) and F5 (T=2 CA steps). If a trained global fails to transfer THERE,
depth budget is eliminated as the explanation and the cause is isolated to
the absence of locality (the operator has no scale-equivariant structure to
transfer). If it transfers, the §7.1 boundary must be narrowed honestly to
depth-scaling tasks.

## Setup (identical to a08 except the family)

- Model: UniversalTransformer, hidden 64, 16 iterations, pre-LN, per-step
  embeddings, grad clip 1.0.
- Curriculum: W=4 (12 ep) → {4,5} (12 ep) → {4,5,6} (24 ep); lr 1e-3 with
  300-step linear warmup; Adam; batch 128; N_TRAIN=3000/W, N_TEST=1000/W.
- Families: F4b = `build_f4b(n, W, W, 0.28, seed)` (K=6, radius 8, exact
  wall-count matching) and F5 = `build_f5(n, W, W, seed)` (T=2), exactly as
  in Stage B.3/B.3.1.
- Eval: W ∈ {4, 5, 6, 8, 10, 12, 16, 20}; 3 seeds {0, 1, 2} (matching a08).
- Existing reference points (5-seed, harden_stats/lengthgen): 2D-local F4b
  0.96 → 0.93 and F5 0.92 → 0.89, both ~flat to W=20; 1D-local 0.57–0.68 /
  ≈0.53.

## Pre-registered readouts

1. **TRAINS gate (per family):** mean acc over W ∈ {4,5,6} ≥ 0.80. A family
   that fails this is reported as a training failure and yields NO transfer
   verdict for that family.
2. **Transfer verdict (per family, only if TRAINS):**
   - "no transfer": mean acc over W ∈ {16, 20} ≤ 0.60;
   - "transfers": acc at W = 20 ≥ 0.80;
   - otherwise "partial" (reported as measured, no rounding toward either
     claim).
3. **Decisive-locality claim** (the reviewer's requested conclusion —
   "depth budget eliminated; locality absence is the cause") is made only
   if BOTH families TRAIN and BOTH return "no transfer", against the
   existing 2D-local ≥ 0.85 at the same widths.
4. Any other pattern (e.g., transfers on bounded-depth tasks) narrows §7.1
   to depth-scaling tasks and is written into the paper as such.

## Artifacts

Runner `a08b_global_bounded.py`; results to `a08b_global_bounded.json`;
paper deltas to §7.1 (+ abstract/limitations as needed).
