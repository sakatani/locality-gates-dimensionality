# Phase 12 — Stage B.1 pre-registration: length-generalisation (G1)

The test that matters for "2D as a computational medium" and the FM aspiration:
**train small, test large.** A.0.5 only tested fixed-W threshold-crossing. Fixed
before `lengthgen.py` runs.

## Setup

- **Train** on grids W ∈ {4, 5, 6} (mixed, per-W batches), H=W, wall_p 0.28.
- **Test** on held-out **larger** W ∈ {7, 8, 9, 10} (headline) + {12} (context) —
  sizes never seen in training.
- Arms: **2d**, **1d** (row-major), **1d_hilbert**, **gnn** (positive control).
- 3 seeds, matched depth 16 / hidden 32 (2D & 1D identical params). Metric =
  held-out test accuracy per test-W (task already class-balanced + density-matched).

## Mechanism under test

Fully-convolutional arms are architecturally size-agnostic. The question is
whether the *learned computation* transfers: a 2D-local propagation rule is
scale-equivariant (same 4-neighbour rule at any W, bounded by depth), whereas the
row-major 1D serialisation is **not** scale-invariant — vertical neighbours sit W
apart, so a kernel learned at gap-6 is miscalibrated at gap-12.

## Pre-registered win condition

> **PASS iff** `mean over test-W ∈ {7,8,9,10} of [acc(2d) − acc(1d)] ≥ 0.10`
> **AND** `acc(2d) ≥ acc(1d)` at **every** headline test-W.

**Positive control (required for a valid PASS):** the **GNN** (scale-invariant on
the graph) must itself generalise — `acc(gnn) ≥ 0.70` at test-W=8 — confirming the
task IS length-generalisable and the 1D failure is the serialisation, not that
*nothing* transfers. If even the GNN collapses, the test is vacuous → re-scope
(likely a depth-bound issue) rather than declare a 2D win.

**Hilbert** reported (expected to degrade like/worse than row-major — its
serialisation is also W-dependent).

## Honest caveats (fixed in advance)

- 2D-local is itself depth-bounded: at the largest test W (10, 12) even a
  perfectly-transferring 2D rule caps because path length > depth 16. The headline
  is therefore **2D-vs-1D retention on unseen W**, not absolute 2D accuracy at
  every W. W=12 is context only.
- Single depth/hidden; multi-scale training refinement (curriculum over W) and a
  capable-global length-gen comparison are later Stage-B items.

## Verdict rule

**PASS** (2D retains, 1D degrades, GNN control generalises) → strong support for
the computational-medium reading + the FM-relevant generalisation axis. **NULL**
(gap < 0.10, or 1D generalises as well as 2D) → the fixed-W A.0.5 advantage does
NOT extend to length-generalisation → temper the claim, reconsider Stage B scope.
Report whichever obtains; no post-hoc threshold changes.
