# Phase 12 — Stage B.2 pre-registration: adaptive depth (break the depth bound, G1)

B.1 showed 2D-local length-generalises but is **depth-bounded** (fixed 16 layers →
fades at W≥10 as path length exceeds depth). B.2 tests whether an **adaptive-depth**
2D operator — a weight-shared local update iterated to scale with problem size —
generalises to *much* larger unseen grids where the fixed-depth model caps. Fixed
before `stageb2.py` runs.

## Setup

- **Train** mixed W ∈ {4,5,6}, iterations **`T ~ U[W, 3W]` per batch** (CA arms) —
  the canonical neural-CA variable-step recipe (makes the readout a stable
  attractor robust to running more steps at test), wall_p 0.28.
- **Test** unseen W ∈ {8,10,12,16,20}, iterations **`T = 3W`** (generous
  propagation headroom for the larger grids).
- Arms:
  - **ca_2d** — adaptive-depth 2D (`ConvCA2D`, weight-shared gated conv update ×T).
  - **fixed_2d** — the B.1 fixed 16-layer `Local2D` (the depth-bounded baseline).
  - **ca_1d** — adaptive-depth 1D control (`ConvCA1D`, same iterated update on the
    row-major order) — tests whether iterating longer rescues the wrong dimension.
- 3 seeds, hidden 32.

## Mechanism

A weight-shared local update is a learned rule; iterating it T times propagates
information T·(window) cells. Trained on small grids (few steps), it can be run for
**more** steps at test to cover larger grids — the neural-CA property. The fixed
16-layer stack cannot (its depth is frozen). The 1D control still fails: more
iterations of a W-dependent serialisation do not fix the miscalibration.

## Pre-registered win condition

> **PASS iff all three hold:**
> 1. `acc(ca_2d) ≥ 0.70` at **W=16** (holds well beyond the fixed-16 depth bound),
> 2. `acc(ca_2d) − acc(fixed_2d) ≥ 0.15` at **W=16** (adaptive extends past fixed depth),
> 3. `acc(ca_2d) − acc(ca_1d) ≥ 0.20` at **W=16** (dimensionality still matters; iteration ≠ rescue).

## Honest caveats (fixed in advance)

- `T = 2W` is a heuristic proxy for "iterate to fixed point"; a true adaptive-halt
  (ACT) is a refinement. If a fixed large T works as well, note it.
- Neural-CA stability over many iterations is a known risk (blow-up). The gated
  residual + GroupNorm are the stabilisers; if ca_2d is unstable at large T, report
  it honestly rather than tuning to a pass.
- Pilot scale, hand-engineered, single family. Structure-exploitable, Goal 1 closed.

## Verdict rule

**PASS** → the depth bound is broken: adaptive-depth 2D-local generalises to much
larger unseen problems, the unbounded form of B.1 and the strongest computational-
medium evidence. **NULL** (ca_2d also caps, or ca_1d catches up) → the B.1
generalisation is depth-bounded in practice; report and reconsider. No post-hoc
threshold changes.
