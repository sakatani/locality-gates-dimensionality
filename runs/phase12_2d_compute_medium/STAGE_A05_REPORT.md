# Phase 12 — Stage A.0.5 decisive probe report

| Field | Value |
|---|---|
| Stage | A.0.5 (the decisive cheap probe gating Stage B; PLAN §6/§7) |
| Verdict | ✅ **PASS** against the pre-registered thresholds (`STAGE_A05_PREREG.md`, committed before the run, G1) — **the project's first positive on the performance line** |
| Pre-reg | τ=0.80, sweep W∈{4,6,8,10,12} (square), 3 seeds, matched depth-16 budget, wall_p 0.28 (fixed before running) |
| Artifacts | `experiments/phase12_2d_compute_medium/probe.py`; `probe.json`; runtime 436 s, MPS |

## Result vs the pre-registered gates

**Headline (F1) — threshold-crossing W\*** (mean over 3 seeds, ≥ τ=0.80):

| W | 4 | 6 | 8 | 10 | 12 | **W\*** |
|---|---|---|---|---|---|---|
| **2D-local** | 1.00 | 0.97 | 0.85 | 0.69 | 0.70 | **8** |
| **1D-local** | 0.99 | 0.72 | 0.65 | 0.62 | 0.63 | **4** |

> **W\*_2D − W\*_1D = 8 − 4 = +4 ≥ 2 → PASS.** Well-separated at the crossing
> (W=6: +0.25, W=8: +0.20), not a τ-boundary artifact. At a matched parameter
> count and matched 9-cell window per layer, the 2D layout clears reachability
> at twice the grid width the row-major 1D layout can — the bandwidth law.

**Control (F3, 1D-intrinsic) — 2D must NOT show the edge:**

| W | 4 | 6 | 8 | 10 | 12 | **W\*** |
|---|---|---|---|---|---|---|
| **2D-local** | 1.00 | 0.96 | 0.84 | 0.74 | 0.69 | **8** |
| **1D-local** | 1.00 | 0.99 | 0.90 | 0.97 | 0.74 | **10** |

> **W\*_2D − W\*_1D = 8 − 10 = −2 ≤ 1 → control OK.** On the 1D-intrinsic task
> 1D-local is *better* (its row stays contiguous in row-major order), exactly as
> a no-bandwidth-advantage control should behave. This rules out the
> generic-capacity confound (H4 / G3): the F1 win is the *layout dimensionality*,
> not 2D arms being stronger in general.

**PASS = (F1 gap +4 ≥ 2) AND (F3 gap −2 ≤ 1) → ✅.**

## Supporting & diagnostic (not gates — reported honestly)

- **depth ∝ W (supporting):** gap(2D−1D) = +0.01 / **+0.26** / **+0.20** / +0.08 / +0.08 at W=4/6/8/10/12 (depth=2W). The gap is large and positive in the mid-range but **non-monotone** — at W≥10 both arms fall (2D=0.68/0.70) because depth 2W=20/24 with hidden 32 is still capacity-bound for the longest winding paths, so the gap closes as 2D also fails. The bandwidth gap is robust where both arms are in their working range; it is not a clean monotone curve at this model scale. (Honest: the W\* gate is the load-bearing evidence; depth∝W corroborates but does not add a clean scaling law here.)
- **H3 efficiency — NOT cleanly demonstrated.** At W=8: 2D-local = 0.87 (149k params) vs `global` = 0.58 (35k params). Global is *cheaper* and *worse* — full attention (4 layers) is architecturally unsuited to iterative reachability, reproducing Phase 11's "global gets nothing from 2D". So this is **not** a "2D at less compute" result; it is "2D-local can do an iterative task that global cannot". The compute-efficiency claim (PLAN H3) remains **unproven** and is deferred to Stage B (matched-FLOPs curves, a looped/iterative global baseline).
- **GNN diagnostic (1 seed, noisy):** W\* = 6 (1.00/0.99/0.71/0.68/0.66). 2D-local (W\*=8) matches/exceeds the GNN handed the true adjacency — consistent with "proximity = schedule" (2D-local recovers the propagation graph from positions alone), but single-seed and noisy; not weight-bearing.

## What this does and does not establish

**Does:** at pilot scale, with a pre-registered, capacity-controlled design, a
2D-local operator has a materially higher problem-size ceiling than a
parameter-and-window-matched 1D-local operator on a low-2D-bandwidth /
high-1D-bandwidth task, and the advantage **vanishes (reverses) on a
1D-intrinsic control**. The effect is bandwidth-attributable (grounded in the
grid-graph bandwidth = min(W,H)), not generic capacity. This is the first
performance-line positive after Phase 11's pilot-negative and the project's 14+2
nulls — and it is consistent with Phase 11 (the advantage requires a *locality
constraint*, which Phase 11's global attention never imposed).

**Does not:** (a) demonstrate compute-efficiency over a *capable* global baseline
(H3 unproven — global was incapable, not merely costlier); (b) show a clean
monotone scaling law (depth∝W non-monotone at this scale); (c) extend beyond the
single headline family F1 (F2 depth-generalisation untested); (d) rule out that
the best 1D serialisation (Hilbert/space-filling) narrows the gap — though the
bandwidth theorem bounds every 1D order at Θ(W). All four are exactly the Stage B
agenda.

## Decision

**Stage A.0.5 PASS → build Stage B** (per the pre-registered verdict rule). Stage
B must convert this pilot positive into a defensible result by adding: F2
depth-generalisation, a Hilbert/space-filling-curve 1D control, a **capable**
iterative-global baseline (looped transformer) for a real compute-matched H3
test, length-generalisation (train W ≤ W0, test W > W0), ≥3 seeds with Holm
correction, and the engineering-application sketch (PLAN §8). The honest framing
throughout: this is a *structure-exploitable* inductive-bias result on a
hand-engineered substrate (Goal 1 stays closed), not emergence and not yet a
foundation-model claim.
