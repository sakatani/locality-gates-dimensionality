# Phase 12 — Stage A.0.5 pre-registration (G1, fix BEFORE running)

The decisive cheap probe that gates Stage B (PLAN §6, §7). Thresholds below are
fixed **before** `probe.py` is run; the result is read against them with no
post-hoc adjustment (no-escape-hatch G1/G2).

## Headline gate — threshold-crossing W\* (chosen formulation)

For each arm, `W*` = the largest grid width W (square, H=W) at which the arm's
**mean-over-seeds** held-out accuracy on F1 is ≥ **τ = 0.80**, at a **matched
budget** (depth `L = 16`, hidden 32 — the 2D-local and 1D-local arms then have
*identical* parameter counts and *identical* 9-cell windows per layer; the only
difference is the layout dimensionality, PLAN H2).

- **W sweep:** S = {4, 6, 8, 10, 12}. **Seeds:** 3. **wall_p:** 0.28.
- **W\*(arm)** = max W ∈ S with mean-acc ≥ τ (0 if none clear τ).

> **PASS iff `W*_2D − W*_1D ≥ 2`** (cells). **NULL iff `< 2`** (incl. tie / negative).

Interpretation: at matched params + window, the 2D-local layout clears
reachability at materially larger problem widths than the row-major 1D layout —
the bandwidth law. (The *effective cell-reach* differs only because the 2D
arrangement spends the same 9-cell window on a 3×3 neighbourhood instead of a
9-long index run; that IS the dimensional bandwidth advantage being measured,
not a receptive-field mismatch.)

## Control gate — F3 must NOT show the 2D edge (H4)

Same sweep / seeds / budget on F3 (1D-intrinsic row-prefix parity).

> **REQUIRE `W*_2D − W*_1D ≤ 1`** on F3. If F3 shows `≥ 2`, the F1 gap is a
> generic-capacity confound → **CLOSE** (G3).

## Supporting evidence — depth ∝ W (monotone gap)

For each W ∈ S, set depth `L = 2W` (matched across both local arms, H2 preserved),
2 seeds. Expectation (not a hard gate): `gap(2D−1D)` is non-decreasing across S —
the 2D arm stays solvable as depth scales while 1D still pays the W-factor.

## Reported diagnostics (not gates)

- **H3 efficiency:** at W=7, 2D-local vs `global` accuracy + parameter counts
  (compute proxy). Global already underperforms on this iterative task (A.0:
  0.686); report honestly, do not gate on it.
- **GNN W\*** for context (the canonical reachability solver, handed the graph).

## Known limitation (Stage-B robustness item)

The 1D arm uses **row-major** order, which realises bandwidth = W = the grid-graph
optimum class (Chvátalová 1975: no 1D linearisation beats min(W,H)). A
Hilbert / space-filling-curve 1D arm — to preempt "you picked a bad
serialisation" — is deferred to Stage B; the theorem already bounds every 1D
order at Θ(W).

## Verdict rule

**A.0.5 PASS** ⇔ headline `W*_2D − W*_1D ≥ 2` on F1 AND control `W*_2D − W*_1D ≤ 1`
on F3. PASS → build Stage B (full families incl. F2 depth-gen, compute-matched
curves, ≥3 seeds, Holm, engineering-application demo). NULL → CLOSE
pilot-negative, no Stage B (G2).
