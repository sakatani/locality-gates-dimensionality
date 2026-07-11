# Phase 12 — Stage B.3 pre-registration: reasoning-task suite (generality, G1)

B.1 showed 2D-local length-generalises on **reachability** (F1). B.3 tests whether
that is a **general** property of 2D-local iteration across *different* reasoning
tasks, or reachability-specific. Fixed before `stageb3.py` runs.

## Families (all grid-based → layout free; isolates *execution* generalization)

- **F1 — reachability** (connectivity). Already validated (B.1); re-run here as the
  reference under the identical protocol.
- **F4 — bounded shortest-hop distance** (metric). S top-row, T bottom-row, random
  walls; label = 1 iff T reachable from S AND hop-distance(S,T) ≤ K with
  `K = 2·(W−1)` (a direct-ish path qualifies, a very winding one does not).
  Distinct from F1: requires a *distance* computation (BFS wave), not just
  connectivity. Balanced 50/50 by rejection.
- **F5 — cellular-automaton rollout** (local dynamics). Random binary initial
  state (no walls); fixed totalistic 4-neighbour rule; label = state of the query
  cell after **T=2** steps. The readout depends on a (2T+1)² = 5×5 neighbourhood —
  local in 2D, spans ~2T·W indices in row-major 1D. Balanced ~50/50 by rejection.
  (A local dynamical-simulation task, distinct from the propagation of F1/F4.)

Arms per family: **2d-local** (should generalise), **1d-local** (row-major, should
fail — not scale-invariant), **gnn** (positive control — should generalise; if it
doesn't, the task isn't length-generalizable and that family is uninformative).

## Protocol (identical to B.1)

Train mixed W ∈ {4,5,6}; test unseen W ∈ {8,10,12,16,20}; 3 seeds; hidden 32,
fixed depth 16; the shared 6-channel tensorisation (H1 parity).

## Per-family gate (reuse the B.1 gate verbatim)

A family **PASSES** iff:
1. mean gap(2D−1D) over test W ∈ {8,10,12} ≥ **0.10**, AND
2. 2D-local ≥ 1D-local at **every** test W, AND
3. GNN positive control @ W=8 ≥ **0.70** (the task *is* length-generalizable).

## Suite headline (fixed)

> **Generality PASS iff BOTH new families (F4 AND F5) pass the per-family gate.**
> One of two → *partial* (2D length-gen extends beyond reachability but not
> universally). Neither → the B.1 effect is reachability-specific; report and
> reconsider the "computational medium" generality claim.

## Honest caveats (fixed in advance)

- Some families may be depth-bounded (fade at large W, cf. B.1/B.2); the gate uses
  mean-gap over W≤12 + "2D≥1D everywhere", robust to graceful fading.
- Grid families only (layout free) — this tests execution generality, NOT the
  learned-layout problem (the flagship, deferred). A suite PASS strengthens the
  base for that next bet; it does not pre-empt it.
- Pilot scale, hand-engineered, supervised. Structure-exploitable, Goal 1 closed.

## Verdict rule

Suite PASS → "2D-local iteration length-generalises across connectivity, metric,
and dynamics reasoning — a general computational-medium property, not
reachability-specific." Partial/NULL → recorded honestly, no post-hoc gate change.
