# Phase 12 — Stage B.3.1 pre-registration: F4 fixed-K redesign (G1)

B.3 left F4 (metric) control-confounded: its threshold `K = 2(W−1)` *scaled with
W*, so the target was not scale-invariant and the GNN positive control failed
(0.56), making the family uninformative — despite a strong 2D-vs-1D gap (+0.22).
This is a **task-design fix, not a threshold change**: redesign F4 to be genuinely
scale-invariant, re-register, and re-run under the identical gate. Fixed before
`stageb3_1.py` runs.

## Root cause

With S top-row / T bottom-row, the *minimum* S→T distance is `W−1` — it scales
with W. Any distance threshold on that geometry is inherently W-coupled, so
"dist ≤ K" is a different question at each W; even a GNN cannot transfer it.

## Redesign (F4b — bounded-radius fixed-K distance)

- **S** random anywhere; **T** placed at a random offset within **Chebyshev
  radius R = 8** of S (clamped to grid); random walls (wall_p 0.28); S, T open.
- **label = 1 iff T reachable from S AND hop-distance(S,T) ≤ K, with FIXED K = 6.**
- Balanced 50/50 by rejection.

Now the computation — "bounded ≤6-hop distance to T" — is **identical at every W**
(scale-invariant): a genuinely local metric task. The 2D advantage persists (a 2D
BFS neighbourhood is non-local in row-major 1D regardless of orientation); the GNN
control *should* now generalise (≤6 hops, 16 layers, scale-free).

## Protocol & gate (identical to B.1 / B.3)

Train mixed W ∈ {4,5,6}; test unseen W ∈ {8,10,12,16,20}; 3 seeds; arms
2d/1d/gnn. **F4b PASSES** iff: mean gap(2D−1D) over W ∈ {8,10,12} ≥ 0.10 AND
2D ≥ 1D at every test W AND GNN@8 ≥ 0.70.

## Verdict rule

- **F4b PASS** → the metric family is now gate-clean; combined with F1
  (connectivity) and F5 (dynamics), the reasoning suite reaches **3/3** — full
  generality of 2D-local length-generalization across connectivity, metric, and
  dynamics.
- **F4b FAIL (GNN still < 0.70)** → bounded-distance is genuinely hard for GNN
  length-gen; keep the B.3 PARTIAL verdict and report honestly (no further
  redesign — avoid motivated iteration, G2).
