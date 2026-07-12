# Phase 13 — PLAN (pre-registration): Layout-as-Schedule Reasoning (flagship)

| Field | Value |
|---|---|
| Phase | 13 (Goal 3 flagship — the **learned-layout** problem, the novel half Phase 12 got free via grids) |
| Thesis | **Layout-as-Schedule Reasoning:** a model that *learns* to place an implicit-structure problem into a 2D lattice (differentiable placement) and computes by local iteration, inheriting Phase 12's flat size-generalization on structure that is **not natively a grid**. |
| Status | 🟢 OPEN — pre-registration (Rev 1). This is the **"try A" gate**; the pivot-to-B rule is fixed below. |
| Worktree | `2d-lang-phase13`, branch `phase13` (off `main` @ `a4c42af`). Runs from main `.venv` via PYTHONPATH. |
| Provenance | Hand-engineered + supervised. A positive = "learned 2D layout is a computationally-exploitable inductive bias"; Goal 1 stays CLOSED. |

## 1. Why implicit-graph (the Grötschla lesson)

The literature scan (2026-07-12) established that **"beat a GNN on size-generalization for propagation on *given* graphs" is already solved**: Grötschla et al. 2022 (RecGRU-E + skip/L2/edge-conv) size-generalizes to **1000× larger** graphs on path/prefix/distance. So any flagship where the graph (edge list) is *given* faces a fair baseline that already wins → the project's #5 null.

**Therefore the layout must be genuinely load-bearing: the relational structure is NOT given as edges — it must be inferred from features AND placed.** Here a message-passing GNN cannot be applied directly (no edges); the honest baselines are a **set-transformer** (fully-connected attention = global, O(N²)) and a **kNN-GNN** (graph guessed from feature similarity). The learned-layout's claimed edge is Phase 12's: a *regular 2D lattice* size-generalizes flatly where global attention and a size-shifting kNN-graph do not.

## 2. The task — Latent-Grid Region Connectivity from an unordered set (LGRC)

- A latent **W×W grid**. Each cell has a true position (x,y). A smooth low-frequency field defines a **land/water** mask (~50% land → connected 2D blobs). Two land cells are marked **S, T**.
- **Label:** are S and T in the same **4-connected land component** (on the true grid)?
- **Input:** the N = W² cells as an **unordered set**. Per-cell features `[u, v, land_bit, is_S, is_T]`, where `(u,v) = R·(x,y) + noise` — the position **affinely scrambled by an unknown rotation/shear R + noise** (recoverable but not given). **No true (x,y), no adjacency.**
- To answer: recover a consistent 2D layout from `(u,v)`, then flood-fill land connectivity. A genuinely *placement-then-local-compute* problem.

Validity: label-balanced 50/50 by rejection; S,T always land; the scramble R re-drawn per instance so absolute coordinates carry no signal (only relative structure).

## 3. Arms (same info, differ in how they use structure)

| Arm | Mechanism |
|---|---|
| **layout-2D** | encoder → **differentiable soft-assignment of cells → W×W lattice cells (Sinkhorn / Gumbel-Sinkhorn)** → Phase 12 local-CA on the lattice → readout at S/T cells |
| **set-transformer** | fully-connected self-attention over the set (the Phase-11 strong baseline; global, O(N²)) |
| **kNN-GNN** | build a k-NN graph in `(u,v)` space, message-pass (GIN/GRU), readout |
| **random-layout** (control) | place cells on the lattice by a *random* permutation (not learned) + local-CA — isolates *learned* placement from *any* lattice placement (the load-bearing control, cf. Phase 11's set-control) |

## 4. Headline metric — size-generalization

Train on small W ∈ {6,8}; test on unseen larger W ∈ {12,16,20}. Report accuracy vs W per arm (3 seeds).

## 5. Pre-registered win / null (the "try A" gate) + pivot rule

**PASS (A has signal) iff:**
1. **layout-2D learns the task** at train size (≥ 0.75 at W=8), AND
2. its **size-generalization retention beats both** set-transformer AND kNN-GNN — `acc(layout-2D) − max(acc(set-tf), acc(kNN-GNN)) ≥ 0.10` at the largest test W, AND
3. **layout-2D beats random-layout** at the largest test W by ≥ 0.10 (the win is *learned* placement, not just "put it on a lattice").

**NULL / PIVOT-TO-B iff:** layout-2D fails to train, OR a baseline ties/beats it on size-gen, OR it does not beat random-layout. Per the user's decision (2026-07-12): **A produces nothing → pivot to Plan B** (reposition Phase 12's *mechanism* — the dimensionality×locality bandwidth/length-gen analysis — as the paper contribution, rather than a new architecture that beats graph methods).

**No-escape-hatch (G):** thresholds fixed here before the run; a NULL closes A cheaply (no full build), exactly as Phase 11/12's cheap-gate discipline.

## 6. Honest risks (fixed in advance)

- **The Phase-11 risk:** the set-transformer is strong (Phase 11: 2D-vs-set ≈ 0). If it size-generalizes as well as layout-2D, A is null. This is the main threat and the reason the gate is *cheap*.
- **kNN-GNN may recover the layout** from `(u,v)` well enough that the lattice adds nothing. If so, a follow-up would weaken the position signal (rely on field-smoothness only) — but only if step-1/step-3 already show promise; not chased on a null.
- **Sinkhorn placement is finicky** to train (temperature, iterations). If it won't train, that is itself a NULL for this mechanism at pilot budget (report honestly; do not tune indefinitely).
- Pilot scale, synthetic, single task. Structure-exploitable, Goal 1 closed.

## 7. Stage plan

- **C.0 — cheap gate (this PLAN):** build the LGRC task + the 4 arms + the size-gen sweep; run the pre-registered gate. ~days, not weeks.
- **C.1+ (gated on C.0 PASS):** harder implicit structure (field-smoothness-only, no `(u,v)`), more task families, stronger baselines (graph-transformer), the differentiable-placement ablations, and the utility north-stars (CLRS / ARC-AGI).

## 8. Literature anchors
- Grötschla, Mathys, Wattenhofer 2022, *Learning Graph Algorithms with Recurrent GNNs* (arXiv:2212.04934) — the recurrent-GNN size-gen result that forces the implicit-graph framing.
- Alon & Yahav 2021 (oversquashing); Di Giovanni et al. 2023; Barbero et al. LASER 2023 (rewiring — the crowded neighbourhood we avoid by going implicit).
- Mena et al. 2018 (Gumbel-Sinkhorn); Cuturi 2013 (Sinkhorn) — differentiable placement.
- Mordvintsev et al. 2020 (Neural CA); Xu & Miikkulainen 2025 (NCA-for-ARC) — local-2D-as-computer.
