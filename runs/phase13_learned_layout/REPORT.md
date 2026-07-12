# Phase 13 — close-out (Goal 3 flagship: the learned-layout problem)

| Field | Value |
|---|---|
| Phase | 13 (Goal 3 flagship — "Layout-as-Schedule Reasoning": can a model *learn* a differentiable 2D placement of an implicit-structure problem, and inherit Phase 12's flat size-generalization on structure that is not natively a grid?) |
| Status | ❌ **CLOSED — NULL at the cheap gate (Stage C.0)** → pivot to Plan B, per the pre-registered rule and the user's pre-authorized decision ("try A; if nothing, pivot to B"). |
| Date | 2026-07-12 → close 2026-07-13 |
| Artifacts | `experiments/phase13_learned_layout/{task,models,gate}.py`; `runs/phase13_learned_layout/{PLAN.md,STAGE_C0_REPORT.md,gate.json}` |

## Verdict

> **At pilot scale, a learned differentiable placement (Sinkhorn soft-assignment
> onto a 2D lattice + local-CA execution) does not beat a set-transformer on an
> implicit-structure connectivity task — neither in-distribution nor on
> size-generalization.** The placement mechanism itself is *trainable* (gradients
> flow through Sinkhorn; it beats the random-layout control by +0.11 at W=20),
> but it underfits the joint placement+execution objective, and the
> set-transformer is flat-and-higher across all sizes. This reproduces the
> project's #5 / Phase-11 pattern — a strong set/attention baseline ties or beats
> the 2D approach — for the third time (Phase 10b, Phase 11, Phase 13).

## The arc

1. **FM-framing (with the user):** modality-agnostic new reasoning architecture,
   novelty-primary; flagship = the learned-layout problem (the half Phase 12 got
   free via grids).
2. **Literature scan (~30 min) reshaped the design:** Grötschla et al. 2022
   (arXiv:2212.04934) — recurrent GNNs size-generalize **1000×** on *given*
   sparse graphs → any given-graph flagship faces a fair baseline that already
   wins. Rewiring/oversquashing is mature and crowded (Di Giovanni, Black,
   LASER); Sinkhorn placement is off-the-shelf. → The layout must be
   load-bearing: **implicit structure** (no edge list), where message-passing
   cannot be applied directly.
3. **Task LGRC:** latent W×W grid, smooth-field land/water mask, S/T land cells;
   label = same 4-connected land component; input = **unordered set** of cells
   with affinely-scrambled coordinates (recoverable, not given).
4. **Stage C.0 cheap gate** (pre-registered thresholds, G): train W∈{6,8}, test
   unseen W∈{12,16,20}, 3 seeds, 4 arms.

## Stage C.0 result (the gate)

| arm | W=8 (train) | 12 | 16 | 20 |
|---|---|---|---|---|
| layout-2D (Sinkhorn + CA) | 0.65 | 0.65 | 0.61 | 0.62 |
| set-transformer | 0.67 | 0.67 | 0.66 | **0.67** |
| kNN-GNN (recurrent GIN-E) | **0.75** | 0.68 | 0.69 | 0.61 |
| random-layout (control) | 0.50 | 0.50 | 0.50 | 0.51 |

Gate: learns@8 = 0.65 < 0.75 ✗; layout − best-baseline @20 = −0.04 < 0.10 ✗;
layout − random @20 = +0.11 ≥ 0.10 ✓ → **2/3 fail → NULL** (no post-hoc changes).

## What survives on the record

- **Differentiable placement works mechanically** (grads flow; beats random
  layout; size-generalizes flatly) but underfits the chicken-and-egg joint
  objective at pilot budget. A staged/warm-started recipe is genuine future work
  — but even a fixed trained layout must then beat a flat-0.67 set-transformer.
- **kNN-graphs degrade with size while lattice and attention stay flat** — a
  small reusable observation about which structures size-generalize.
- **Methodology:** ~15 min of gate compute (after the 30-min literature scan)
  avoided a multi-week build. The cheap-gate discipline worked a third time.

## Decision

**Phase 13 CLOSED — NULL.** Goal 3's performance line is now answered three ways
(notation 10b, input-encoding 11, learned-layout 13) with one narrow positive
(computational-medium 12, locality-conditioned). **Plan B** proceeds: reposition
Phase 12's mechanism — *locality gates dimensionality* (bandwidth separation,
scale-equivariance, depth taxonomy) — as the research contribution, with Phases
11/13 and the Grötschla result as the boundary map. Hand-engineered + supervised
throughout; Goal 1 stays closed at 14 nulls.
