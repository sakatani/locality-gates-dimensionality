# Phase 13 — Stage C.0 report: learned-layout cheap gate (NULL → pivot to B)

| Field | Value |
|---|---|
| Stage | C.0 (the "try A" cheap gate for the learned-layout flagship; PLAN §5, thresholds fixed before the run, G) |
| Verdict | ❌ **NULL** on the pre-registered gate → **pivot to Plan B** (per PLAN §5 + the user's 2026-07-12 decision). |
| Artifacts | `experiments/phase13_learned_layout/{task,models,gate}.py`; `gate.json`; 3 seeds; runtime 899 s |

## Result — train W ∈ {6,8}, test unseen W, 3 seeds

| arm | 8 (train) | 12 | 16 | 20 |
|---|---|---|---|---|
| **layout-2D** (Sinkhorn place + CA) | 0.65 | 0.65 | 0.61 | 0.62 |
| **set-transformer** (Phase-11 baseline) | 0.67 | 0.67 | 0.66 | 0.67 |
| **kNN-GNN** (recurrent GIN-E on kNN(u,v)) | 0.75 | 0.68 | 0.69 | 0.61 |
| **random-layout** (control) | 0.50 | 0.50 | 0.50 | 0.51 |

**Pre-registered gate (all three required):**
1. layout learns @8 ≥ 0.75 → **0.65 ✗**
2. layout − max(set-tf, kNN) @20 ≥ 0.10 → **−0.04 ✗**
3. layout − random @20 ≥ 0.10 → **+0.11 ✓**

Two of three fail → **NULL**.

## Honest reading

- **The learned placement is not broken** — layout-2D beats the random-layout
  control by +0.11 and size-generalizes *flatly* (0.65→0.62). The Sinkhorn
  placement extracts real structure and gradients flow. But it **underfits**
  (0.65 at train size vs kNN's 0.75), a joint placement+CA (chicken-and-egg)
  optimization difficulty.
- **The set-transformer beats it** (0.67, flat) — the decisive failure. This is
  the **Phase-11 wall reproduced**: a strong set/attention baseline ties-or-beats
  the 2D approach, size-robustly. The project's recurring #5 pattern at the
  learned-layout level.
- **Partial 2D signal, not enough:** layout and set-transformer are both flat
  while kNN-GNN *degrades* with size (0.75→0.61) — mild support for "regular
  structure size-generalizes where a kNN-graph doesn't". But the set-transformer
  is *also* flat (and higher), so a 2D lattice is not *uniquely* responsible, and
  layout-2D doesn't clear the bar.

## Why not chase a fix (no-escape-hatch)

A competent training fix (placement warm-start / auxiliary layout loss) might lift
layout-2D above 0.75 in-distribution. But even then it must beat a **flat-0.67
set-transformer by +0.10** on size-generalization (reach ~0.77 at W=20 from the
current 0.62) — a steep hill against the Phase-11 wall. Per the pre-registered
rule and the cheap-gate discipline (a NULL closes A cheaply, no motivated
iteration toward a pass), Stage C.0 is recorded as **NULL**.

## Decision

**Phase 13 (learned-layout flagship, Plan A) — NULL at the cheap gate.** The
differentiable-placement mechanism trains but underfits, and a strong
set-transformer baseline beats it size-robustly — the same wall Phase 11 hit. Per
the user's pre-authorized decision, **pivot to Plan B**: reposition Phase 12's
*mechanism* (the dimensionality × locality bandwidth / length-generalization
analysis — "when and why does 2D locality help") as the research contribution,
rather than a new architecture that beats graph methods. The cheap gate did its
job: ~15 min of compute (after a ~30 min lit scan) avoided a multi-week build for
a low-prior outcome, exactly as intended.

## What A leaves on the record (for B and future)

- Differentiable Sinkhorn placement + local CA is *trainable* (grads flow, beats
  random-layout) but *underfits* the joint objective at pilot budget — a real
  mechanism, not a dead end, should anyone revisit with a warm-start / staged
  training.
- kNN-GNN degrades with size while lattice/attention stay flat — a small,
  reusable observation about which structures size-generalize.
