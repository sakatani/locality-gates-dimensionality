# Phase 12 — close-out (Goal 3 axis ②: 2D as a computational medium)

| Field | Value |
|---|---|
| Phase | 12 (Goal 3 reframe — "is a 2D **field + spatially-local operators** a better computational substrate than a 1D serialization, attributable to the *dimensionality × locality* interaction?") |
| Status | ✅ **PILOT-POSITIVE on the headline** (2 pre-registered positives), with honest boundaries recorded. Distinct from Phase 11 (axis ①, input-encoding, CLOSED null): this axis imposes a *locality constraint* Phase 11 never did. Phase remains **open** for further Stage-B hardening. |
| Substrate / "language" | **layout-as-schedule (LaS)** — a 2D placement where spatial adjacency = one-step interaction; meaning = fixed point of iterated local updates. NOT v2.x. |
| Date | 2026-06-28 → 2026-07-04 |
| Branch / worktree | `phase12` / `2d-lang-phase12` (runs from main `.venv` via PYTHONPATH; MPS) |
| Artifacts | `experiments/phase12_2d_compute_medium/{substrate,families,models,calibrate,probe,harden,a07,lengthgen,stageb2}.py` + `tests_pytest/` (23 passed); `PLAN.md`, `STAGE_A0_REPORT.md`, `STAGE_A05_*`, `STAGE_A06_*`, `STAGE_A07_*`, `STAGE_A_SYNTHESIS.md`, `STAGE_B1_*`, `STAGE_B2_*`; `*.json` |

## Verdict

> **At pilot scale, a 2D layout with spatially-local operators is a better
> computational substrate than the best 1D serialization for iterative relational
> reasoning — attributable to the dimensionality of the layout under a locality
> constraint, not to generic capacity or iteration.** Two independent
> pre-registered positives: **(1) bandwidth** — at fixed problem width, a
> 2D-local operator clears the task at materially larger widths than a
> parameter-and-window-matched 1D-local operator (and than the best
> space-filling-curve serialization); **(2) length-generalization** — a 2D-local
> operator trained on small grids transfers to unseen larger grids where the 1D
> serialization is at exact chance. This is the project's **first performance-line
> positive that survives its own controls**, and it is **consistent with Phase
> 11** (the advantage requires the locality constraint Phase 11's global attention
> never imposed).

Honest boundaries (equally load-bearing): the advantage is **depth-bounded** and
adaptive depth did **not** break it at pilot budget (B.2 NULL); the **"2D vs
iteration" question is open** (A.0.7 could not train a capable global). This is a
**structure-exploitable** result on a hand-engineered, supervised substrate —
**Goal 1 stays closed** (14 nulls); a **positive signal toward**, not proof of,
the foundation-model aspiration.

## The six gates (all pre-registered; G1 audit trail committed before each run)

| Gate | Question | Result |
|---|---|---|
| **A.0** | families valid & non-vacuous; does W decouple 1D/2D bandwidth? | ✅ GREEN — 2D-bw=1 vs best-1D-bw=W; solvable (GNN 0.965 / 2D 0.935); F3 control clean |
| **A.0.5** | 2D-local vs matched 1D-local at fixed W (threshold-crossing W\*) | ✅ **PASS** — W\*_2D=8 vs W\*_1D=4 (gap +4 ≥ 2); F3 control gap −2 |
| **A.0.6** | does the *best* 1D serialization (space-filling/Hilbert) rescue 1D? | ✅ **PASS (gating)** — Hilbert W\*=4 = row-major; "bad-serialization" objection defeated |
| **A.0.7** | is the advantage 2D, or just iteration? | ⚠️ **INCONCLUSIVE** — a param-matched, stabilised Universal Transformer failed to train (W\*=0); locality is the differentiator *among trainable arms* |
| **B.1** | train small W, test unseen large W (length-generalization) | ✅ **PASS** — 2D 0.93→0.74 on unseen W=7–12; 1D at exact chance; mean gap +0.320; GNN control generalises |
| **B.2** | does adaptive depth (neural-CA) break the depth bound? | ❌ **NULL** — CA wins mid-range but ties@16 / loses@20 vs fixed-depth; depth bound unbroken |

## The arc (what was decided and built)

1. **Goal 3 re-opened (with the user)** after Phase 11's pilot-negative; chose axis
   ② (2D as a computational medium) over the trace/controllability and
   diagrammatic-irreducibility axes, and pre-registered a PLAN first (the
   bandwidth thesis: P2 multi-axis-independence recast as graph bandwidth under
   the best layout).
2. **Calibration → decisive probe → adversarial hardening** (A.0 → A.0.5 → A.0.6):
   established the headline and defeated its sharpest objection (best-1D
   serialization) with matched-param, matched-window controls + an F3
   capacity control.
3. **Adversarial "iteration" check** (A.0.7): gave a capable iterative-global a
   fair shot (param-parity, pre-LN, step embeddings, grad-clip); it failed to
   train — inconclusive, honestly recorded (not counted as a 2D win).
4. **The FM-relevant test** (B.1): length-generalization — the strongest result;
   the learned 2D computation transfers small→large where 1D cannot.
5. **The unbounded prize** (B.2): adaptive-depth neural-CA to break the fixed-depth
   bound — NULL; the CA helps mid-range but does not decisively beat fixed-depth,
   which is itself a surprisingly strong flat generaliser. Recorded per
   no-escape-hatch; not chased further.

## Why this is credible (methodology)

The cheap-gate-before-investment discipline (Phase 11's win) held: six
pre-registered gates (~1 hour total compute) located two robust positives, one
inconclusive, and one clean null — with the load-bearing controls (matched
locality budget, Hilbert serialization, F3 capacity control, GNN positive
control) all run. The discipline worked **both ways**: five gates confirmed, B.2
NULLed against its own pre-registration, and no threshold was moved post-hoc.

## Caveats (honest scope)

- **Pilot, not a formal study.** Small models (hidden 32), 2–3 seeds, no Holm
  correction across the family, single headline task family (F1 reachability).
- **Depth-bounded.** The generalization is retention/graceful-degradation, not
  unbounded (B.2). A genuinely unbounded 2D computational medium needs a stronger
  adaptive-depth recipe (capacity parity, fixed-point/ACT training) — future work.
- **"2D vs iteration" open.** A capable iterative-global could not be trained
  (A.0.7); the claim is locality-conditioned, not "2D > any global".
- **Supervised, hand-engineered.** Structure-exploitable, **not emergent** — Goal
  1 stays closed at 14 nulls. Per the provenance guard, a positive here does not
  bear on Goal 1.

## Implication for the foundation-model aspiration

The user's FM aspiration ([[foundation_model_aspiration]]) was gated on finding a
representational hypothesis where 2D is genuinely load-bearing over fair
baselines. Phase 11 argued *against* the input-encoding form. **Phase 12 provides
the first positive signal for a different form** — 2D as a *computational medium*
(locality + layout), with length-generalization (the FM-relevant axis)
demonstrated at pilot scale. This does **not** establish an FM-scale result, but
it is the evidence that *licenses* opening that line as a considered next step:
the principled scaled version would test whether locality-structured 2D
computation, iterated adaptively, gives an inductive-bias / length-generalization
advantage at scale — with the B.2 depth-bound as the first engineering problem to
solve.

## Open Stage-B items (independent, not blocking the close-out)

- Capable iterative-global via curriculum / search-transformer recipe (settle
  A.0.7 + a real matched-FLOPs H3).
- F2 depth-generalization (the CA-rollout family).
- Statistical hardening (≥3 seeds + Holm across the full sweep).
- A genuine adaptive-depth recipe (capacity parity / fixed-point loss / ACT) to
  revisit the B.2 depth bound.
- Engineering-application sketch (2D-native reasoning sublayer / layout-as-schedule).

## Decision

**Phase 12 pilot CLOSED as pilot-positive on the headline** (bandwidth +
length-generalization), with the depth bound and the iteration question recorded
as honest open limitations. Branch `phase12` retained for the Stage-B items above;
merge to `main` when the phase is taken further or explicitly wrapped.
