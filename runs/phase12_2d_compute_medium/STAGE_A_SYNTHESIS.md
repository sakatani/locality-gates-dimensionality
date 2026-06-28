# Phase 12 — Stage A pilot synthesis (the bandwidth thesis, pilot-positive)

Status: **Stage A pilot COMPLETE — positive on the headline, one threat deferred.**
Stage B not yet built. Branch `phase12`.

## The four gates

| Gate | Question | Result |
|---|---|---|
| **A.0** calibration | families valid? knob decouples bandwidth? | ✅ GREEN — 2D-bw=1 vs best-1D-bw=W; solvable (GNN 0.965 / 2D 0.935); F3 control clean |
| **A.0.5** decisive probe | does 2D-local clear more W than matched 1D-local? | ✅ **PASS** — W\*_2D=8 vs W\*_1D=4 (gap +4 ≥ 2); F3 control gap −2 |
| **A.0.6** Hilbert control | does the *best* 1D serialisation rescue 1D? | ✅ **PASS (gating)** — Hilbert W\*=4 = row-major; "bad serialisation" objection defeated |
| **A.0.7** capable global | is it 2D, or just iteration? | ⚠️ **INCONCLUSIVE** — fair param-matched Universal Transformer fails to train (W\*=0); deferred to Stage B |

## The honest claim Phase 12 can make now

> **At pilot scale, among matched-locality operators, a 2D layout solves an
> iterative relational task (reachability) at materially larger problem widths
> than the best 1D layout — at identical params, window, and iteration budget.
> The advantage is bandwidth-attributable (grid-graph bandwidth = min(W,H)),
> survives the space-filling-curve control, and vanishes/reverses on a
> 1D-intrinsic control. Locality is necessary for the iterative computation to be
> learnable at all (global-attention baselines do not train); given locality, the
> 2D layout wins.**

This is the project's **first performance-line positive that survives its own
controls**, after 14 structural nulls + Phase 11's pilot-negative + Phase 10b's
utility null. It does **not** contradict Phase 11: the advantage requires a
*locality constraint* that Phase 11's global attention never imposed — A.0.7's
failed global baselines reproduce Phase 11's "global gets nothing from 2D".

## What is NOT yet established (the Stage-B agenda)

1. **"2D vs iteration", strong form** — a *capable* iterative-global (Universal
   Transformer with curriculum / search-transformer recipe) for a decisive test;
   the current baselines failed to train (inconclusive, not defeated).
2. **Compute-efficiency (H3)** — matched-FLOPs curves vs a *capable* global
   (per-step O(N·9) vs O(N²) is asymptotically clear but needs a trained global).
3. **Length-generalisation** — train W ≤ W0, test W > W0 (the headline
   generalisation axis; only fixed-W threshold-crossing tested so far).
4. **Depth-generalisation (F2)** — the CA-rollout family, untested.
5. **Statistical rigour** — ≥3 seeds with Holm correction across the full sweep.
6. **Engineering-application sketch** (PLAN §8) — a 2D-native reasoning sublayer
   / layout-as-schedule, gated on the above.

## Framing guardrails (unchanged)

Hand-engineered + supervised → **structure-exploitable**, NOT emergent (Goal 1
stays closed at 14 nulls). Pilot scale, single headline family. **Not yet a
foundation-model claim** ([[foundation_model_aspiration]]); a positive Stage B
(esp. length-generalisation + a capable-global comparison) would be the evidence
that gates opening that line.

## Methodology note

The cheap-gate-before-investment discipline (Phase 11's win) held: four
pre-registered gates (~40 min total compute) located a robust positive AND its
open threat before any heavy Stage-B build. The Hilbert control and the
capacity-matched global baseline were both load-bearing — without them the result
would have been over- or under-claimed.
