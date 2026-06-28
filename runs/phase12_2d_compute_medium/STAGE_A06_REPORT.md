# Phase 12 — Stage A.0.6 adversarial hardening report

| Field | Value |
|---|---|
| Stage | A.0.6 (attack the A.0.5 PASS before Stage B; pre-reg `STAGE_A06_PREREG.md`, committed before the run, G1) |
| Verdict | ✅ **PASS on the gating check (Hilbert control)** — the headline survives its sharpest objection. Threat-2 (looped-global) **INCONCLUSIVE** (failed baseline, not a 2D win). |
| Artifacts | `experiments/phase12_2d_compute_medium/{harden.py,models.py(Local1DCurve,GlobalLooped),substrate.py(gilbert_order)}`; `harden.json`; runtime 710 s |

## Threat 1 — "you picked a bad serialisation (row-major)" → Hilbert control (GATING): PASS

Threshold-crossing W\* on F1, 3 seeds, matched depth-16 (identical conv operator
and param count; only the linearisation order differs):

| W | 4 | 6 | 8 | 10 | 12 | **W\*** |
|---|---|---|---|---|---|---|
| **2D-local** | 1.00 | 0.97 | 0.85 | 0.69 | 0.70 | **8** |
| **1D-local (row-major)** | 0.99 | 0.72 | 0.65 | 0.62 | 0.63 | **4** |
| **1D-local (space-filling/Hilbert)** | 0.93 | 0.64 | 0.63 | 0.62 | 0.64 | **4** |

> **W\*_2D − W\*_1d_hilbert = 8 − 4 = +4 ≥ 2 → PASS.** The locality-preserving
> space-filling order does **not** rescue 1D — `W*_hilbert = W*_rowmajor = 4`, and
> Hilbert is in fact *slightly worse* than row-major in the working range (W=4:
> 0.93 vs 0.99; W=6: 0.64 vs 0.72). This is the grid-graph bandwidth theorem made
> empirical: **every** 1D linearisation is bounded at Θ(min(W,H)), so no choice of
> serialisation closes the gap. The "bad serialisation" objection — the single
> most likely reviewer critique of A.0.5 — is **defeated**.

## Threat 2 — "it's iteration, not 2D" → looped-global: INCONCLUSIVE (failed baseline)

A capable iterative global baseline (`GlobalLooped`: one weight-shared Transformer
encoder layer applied 16×) was meant to test whether iterated *global* attention
matches the 2D-local operator.

| W | 4 | 6 | 8 | 10 | 12 | W\* |
|---|---|---|---|---|---|---|
| **looped-global** | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | **0** |

> **Inconclusive — NOT counted as a 2D win.** The arm sits at exactly chance
> across the *entire* sweep, **including W=4**, a size that 2D-local (1.00), GNN
> (1.00), and even row-major 1D (0.99) solve. A baseline at chance on a trivially
> solvable instance is an **optimisation failure** of the naive 16×-recurrent
> weight-shared transformer (Universal/looped transformers are unstable without
> residual scaling / ACT / careful normalisation), not evidence that iterated
> global attention is incapable. Claiming "2D beats iterated-global" from this
> would be an overclaim. **Deferred to Stage B**: a properly-implemented Universal
> Transformer baseline.

## What the "2D vs iteration?" question already resolves (partially)

All three *local* arms share the same depth-16 iterative budget, yet only the **2D
layout** clears the high W; the row-major and Hilbert 1D layouts — same iteration,
same params — cap at W\*=4. So **among matched iterative-local operators, the
differentiator is the layout dimensionality, not iteration per se.** The
GNN (iterative + local message passing) also solves it (A.0.5: 0.965), confirming
iteration+locality is sufficient. The remaining open cell is iteration+*global*
(the looped arm), unresolved here because the baseline did not train.

## Honest status after hardening

- **Defeated:** the serialisation objection (Hilbert control, the gating check).
- **Standing:** the A.0.5 headline (2D-local W\* materially exceeds the best 1D
  order at matched params/window/iteration) and the F3 capacity control.
- **Open (Stage B):** (1) a *capable* iterative-global baseline (Universal
  Transformer) for the real "2D vs iteration" and the matched-FLOPs H3 test;
  (2) F2 depth-generalisation; (3) length-generalisation; (4) ≥3 seeds + Holm;
  (5) the engineering-application sketch.

## Decision

**A.0.6 PASS (gating).** The headline survives its sharpest adversarial check.
Proceed to **Stage B** — but Stage B must carry the open items above, especially a
correctly-trained iterative-global baseline, so the "2D vs iteration" framing is
settled rather than asserted. Framing unchanged: a *structure-exploitable*
inductive-bias result on a hand-engineered substrate (Goal 1 closed), a pilot
positive, not yet a foundation-model claim.
