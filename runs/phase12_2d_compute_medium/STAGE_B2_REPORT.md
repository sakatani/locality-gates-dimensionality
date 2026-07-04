# Phase 12 — Stage B.2 report: adaptive depth (NULL, qualified)

| Field | Value |
|---|---|
| Stage | B.2 (does adaptive-depth break the depth bound? pre-reg `STAGE_B2_PREREG.md`, committed before the run, G1) |
| Verdict | ❌ **NULL on the pre-registered gate** — the adaptive CA does not decisively beat the fixed-depth model at large W. Qualified: it wins mid-range, ties at W=16, loses at W=20. |
| Artifacts | `experiments/phase12_2d_compute_medium/{stageb2.py, models.ConvCA2D/ConvCA1D}`; `stageb2.json`; 3 seeds; runtime 97 s |

## Result — train W ∈ {4,5,6}, test unseen W, CA test-iters = 3W

| arm | 8 | 10 | 12 | 16 | 20 |
|---|---|---|---|---|---|
| **ca_2d (adaptive, variable-step CA)** | 0.95 | 0.89 | 0.84 | 0.72 | 0.64 |
| **fixed_2d (B.1's 16-layer Local2D)** | 0.90 | 0.76 | 0.73 | 0.73 | 0.72 |
| **ca_1d (adaptive 1D control)** | 0.53 | 0.51 | 0.52 | 0.50 | 0.49 |

**Pre-registered gate** (all three required):
1. ca_2d@16 = 0.72 ≥ 0.70 ✓
2. ca_2d − fixed_2d @16 = **−0.01** ≥ 0.15 ✗
3. ca_2d − ca_1d @16 = +0.22 ≥ 0.20 ✓

Condition 2 fails → **NULL** (no post-hoc threshold change, G1/G2).

## Honest reading

- **The adaptive CA does NOT break the depth bound.** The pre-registered claim —
  adaptive iteration lets 2D-local generalise decisively past the fixed-depth
  model at large W — is **not** supported. ca_2d ties fixed_2d at W=16 and
  **loses at W=20** (0.64 vs 0.72), likely CA drift/wash-out over the ~60
  iterations (3W) needed at W=20 even with the gated-residual + GroupNorm
  stabilisers and variable-step training.

- **But it is not a flat null.** The CA clearly **wins in the mid-range**
  (W=10: +0.13, W=12: +0.11) and holds ≥0.70 to W=16 — adaptive depth *does*
  extend the useful range somewhat, just not decisively, and it reverses at the
  largest W. So the honest statement is "adaptive depth helps moderately but the
  fixed-depth model is competitive and more robust at the extremes."

- **Dimensionality still matters (control holds).** ca_2d − ca_1d = +0.22 at
  W=16: iterating a *1D* serialisation longer does not rescue it (ca_1d at chance
  everywhere), consistent with B.1. The 2D-vs-1D story is untouched.

- **Fixed-depth 2D is a strong, flat generaliser** (0.72–0.73 across W=8–20) —
  the more notable finding here. A 16-layer conv, trained mixed on W≤6, holds
  ~0.72 out to W=20 without adaptive iteration. The depth bound is *softer* than
  expected (the readout appears to learn size-robust heuristics), which is partly
  why the CA has little headroom to beat it.

## What this does / does not change

- **Unchanged:** the two headline positives — bandwidth (A.0.5/A.0.6) and
  length-generalisation (B.1). B.2 does not touch them; fixed-depth 2D-local
  *already* length-generalises well (B.1), and B.2 confirms it is hard to beat.
- **Not achieved:** the "unbounded computational medium" prize — a 2D operator
  that generalises to *arbitrarily* larger problems via adaptive iteration. At
  pilot budget, a standard-capacity variable-step neural-CA does not deliver it.

## Honest caveats & why not to chase it further now

- Neural CAs for algorithmic tasks are a known-hard training problem; a stronger
  recipe (larger shared block for capacity parity with the 149k-param fixed
  baseline, many more epochs, an explicit fixed-point / pool loss, or learned
  adaptive-halt ACT) might yet break the bound. That is **genuine future work**,
  not a pilot deliverable — and pushing further risks motivated tuning against a
  pre-registered gate that already returned NULL. Per the project's
  no-escape-hatch discipline, B.2 is recorded as **NULL** and left there.

## Decision

**B.2 NULL (qualified).** Adaptive depth does not decisively break the depth
bound at pilot scale; fixed-depth 2D-local remains the strongest generaliser at
large W. The Phase 12 headline stands on A.0.5/A.0.6 (bandwidth) + B.1
(length-generalisation), with the depth bound recorded as a real, unbroken
limitation. Remaining Stage-B items (capable-global comparison, F2 depth-gen,
Holm, application sketch) are independent of this null.
