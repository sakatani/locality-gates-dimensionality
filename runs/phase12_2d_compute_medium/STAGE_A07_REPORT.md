# Phase 12 — Stage A.0.7 report (capable iterative-global)

| Field | Value |
|---|---|
| Stage | A.0.7 (clarifying gate: "is the A.0.5/A.0.6 advantage 2D or just iteration?"; pre-reg `STAGE_A07_PREREG.md`, committed before the run, G1) |
| Outcome | ⚠️ **INCONCLUSIVE per the pre-registered rule** (the iterative-global baseline did not reach the W\*≥4 validity bar) — but the *converging* evidence across two fair attempts supports the locality reading. The headline (A.0.5/A.0.6) is untouched. |
| Artifacts | `experiments/phase12_2d_compute_medium/a07.py`, `models.UniversalTransformer`; `a07.json`; runtime 1485 s |

## Result

A **fair, capacity-matched, stabilised** Universal Transformer (pre-LN +
per-iteration step embedding + grad-clip 1.0 + lr 1e-3 + 32 epochs; hidden 96 →
86.6k params, vs 2D-local's 149k) vs 2D-local on F1 (W\*, τ=0.80):

| W | 4 | 6 | 8 | 10 | 12 | **W\*** | params | per-step |
|---|---|---|---|---|---|---|---|---|
| **univ (iterative-global)** | 0.70 | 0.56 | 0.53 | 0.50 | 0.50 | **0** | 86.6k | O(N²) |
| **2D-local** | 1.00 | 0.97 | 0.85 | 0.69 | 0.70 | **8** | 149k | O(N·9) |

## Honest reading (per the pre-registered framing)

- **Formally INCONCLUSIVE.** The pre-reg required `W*_univ ≥ 4` (the baseline must
  solve at least the trivial sizes) before a clean 2D-vs-global comparison. The
  Universal Transformer reached only **0.70 at W=4** — learning, but not to
  threshold — so `W*_univ = 0`. Per G1 I do **not** claim "2D-local beats a
  capable iterative-global"; the baseline did not clear the validity bar.

- **But the evidence converges.** This is the **second** fair attempt at an
  iterative-global baseline: the naive `GlobalLooped` (A.0.6) sat at exactly
  chance; this stabilised, param-matched `univ` learns *weakly* (0.70 → chance by
  W=10). Both fail to train to threshold, while **both iterative-LOCAL arms solve
  the task easily** (2D-local 0.935, GNN 0.965). The differentiator among
  *trainable* arms is the **locality structure**, not iteration per se — every
  arm here has 16 iterations of compute; only the local ones learn the iterative
  propagation. This is consistent with the literature on transformers failing to
  learn graph search / connectivity robustly.

- **Caveat (do not overclaim).** This is a **training-difficulty** observation at
  pilot budget, **not** a proof that global attention is *incapable* of
  reachability. A constant-lr Universal Transformer (per the pre-reg recipe) was
  tested; warmup schedules, curricula, and known search-transformer techniques
  were not. A definitive "2D vs iteration" capability claim is a **Stage-B** item.

- **Compute (H3) stands regardless.** Even where it learns, `univ` costs O(N²) per
  step vs 2D-local's O(N·9); the per-step gap widens with N. So on the compute
  axis 2D-local is unambiguously favourable.

## What this settles for the "2D vs iteration?" question

| Operator | iteration | locality | learns the task? |
|---|---|---|---|
| 2D-local | ✓ (depth 16) | ✓ 2D window | **yes** (0.94) |
| 1D-local / Hilbert | ✓ (depth 16) | ✓ 1D window | only small W (W\*=4) |
| GNN | ✓ | ✓ true graph | **yes** (0.97) |
| looped-global | ✓ (16 iters) | ✗ global | no (chance) |
| univ-global | ✓ (16 iters) | ✗ global | weak (W\*=0) |

**Locality is necessary for the iterative computation to be learnable here; given
locality, the 2D layout beats the best 1D layout (the A.0.5/A.0.6 headline).** The
strong form "2D beats a *capable* global" is unproven only because a capable
global could not be trained — an open Stage-B threat, not a defeated one.

## Decision

A.0.7 does not change the **headline** (2D-local ≫ matched 1D-local / Hilbert,
bandwidth-attributed, F3-controlled). It leaves the iterative-global comparison
**open** (inconclusive baseline). Stage B must either (a) get a capable
iterative-global trained (curriculum / search-transformer recipe) for a decisive
"2D vs iteration" + matched-FLOPs H3 test, or (b) frame around the GNN as the
working iterative reference and the locality result. Until then the honest
Phase-12 claim is the locality-conditioned one, not a blanket "2D > global".
