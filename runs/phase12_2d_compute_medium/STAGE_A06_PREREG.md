# Phase 12 — Stage A.0.6 pre-registration (adversarial hardening, G1)

A.0.5 PASSed. Before investing in Stage B, attack the two biggest threats to the
result. Thresholds fixed **before** `harden.py` runs (no-escape-hatch).

Same regime as A.0.5: F1, sweep W∈{4,6,8,10,12}, 3 seeds, matched depth-16,
wall_p 0.28, τ=0.80, threshold-crossing W\*.

## Threat 1 — "you just picked a bad serialisation (row-major)" → Hilbert control (GATING)

New arm `1d_hilbert`: the **identical** kernel-9 conv operator (same params,
verified) on the **space-filling-curve** (generalised-Hilbert) order — the
*locality-preserving* 1D linearisation — instead of row-major.

> **A.0.6 PASS (headline survives) iff `W*_2D − W*_1d_hilbert ≥ 2`.**
> If `< 2`, the best 1D order rescues 1D ⇒ the A.0.5 win was a row-major artifact
> ⇒ **DOWNGRADE** to "best-1D ≈ 2D" (a weaker / known result); do not start Stage
> B without rethinking. Also report `W*_1d_hilbert` vs `W*_1d_rowmajor` (expected
> ≈ equal — the grid-graph bandwidth theorem bounds every 1D order at Θ(min(W,H))).

## Threat 2 — "it's iteration, not 2D" → capable looped-global baseline (REPORTED)

New arm `looped`: ONE weight-shared Transformer encoder layer applied `iters=16`
times — full attention that *can* propagate over many steps (unlike A.0.5's
shallow `global`, which floored at 0.58 because it could not iterate).

> Report `W*_looped` on F1 (not a hard gate). Reading:
> - looped-global **solves** (W\* ≈ 2D): iteration is sufficient *with* global
>   attention ⇒ the headline still holds *among matched-locality arms* (2D-local
>   beats 1D-local / 1d_hilbert), but the broader claim narrows to "locality is
>   one sufficient route, not the only one"; H3 (locality buys compute) needs the
>   matched-FLOPs Stage-B test.
> - looped-global **fails** (W\* < 2D): even iterated global attention does not
>   match the 2D-local operator at this budget ⇒ strengthens the locality story.

## Verdict rule

**A.0.6 PASS** ⇔ Hilbert control holds (`W*_2D − W*_1d_hilbert ≥ 2`). PASS →
proceed to Stage B (the looped-global reading sets how strongly H3 can be
framed). **DOWNGRADE/NULL** ⇔ Hilbert rescues 1D → reconsider before any Stage B
investment.
