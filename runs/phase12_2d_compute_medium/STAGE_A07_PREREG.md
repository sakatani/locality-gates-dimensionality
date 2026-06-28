# Phase 12 — Stage A.0.7 pre-registration (capable iterative-global, G1)

The deepest remaining threat to the A.0.5/A.0.6 positive: **"is the advantage 2D,
or just iteration?"** A.0.6's `GlobalLooped` failed to train (chance everywhere),
so it was inconclusive. A.0.7 runs a **properly-built Universal Transformer**
(`univ`: pre-LN + per-iteration step embedding + grad-clip 1.0 + lr 1e-3) — a
*capable* iterative-global baseline — and compares it to 2D-local. Thresholds
fixed before `a07.py` runs.

Same regime: F1, sweep W∈{4,6,8,10,12}, matched iters/depth = 16, τ=0.80,
threshold-crossing W\*. (`univ` is O(N²)·iters → 2 seeds to bound cost; 2D 3 seeds.)

This is a **clarifying gate, not a kill gate** — the headline (2D-local vs matched
1D-local / Hilbert) already passed A.0.5 + A.0.6. A.0.7 fixes how strongly the
"2D vs iteration" and H3 (compute) claims may be framed.

## Precondition — baseline validity

`univ` must actually train: **W\*_univ ≥ 4** (it solves at least the trivial
sizes 2D-local/GNN solve at 1.00). If `univ` still floors at chance (W\*=0), the
global comparison is **inconclusive** — report it as "even a stabilised Universal
Transformer was hard to train at this budget" and fall back on the GNN
(iterative + local, which DID solve at 0.965) as the iterative reference; do NOT
claim a 2D-beats-global result from a non-training baseline.

## Framing rules (fixed before run; report whichever obtains)

Given a valid `univ`:

- **Capability:** compare `W*_2d` vs `W*_univ`.
  - `W*_2d > W*_univ` → **"2D-local beats a capable iterative-global on the task itself"** (strongest reading for 2D).
  - `W*_univ ≥ W*_2d` → **"iterated global attention matches/exceeds 2D-local on accuracy"** → the 2D case then rests on compute, not capability.
- **Compute (H3, now with a capable global):** report parameter counts and the
  per-step cost asymptotics — 2D-local is O(N·k) per step (k=9), `univ` is O(N²)
  per step. At the largest W both clear τ, state the honest H3 verdict:
  2D-local is materially cheaper per step and the gap widens with N.

In **all** cases the matched-locality 2D-vs-1D / 2D-vs-Hilbert headline is
untouched — A.0.7 only adds the global-iteration comparison.

## Verdict

A.0.7 has no PASS/FAIL kill condition; it outputs the **framing** for the "2D vs
iteration" and H3 claims that Stage B will carry. Record the obtaining case.
