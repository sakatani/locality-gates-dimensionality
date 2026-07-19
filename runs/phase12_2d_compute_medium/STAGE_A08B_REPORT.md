# Stage A.0.8b report — trained global on bounded-depth tasks (F4b, F5)

Pre-registration: `STAGE_A08B_PREREG.md` (committed `ead3316`, before the
run). Raw results: `a08b_global_bounded.json`. Runtime 369 s, 3 seeds.

## Results (mean over 3 seeds)

| W | 4 | 5 | 6 | 8 | 10 | 12 | 16 | 20 |
|---|---|---|---|---|----|----|----|----|
| UT on F4b | 0.98 | 0.96 | 0.93 | 0.86 | 0.81 | 0.69 | 0.54 | 0.50 |
| UT on F5  | 0.81 | 0.79 | 0.73 | 0.63 | 0.56 | 0.53 | 0.50 | 0.51 |

Reference (5-seed, existing): 2D-local F4b 0.96 → 0.93 flat to W=20; F5
0.92 → 0.89 flat.

## Pre-registered verdicts

- **F4b: TRAINS (0.958 ≥ 0.80) → "no transfer" (far mean 0.523 ≤ 0.60).**
  F4b's intrinsic depth (~K = 6 steps) is well within the UT's 16 iterations
  at every evaluated width, so the depth-budget explanation for the global
  arm's transfer failure is **eliminated on this family**: a trained global
  model with ample iteration depth still decays to chance by W = 16 while
  the 2D-local arm holds 0.93 at W = 20. This is the reviewer's requested
  isolation — the cause is the operator's lack of scale-transferable
  (local, weight-shared) structure, not iteration count.
- **F5: TRAINING FAILURE (0.779 < 0.80 gate) → no transfer verdict** per
  pre-registration. Reported as such; for completeness its unseen-size
  accuracies show the same decay to chance (0.63 → 0.50), but no claim is
  made on this family.
- **Joint "decisive-locality" label (readout 3): NOT claimed** — it
  required BOTH families to train. The paper states the F4b-based isolation
  plainly and reports F5's near-miss training failure honestly.

## Paper deltas

§7.1 gains a bounded-depth-isolation paragraph; abstract and §9
(capable-global bullet) updated to note the confound removal on F4b.
