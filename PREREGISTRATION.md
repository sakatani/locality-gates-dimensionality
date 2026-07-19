# Pre-registration audit map

Every experimental stage in this program was pre-registered: win/null
thresholds were committed to version control **before** the corresponding
run, under a no-escape-hatch rule (a null closes the stage; no post-hoc
threshold changes). This file maps each stage's pre-registration commit to
its registered-run commit **in this repository's own git history**, so the
ordering can be verified directly with `git log`.

The history here is a path-filtered export (`git-filter-repo`) of the
authors' research repository, restricted to
`experiments/phase12_2d_compute_medium`, `experiments/phase13_learned_layout`,
`runs/phase12_2d_compute_medium`, and `runs/phase13_learned_layout`.
Original commit timestamps, messages, and file contents are preserved
verbatim; only commits touching unrelated paths were dropped.

**Honesty caveat.** Git timestamps are self-reported metadata, not
cryptographic proof of ordering — this map documents, at the same standard
claimed in the paper ("committed in-repo before the run"), what we did and
when. Externally anchored timestamps (e.g., an OSF registration) are the
stronger practice and are what we intend for future programs.

## Stage map (main program, pre-registered)

| Stage | Pre-registration commit | Registered run / verdict commit | Verdict |
|---|---|---|---|
| Program plan (bandwidth thesis, gates G1–G2) | `17a70de` 2026-06-28 | — | — |
| A.0 calibration | (within `17a70de` PLAN) | `683f111` 2026-06-28 | GREEN |
| A.0.5 threshold-crossing W\* gate | `1da4b87` 2026-06-28 | `77d9f03` 2026-06-29 | PASS |
| A.0.6 space-filling-curve (Hilbert) control | `fa261f3` 2026-06-29 | `fb19c5c` 2026-06-29 | PASS |
| A.0.7 capable iterative-global arm | `6026b81` 2026-06-29 | `f7cf06a` 2026-06-29 | INCONCLUSIVE |
| B.1 length-generalization | `aed1d44` 2026-07-04 | `0e5ff27` 2026-07-04 | PASS |
| B.2 adaptive-depth CA | `f087a81` 2026-07-04 | `fe81d98` 2026-07-04 | NULL |
| B.3 reasoning suite (F1/F4/F5) | `f79a286` 2026-07-11 | `9eb9b18` 2026-07-11 | PARTIAL (F4 control caught) |
| B.3.1 F4b scale-invariant redesign | `32ad59e` 2026-07-11 | `45b5ffa` 2026-07-11 | PASS (suite 3/3) |
| Learned-layout plan (LGRC task, gate) | `bcaed83` 2026-07-12 (machinery `cc737bf`) | `3e3a092` 2026-07-12 | NULL (2/3 gate criteria fail) |

Note the nulls: B.2 and the learned-layout gate were closed as NULL under
the no-escape-hatch rule and reported as such in the paper — the commits
above are the audit trail of that.

## Revision-phase runs (post-hoc, not pre-registered)

Commits `61437f4` through `6d2bbac` (2026-07-16 to 2026-07-19) contain the
peer-review-response work: the 5-seed statistical hardening
(`harden_stats.json`), the placement-neutral F1 and stabilized-RecGNN
robustness checks (`robustness.json`), the curriculum-trained Universal
Transformer (`a08_curriculum.json`), the 5-seed re-run of the learned-layout
gate (`gate5.json`), and the external-benchmark checks (`external_clrs.json`,
`external_maze*.json`). These were run **after** the pre-registered program
in response to reviewer concerns and are labeled as such in the paper; the
5-seed hardening re-checked the original pre-registered gate criteria
unchanged (none flipped), and the gate5 re-run confirmed the already-closed
null rather than re-litigating it.
