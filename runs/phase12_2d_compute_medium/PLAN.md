# Phase 12 — PLAN (pre-registration)
## Goal 3 reframe: 2D as a *computational medium* — the bandwidth thesis

| Field | Value |
|---|---|
| Phase | 12 (Goal 3, axis ②: "is a 2D **field + spatially-local operators** a better computational substrate than a 1D serialization, attributable to the *dimensionality × locality* interaction?") |
| Status | 🟢 OPEN — pre-registration draft (Rev 1) |
| Lineage | Follows Phase 11 (axis ①, input-encoding — CLOSED pilot-negative). This is a **different axis**: Phase 11 used *global* set-attention, so a locality constraint never bit. ② tests the regime Phase 11 structurally could not reach. |
| Substrate / "language" | **layout-as-schedule (LaS)** — provisional name. NOT v2.x. A 2D placement of a relational/computational structure where *spatial adjacency = one-step interaction*; meaning = fixed point of iterated local updates. |
| Worktree | `2d-lang-phase12`, branch `phase12` (off `main` @ `a45f9f8`). Runs from main `.venv` via `PYTHONPATH=<worktree>` (worktree has no torch — same as Phase 11). |
| Provenance guard | Hand-engineered + supervised. A positive = "2D structure is **computationally exploitable** as a scaling/length-generalization prior", NOT emergence. Goal 1 stays CLOSED (14 structural nulls). |

---

## 1. The question and the thesis

Phase 11 killed exactly one hypothesis: *2D as an input encoding* gives no reasoning advantage over a set baseline, because a set of `(x,y)` tokens is computationally "a set with 2-dim positional features", and set-attention is already translation-robust. The robust effect was `set ≫ sequence` (known, Sun et al. 2024), not 2D.

Phase 11 never imposed a **locality constraint** (it used global attention everywhere), so *dimensionality could not matter*: when every token sees every other token, the layout of the tokens is irrelevant.

**Thesis ② (bandwidth thesis).** When the model's state is a 2D field and its update operator is **spatially local** (fixed receptive field, weight-shared across positions — conv / windowed-attention / cellular-automaton update), a class of relational computations becomes:
- **linear-compute** in problem size (per-step cost = #cells × fixed window), and
- **length/depth-generalizing** at fixed per-cell compute,

*for problem structures whose dependency graph has low **2D bandwidth** but high **1D bandwidth*** — i.e. where every one-step interaction is spatially adjacent in 2D, but no 1D linearization keeps all one-step interactions adjacent.

This is the project's **P2 (multi-axis independence)** recast as a *computational* property. The governing quantity is the **bandwidth of the dependency graph under the best layout**: small in 2D (look at neighbors → advance one step), large in any 1D order (must reach far to advance). The **gap between 1D-local and 2D-local accuracy, and its growth with problem size, is the 2D-irreducibility signal.**

### Why this is NOT a re-run of the #5 null / Phase 11
- #5 / Phase 11 measured **representational equivalence on translation-OOD accuracy** under *global* attention. ② measures **compute-efficiency and length-generalization under a locality constraint**. Orthogonal axis, orthogonal metric.
- The win condition is *not* "2D beats 1D on accuracy at fixed size" (that is the trap that keeps tying). It is "the 2D-vs-1D accuracy gap **grows with problem size W**, at matched locality budget" — a *scaling law*, not a level.

---

## 2. Honesty constraints (mirror Phase 11 D1–D4)

- **H1 (same information, bias-only difference).** Every arm consumes the *same lossless* representation (the same cells, the same `(x,y)`, the same adjacency). Arms differ **only** in the locality structure of their operator. A 2D win can therefore only be an inductive-bias / generalization win, never extra information. (= Phase 11 D1.)
- **H2 (matched locality budget).** The decisive comparison `2D-local vs 1D-local` uses the **same receptive-field size / parameter count / depth**. If 2D wins, it is the *dimensionality of the layout*, not a bigger window. Without this control the result is the trivial "locality helps".
- **H3 (matched compute for the efficiency claim).** The `2D-local vs global` comparison reports accuracy at **matched FLOPs/attention-cost** and the compute-to-threshold. The global arm is literally Phase 11's regime — including it lets us *show the gap appears only under locality* (global should tie 2D-local on small W and lose only on compute).
- **H4 (refutation controls).** A headline positive requires the gap to appear **only on low-2D-bandwidth families** and **vanish on the 1D-intrinsic control** (where 1D-local should match/beat 2D-local). If 2D-local wins *even on the 1D-control*, the effect is generic capacity, not 2D-bandwidth → refuted. (= Phase 11 D4 graph-bias refutation, transposed.)

---

## 3. Arms (the model comparison set)

All arms: same tokens/cells in, supervised readout, standard components, no novel attention. They differ only in *connectivity of the operator*.

| Arm | Operator connectivity | Isolates |
|---|---|---|
| **2D-local** | conv / windowed-attention over a `k×k` window in `(x,y)` space | the thesis arm |
| **1D-local** | same window budget over the **best 1D serialization** (row-major) — `±r` index window | **dimensionality** (H2) — the load-bearing control |
| **global** | full attention (Phase 11's regime), `O(n²)` | **compute-efficiency** (H3); should erase the gap on accuracy, lose on compute |
| **GNN-adjacency** | message-passing on the *true* edge set (handed the graph) | **geometry-vs-adjacency diagnostic** — does 2D *proximity alone* (no edge list) match being handed the graph? (interpretive, not gating) |

Headline 2D claim = **`2D-local − 1D-local`**, its **growth with W**, and **`2D-local ≈/> global` at materially lower compute**. The GNN arm is documented but not part of the gate (a grid's adjacency *is* the 2D schedule, so 2D-local≈GNN would *confirm* "proximity = schedule", not refute it).

---

## 4. Task families (the bandwidth knob)

Each family is **symbolic/token-based** (cells = entities + adjacency relations), NOT pixels — to stay in the 2D-*language* frame and keep the 1D-serialization comparison fair (the 1D arm gets the identical tokens in row-major order). A free parameter **W** decouples 1D-bandwidth from 2D-bandwidth.

- **F1 — 2D multi-hop propagation (connectivity / shortest-hop / flood-fill).** `W×H` cell grid, walls, source `S`, target `T`. 4-neighbor adjacency → 2D-bandwidth `O(1)`/step, needs `O(W+H)` *local* steps. Row-major 1D layout → vertical neighbors are `W` apart → **1D-bandwidth = W**. **Knob: W.** Prediction: 2D-local flat in W; 1D-local collapses once `W > r`; gap grows with W; global solves but at `O((WH)²)`.
- **F2 — 2D local-rule rollout (cellular automaton / diffusion).** Predict state after `T` steps of a local 2D rule. 2D-local matches the generative process; 1D-local cannot keep the 2D neighborhood local. **Knobs: W (width), T (depth)** → also tests *depth*-generalization (train `T`, test `T'>T`).
- **F3 — 1D-intrinsic control (refutation).** A task whose dependency is intrinsically along one axis (running aggregate / scan along rows). Here **1D-local should match/beat 2D-local**; global ties both. If 2D-local wins here → generic-capacity confound → CLOSE.

(Calibration may add a 4th *high-bandwidth-in-every-layout* control — a random graph with no good 2D embedding — to check 2D-local does NOT beat GNN there. Optional; not gating.)

---

## 5. Metrics

- **Gap curve:** `acc(2D-local) − acc(1D-local)` as a function of **W** (and **T** for F2). Headline.
- **Length/depth generalization:** train on `W ≤ W0` / `T ≤ T0`, test on larger. The generalization axis (Phase 11's tested axis was translation; this is the untested scaling axis).
- **Compute-matched efficiency:** accuracy vs FLOPs/attention-cost for `2D-local` vs `global`; compute-to-threshold.
- **Difficulty gate (calibration):** reference (global) model accuracy in `[0.55, 0.90]` per family/seed (= Phase 11 band).

---

## 6. Pre-registered win / null conditions + no-escape-hatch

**PASS (headline)** iff, on F1 (and F2):
1. `2D-local ≫ 1D-local` at **matched locality budget** (H2), AND
2. the gap **increases with W** (a scaling signal, not a fixed-size bump), AND
3. `2D-local ≈/> global` at **materially lower compute** (H3),
WHILE on F3 control:
4. `2D-local` does **NOT** beat `1D-local` (no generic-capacity confound, H4).

**NULL** iff any of: gap `≈ 0`; gap does **not** grow with W; or the F3 control shows 2D-local winning.

**No-escape-hatch (G-rules, = Phase 11 G3 discipline):**
- **G1.** Fix the numeric thresholds for "≫", "grows", "materially lower compute" **before** running Stage A.0.5 (written into `STAGE_A0_REPORT.md`).
- **G2.** If A.0.5 is NULL → **CLOSE pilot-negative, NO Stage B.** Do not run a heavier version of the same axis.
- **G3.** If the F3 control is violated → CLOSE (confound).

---

## 7. Stage structure (cheap-gate-before-investment, per Phase 11)

- **Stage A.0 — calibration (GREEN gate).** Families valid & non-vacuous: difficulty band on the global arm; **verify the knob decouples** — measure 1D vs 2D bandwidth analytically AND confirm 1D-local degrades with W at small scale, 2D-local does not. Confirm F3 behaves as a 1D-intrinsic control.
- **Stage A.0.5 — the decisive cheap probe (~1–2 days).** Small models, all 4 arms, **F1 + F3**, sweep `W ∈ {small, mid, large}`. Read the **gap-vs-W curve** and the compute-matched point. This is the gate. (Phase 11 reached a confident read in ~1 day this way.)
- **Stage B — formal (gated on A.0.5 PASS).** Full family set incl. F2 depth-gen, matched-compute curves, ≥3 seeds, Holm correction, and the **engineering-application demo** (§8).

---

## 8. Engineering-application implications (gated on a positive — not claimed now)

If ② holds, the application surface is:
- **2D-native reasoning sublayer** — a local-2D computational module that length-generalizes algorithmic/program-execution tasks; insertable into an LM (the bridge to the [[foundation_model_aspiration]]).
- **layout-as-schedule compiler** — compute graph → 2D placement → local-hardware schedule (systolic / spatial accelerator); a readable, editable intermediate representation.
These are **hypotheses pending the probe**; stated here only to fix the target the probe is a gate for.

---

## 9. Out of scope / provenance

- Emergence (Goal 1) — out; closed at 14 nulls. A positive here is "structure-exploitable", not "learned/emergent".
- Foundation-scale pretraining, 2D chain-of-thought with 2D intermediate states, natural-data→2D — Phase 13+ (per Phase 11 PLAN's deferral list), gated on a positive ② pilot.
- Pixel/vision framing — explicitly avoided; families are symbolic to keep the 1D-serialization comparison fair and stay in the 2D-language frame.

## 10. Literature anchors (verify during build)
- Larkin & Simon 1987, *Why a Diagram is (Sometimes) Worth Ten Thousand Words* — diagrams reduce search via locality; the origin of ②'s complexity argument.
- Mordvintsev et al. 2020, *Growing Neural Cellular Automata* — local-2D update as a learned computational medium.
- Delétang et al. 2023, *Neural Networks and the Chomsky Hierarchy* — how to measure length-generalization cleanly.
- Graph **bandwidth** / **systolic arrays** — the layout-as-schedule formalism.
