# Locality Gates Dimensionality

Code, pre-registrations, and raw per-seed results for the paper:

> **Locality Gates Dimensionality: When (and Why) a 2D Computational Medium
> Beats Its 1D Serialization** — Yoshihiro Sakatani, 2026. (arXiv link TBA)

**TL;DR.** Whether laying a problem out in 2D helps a neural network is three
questions, not one. As an input encoding under global attention: no. As a
learned representation from implicit structure: not yet (a set-transformer
remains the wall). As a *computational medium* under a locality constraint:
yes, decisively — a bandwidth separation no serialization can close, a
scale-equivariance that turns small-instance training into large-instance
competence, and a depth taxonomy that predicts when transfer is flat. The
paper maps the boundary of that regime with pre-registered experiments and
purpose-built controls.

## Repository map

```
paper/         main.tex, refs.bib, figures + make_figures.py (regenerates
               fig1/fig2 from the raw JSON), PAPER.md (markdown source)
experiments/
  phase12_2d_compute_medium/   substrate, task families (F1/F3/F4b/F5), model
               arms (2D-local / 1D-local / Hilbert / global / UT / GNN /
               RecGNN / CA), stage runners, statistical hardening, external
               benchmark checks (Schwarzschild mazes, official CLRS-30 BFS),
               unit tests (pytest)
  phase13_learned_layout/      the learned-layout (Sinkhorn placement + CA)
               null: task, arms, pre-registered gate, 5-seed hardening
runs/          per-stage PRE-REGISTRATIONS (committed before each run),
               reports, and raw per-seed result JSONs
```

## Methodological note

Every experimental stage in `runs/` was **pre-registered**: win/null
thresholds were committed to version control *before* the corresponding run,
under a no-escape-hatch rule (a null closes the stage; no post-hoc threshold
changes). Nulls and inconclusive outcomes are reported alongside positives.

**This is verifiable in this repository's own git history**: the history is
a path-filtered export (`git-filter-repo`) of the authors' research
repository, preserving original commit timestamps and messages, so each
"pre-reg (before running)" commit visibly precedes its result commit. See
[`PREREGISTRATION.md`](PREREGISTRATION.md) for the stage-by-stage commit
map, including the stages that closed as NULL and the post-hoc
revision-phase runs (labeled as such). Caveat: git timestamps are
self-reported, not cryptographic proof; externally anchored registration
(e.g., OSF) is the stronger practice we intend going forward.

## Reproducing

Python ≥ 3.11 with PyTorch (results in the paper were produced on Apple MPS;
CPU/CUDA work too).

```bash
pip install -r requirements.txt
# easy-to-hard-data pins an old reportlab that fails to build; install without deps:
pip install --no-deps easy-to-hard-data
# optional, only for the CLRS external check:
pip install dm-clrs
```

Main entry points (run from the repo root):

```bash
python -m experiments.phase12_2d_compute_medium.calibrate      # Stage A.0
python -m experiments.phase12_2d_compute_medium.probe          # A.0.5 W* gate
python -m experiments.phase12_2d_compute_medium.harden         # A.0.6 Hilbert control
python -m experiments.phase12_2d_compute_medium.lengthgen      # B.1 length-gen
python -m experiments.phase12_2d_compute_medium.stageb3        # B.3 suite
python -m experiments.phase12_2d_compute_medium.harden_stats   # 5-seed stats + Holm
python -m experiments.phase12_2d_compute_medium.robustness     # neutral placement + RecGNN
python -m experiments.phase12_2d_compute_medium.a08_curriculum # curriculum UT (trained global)
python -m experiments.phase12_2d_compute_medium.external_clrs  # official CLRS-30 BFS check
python -m experiments.phase12_2d_compute_medium.external_maze  # Schwarzschild maze check
python -m experiments.phase13_learned_layout.gate              # learned-layout gate (null)
```

Set `PHASE12_QUICK=1` (or `PHASE13_QUICK=1`) for fast smoke runs. The maze
check downloads the easy-to-hard maze data (~1.3 GB) on first use; note the
package's tracking URL is defunct — fetch the tarballs directly from
`https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/` if the built-in
downloader fails (see `external_maze.py`).

Figures: `python paper/make_figures.py`. Paper: `tectonic paper/main.tex`.

## License

MIT (see `LICENSE`).

## Citation

BibTeX TBA (after arXiv posting).
