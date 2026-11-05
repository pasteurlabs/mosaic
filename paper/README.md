# Mosaic paper

LaTeX source of _Mosaic: A Benchmark Suite for Differentiable Physics Solvers_, in two variants built from the same content:

- `neurips.tex`: NeurIPS 2026 Evaluations & Datasets Track camera-ready version, with checklist.
- `preprint.tex`: preprint, styled by `pasteurlabs.sty`.

Both wrap the shared `abstract.tex`, `body.tex` and `appendix.tex`, and set the template, title block, bibliography and float placement themselves. If content has to differ between the variants, for example to meet the NeurIPS page limit, wrap it in `\ifpreprint ... \else ... \fi`. The NeurIPS wrapper appends the period to run-in `\paragraph` titles, so write them without one.

## Building

```bash
latexmk -pdf -cd paper/neurips.tex
latexmk -pdf -cd paper/preprint.tex
```

A standard TeX Live installation covers both. The preprint additionally uses the `libertine`, `newtx`, `sourcesanspro`, `inconsolata` and `orcidlink` packages.

## Regenerating figures and tables

`make_figures.py` rebuilds every data-driven figure in `figures/`, and `make_tables.py` every data-driven table in `tables/`, from a Mosaic results directory (the output of `mosaic run`):

```bash
export MOSAIC_RESULTS_DIR=path/to/a100-run/results
uv run paper/make_figures.py                     # all figures
uv run paper/make_figures.py --only scaling,fd_check
uv run paper/make_tables.py --v100-results path/to/v100-results --t4-results path/to/t4-results
```

Results are read from `$MOSAIC_RESULTS_DIR` if set, else `./mosaic-results`. `uv run` installs the pinned dependencies declared at the top of each script. Without uv, install `matplotlib==3.10.9`, `numpy==2.4.4` and `pyyaml==6.0.3` and run the scripts with `python`. Both scripts also read the solver metadata (differentiation strategy, GPU use) from `mosaic/tesseracts/*/*/tesseract_config.yaml` in this repository.

Each module in `plots/` writes one or more figures to `figures/` (or `--out-dir`). `pl-labs.png` (logo), `recovery_overview.pdf`, `recovery_adam_proj.pdf`, `topopt_overview.pdf` and `appendix_ucurves_f3.pdf` are static and have no generator. `figures/domain_*.pdf` are high-resolution raster exports of the schematics in `docs/figures/`, which `docs/figures/make_domain_figures.py` draws.

`make_tables.py` writes these files to `tables/` (or `--out-dir`) and prints the key numbers they contain:

| File                  | Label              | Contents                                                                                       |
| --------------------- | ------------------ | ---------------------------------------------------------------------------------------------- |
| `merged.tex`          | `tab:merged`       | Wall times, errors, convergence, gradient support and integration effort per solver and domain |
| `gradient_checks.tex` | `tab:gradchecks`   | F2 gradient checks: finite differences, analytic gradient, dJ/dν                               |
| `vjp_ratios.tex`      | `tab:app:ratios`   | VJP/forward time ratio at the Table 3 sizes and over each cost sweep                           |
| `variance.tex`        | `tab:app:variance` | Coefficient of variation of the timed trials per problem                                       |
| `outcomes16gb.tex`    | `tab:app:outcomes` | Configurations that failed on 16 GB GPUs and their outcome in the results                      |

`outcomes16gb.tex` also holds `tab:app:a100failures`, the failure points of the A100 fluid sweeps. It compares against two runs on 16 GB GPUs: a V100 run ([Zenodo](https://zenodo.org/records/20067888), `--v100-results`) and a T4 CI run (`--t4-results`). Without both flags it is skipped. `--extra-results` adds the extended mesh cost sweeps (`cost/spatial_cost_large`, `cost/vjp_cost_large/by_N`), which fill in mesh sizes beyond the main sweep. The convergence and gradient-support columns of `merged.tex` come from the `mosaic status` JSON, read from `status.json` next to the results directory unless `--status` is given. The integration-effort columns of `merged.tex` (bespoke lines of code per wrapper) come from `count_loc.py`, described below.

## Counting lines of code

`count_loc.py` counts the bespoke lines of every Tesseract wrapper in `mosaic/tesseracts/` from the `mosaic:<category>` tags in its code. `make_tables.py` imports it for the integration-effort columns of `merged.tex`, and running it on its own prints the counts per wrapper:

```bash
uv run paper/count_loc.py
```

In `tesseract_api.py`, a tag sits on the signature of each top-level function or class (`def build_mesh(...):  # mosaic:init`) and covers its whole body. In Julia and C/C++/CUDA files, a comment marker (`# mosaic:<category>` or `// mosaic:<category>`) covers every line up to the next marker. The schema classes, `apply`, `abstract_eval`, the `vjp_jit` and `jvp_jit` helpers, imports and other module-level statements are glue and are not counted.

| Tag       | Contents                                            | Column |
| --------- | --------------------------------------------------- | ------ |
| `io`      | Data exchange with external solver processes        | IO     |
| `init`    | Solver initialization and mesh setup                | Conf.  |
| `physics` | Discretization, time stepping, boundary conditions  | Num.   |
| `grad`    | Gradient code (VJPs, adjoint solves, checkpointing) | Num.   |
| `util`    | Helpers shared by the above                         | Num.   |

The script also prints the `grad` lines on their own (they are part of Num.), which the text of Section 4.1 quotes, and an `unknown` column for untagged code, which counts as Num. and should be 0.
