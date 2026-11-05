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

Results are read from `$MOSAIC_RESULTS_DIR` if set, else `./mosaic-results`. `uv run` installs the pinned dependencies declared at the top of each script. Without uv, install `matplotlib==3.10.9`, `numpy==2.4.4` and `pyyaml==6.0.3` and run the scripts with `python`. Both scripts also read the solver metadata (differentiation strategy, GPU use, intervention records) from `mosaic/tesseracts/*/*/tesseract_config.yaml` in this repository.

Each module in `plots/` writes one or more figures to `figures/` (or `--out-dir`). `pl-labs.png` (logo), `recovery_overview.pdf`, `recovery_adam_proj.pdf` and `topopt_overview.pdf` are static and have no generator.

`make_tables.py` writes these files to `tables/` (or `--out-dir`) and prints the key numbers they contain:

| File                  | Label                                      | Contents                                                                                                                        |
| --------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `merged.tex`          | `tab:merged`                               | Wall times, errors, convergence, gradient support and interventions per solver and domain                                       |
| `gradient_checks.tex` | `tab:gradchecks`                           | F2 gradient checks: finite differences, analytic gradient, dJ/dν                                                                |
| `vjp_ratios.tex`      | `tab:app:ratios`                           | VJP/forward time ratio at the Table 3 sizes and over each cost sweep                                                            |
| `variance.tex`        | `tab:app:variance`                         | Coefficient of variation of the timed trials per problem                                                                        |
| `outcomes16gb.tex`    | `tab:app:outcomes`, `tab:app:a100failures` | Configurations that failed on 16 GB GPUs and their outcome in the results, then the failure points of the results' fluid sweeps |

`outcomes16gb.tex` compares against two runs on 16 GB GPUs: a V100 run ([Zenodo](https://zenodo.org/records/20067888), `--v100-results`) and a T4 CI run (`--t4-results`). Without both flags it is skipped. `--extra-results` adds the extended mesh cost sweeps (`cost/spatial_cost_large`, `cost/vjp_cost_large/by_N`), which fill in mesh sizes beyond the main sweep. The convergence and gradient-support columns of `merged.tex` come from the `mosaic status` JSON, read from `status.json` next to the results directory unless `--status` is given.
