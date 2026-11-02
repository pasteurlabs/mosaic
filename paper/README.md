# Mosaic paper

LaTeX source of _Mosaic: A Benchmark Suite for Differentiable Physics Solvers_, in two variants that share figures and bibliography:

- `main.tex`: NeurIPS 2026 Evaluations & Datasets Track submission (anonymized, with checklist).
- `pasteurlabs/main.tex`: preprint with author information.

## Building

```bash
latexmk -pdf -cd paper/main.tex
latexmk -pdf -cd paper/pasteurlabs/main.tex
```

A standard TeX Live installation covers both. The preprint additionally uses the `libertine`, `newtx`, `sourcesanspro`, `inconsolata` and `orcidlink` packages.

## Regenerating figures

`make_figures.py` rebuilds every data-driven figure in `figures/` from a Mosaic results directory. The results behind the published figures are archived on [Zenodo](https://zenodo.org/records/20067888):

```bash
wget -qO- 'https://zenodo.org/records/20067888/files/mosaic-results.tar?download=1' | tar x
uv run paper/make_figures.py                     # all figures
uv run paper/make_figures.py --only scaling,fd_check
```

Results are read from `$MOSAIC_RESULTS_DIR` if set, else `./mosaic-results`. Figures are written to `paper/figures/` unless `--out-dir` is given. `uv run` installs the pinned plotting dependencies declared at the top of the script. Without uv, install `matplotlib==3.10.9` and `numpy==2.4.4` and run it with `python`.

Each module in `plots/` writes one or more figures. `figures/pl-labs.png` is a static logo and has no generator.
