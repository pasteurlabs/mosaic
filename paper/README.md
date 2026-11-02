# Mosaic paper

LaTeX source of _Mosaic: A Benchmark Suite for Differentiable Physics Solvers_, in two variants built from the same content:

- `neurips.tex`: NeurIPS 2026 Evaluations & Datasets Track submission (anonymized, with checklist).
- `preprint.tex`: preprint with author information, styled by `pasteurlabs.sty`.

Both wrap the shared `abstract.tex`, `body.tex` and `appendix.tex`, and set the template, title block, bibliography and float placement themselves. Content that differs between the variants is wrapped in `\ifpreprint ... \else ... \fi`. The NeurIPS wrapper appends the period to run-in `\paragraph` titles, so write them without one.

## Building

```bash
latexmk -pdf -cd paper/neurips.tex
latexmk -pdf -cd paper/preprint.tex
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
