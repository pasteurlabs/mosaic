# Path-average gradient diagnostic evaluation

Evidence for an isolated research draft in Mosaic. These files do not modify benchmark acceptance rules.

The native solver and shared objective come from source commit `1b47f6213822dd4c4d20ca27682b306b39fe274d`. The frozen external driver is `script-v2/evaluate.py`; `protocol.json` records provenance and Slurm jobs. Full numerical results are in `full-v2/results/evaluation.json`; all declared cells and quadrature orders are retained. Plots and their generator are under `plots/`.

## Reproduce

Use a separate checkout of the source commit above. Install the dependencies recorded in `full-v2/python-environment.txt` (replace temporary local package paths with that checkout and its `mosaic/mosaic_shared` directory). Copy `script-v2/evaluate.py` into the checkout. Run on a compute node:

```bash
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python evaluate.py --mode controls --output results-controls
python evaluate.py --mode reduced --output results-reduced
python evaluate.py --mode full --output results-full
python /path/to/plots/analyze.py results-full/evaluation.json --output plots
```

The original Slurm node scripts and submission receipts preserve the exact original commands, including environment setup. Four CPU cores and 24 GB memory were requested; the local workstation was not used for numerical experiments. Artifact paths in scripts refer to the original campaign and must be adjusted when reproducing elsewhere.

## Interpretation and retained failures

Local AD versus finite differences tests a point derivative; path-average AD versus the same finite differences tests interval consistency. Analytic controls include correct derivatives, intentionally wrong derivatives, an undefined center derivative, and a discontinuous forward map. Agreement does not certify local correctness. Curated examples are not population false-pass rates.

Version 1 used a call-counting proxy rejected by the Tesseract bridge before native evaluation. Its failed preflight and source are retained. Version 2 changes only instrumentation to preserve the real Tesseract instance; controls and native preflight were rerun before the full campaign. No solver or experiment settings were changed in response to outcomes.

Prior buoyancy evidence cited by the report used a different source commit and is not part of this main-source campaign.

Timing samples follow center-forward and center-gradient warmup. First-use perturbation or FD reduction operations can still compile; all samples, including these outliers, are retained. These are in-process CPU measurements, not end-to-end container or CI estimates.
