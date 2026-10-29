# Buoyancy CI plot gallery

Figures regenerated through the production merge and plotting pipeline on Slurm job 3615350, plot source `7377f84`. Numerical inputs are the eight CPU/GPU artifacts from [CI run 38577109543](https://github.com/pasteurlabs/mosaic/actions/runs/38577109543), source `4887715`, tested merge `66f632e`. No simulations or training were repeated for these plots.

The gallery includes all rendered forward, cost, gradient and optimization plots. Horizon Sweep now includes all five solvers. Solver In Loop has one 101-frame error GIF, one combined velocity/scalar error plot, and one combined validation-loss plot. The PNG beside the GIF is its poster, not an additional figure.

The combined initial-condition figure contains all three temperature fields. All 23 focused plot tests passed. The gallery has 11 PNGs, one GIF and 10 Vega-Lite sidecars. Independent OpenFOAM agreement references are still being validated and are not represented by the existing same-solver refinement plots. Temperature anomaly is the dimensionless scalar channel.

The receipt records source, numerical inputs and output hashes. Vega-Lite sidecars accompany interactive figures.

The GIF and its poster were subsequently recolored on Slurm job 3615508, source `1cb7256`: temperature-anomaly error uses a red ramp and velocity error a blue ramp, both black at zero. Five animation tests passed; data, fixed scales and all 101 frames are unchanged. `palette_validation_receipt.json` supersedes the earlier receipt for these outputs.
