# Buoyancy CI plot gallery

Figures regenerated through the production merge and plotting pipeline on Slurm job 3615077, plot source `695a70a33ac51a3d23b8b5e70bbf0dafe3422dbc`. Numerical inputs are the eight CPU/GPU artifacts from [CI run 38577109543](https://github.com/pasteurlabs/mosaic/actions/runs/38577109543), source `4887715`, tested merge `66f632e`. No simulations or training were repeated for these plots.

The gallery includes all rendered forward, cost, gradient and optimization plots. Horizon Sweep now includes all five solvers. Solver In Loop has one 101-frame error GIF, one combined velocity/scalar error plot, and one combined validation-loss plot. The PNG beside the GIF is its poster, not an additional figure.

Initial-condition figures are pending a fix to scalar-field rendering. Independent OpenFOAM agreement references are still being validated and are not represented by the existing same-solver refinement plots. Temperature anomaly is the dimensionless scalar channel.

The receipt records source, numerical inputs and output hashes. Vega-Lite sidecars accompany interactive figures.
