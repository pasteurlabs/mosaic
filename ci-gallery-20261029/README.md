# Current buoyancy review gallery

This directory contains the figures embedded in PR #211. PNGs and Vega-Lite sidecars share data; the GIF's PNG is its final-frame poster.

## Numerical sources

- Diffusion, same-solver refinement, physical laws, cost, original corrector errors/validation and Jacobian arrays: [canonical CI 38577109543](https://github.com/pasteurlabs/mosaic/actions/runs/38577109543), source `4887715`, tested merge `66f632e`. Original training uses velocity-only width-8 Adam at LR .003. These are not results from the later optimizer sweep.
- Independent OpenFOAM agreement and expanded finite differences/horizons: targeted native validations on October 29. All six solvers passed agreement against the refined OpenFOAM reference; all five differentiable solvers passed six-epsilon FD and horizon checks. OpenFOAM is excluded from derivatives. `targeted_admission_summary.json` records job IDs, bridges, sources and environments. This is not a completed new canonical container campaign.

## Current displays

- The GIF has one selected corrected JAX-CFD temperature/speed pair at the top, explicitly for orientation rather than common ground truth. All five solver rows below show only their uncorrected/corrected errors against their own fine reference. Red represents temperature and blue velocity. Error maxima differ by row, but paired arms share each row's scales fixed over all 101 frames; compare physical limits across rows. Source `116cd4a`, seven tests and actual saved-array rendering passed on Slurm 3616995; see `orientation_validation_receipt.json`.
- The Jacobian figure marks expected rank 113/192, shades the structural nullspace tail, and separates full and non-null singular-value ratios using relative cutoff 1e-6. Ratios 1.15–1.23 describe the short resting-state packed-coordinate map outside its numerical kernel, not long-rollout conditioning. Source `dac275a`, 24 tests and render passed on Slurm 3616275; see `diagnostic_validation_receipt.json`.
- The dense FD graph displays mean relative error across directions; admission uses best-epsilon median and cosine. Near-zero directional derivatives can inflate the mean, notably for JAX-CFD. Every epsilon is retained. The horizon chart includes all five differentiable solvers. Narrow log-axis rendering passed its dedicated regression.
- Initial conditions are temperature anomalies, rendered through the configured scalar projection. Other review-gallery changes passed 23 tests at `7377f84`.

`current_asset_hashes.json` identifies current bytes. Older receipts retained here document superseded displays; the specific receipts above supersede them for updated outputs. No simulations were repeated solely to restyle figures.
