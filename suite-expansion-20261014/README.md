# Expanded buoyancy benchmark

Numerical data: [CI run 38577109543](https://github.com/pasteurlabs/mosaic/actions/runs/38577109543), PR head `4887715c119622e48ad360c567a10cffb393919c`, tested merge `66f632ee0f679fd95e77c2fd142de1abe666b049`.

Both CPU and GPU forward suites completed. The figures combine all five solvers using the production result merge and plot pipeline on Slurm job 3082788. Plot source `1146803288f7b6155821933ef031d429bd6eaa18` differs only in small-number axis formatting and its regression test. No simulations were repeated to render these figures.

- All five solvers show decreasing coupled space/time refinement discrepancies from N=16 to N=64. The reference is the same solver at 2N, dt/2 and twice the native steps, at fixed physical time0.2. This is refinement consistency, not independent ground truth or a convergence-order estimate.
- PhiFlow loses 1.8313% of scalar mass at128steps, time0.64. The unchanged1% conservation check flags this as an anomaly. Its fields remain finite with small divergence; nonconservative semi-Lagrangian transport is distinct from instability.
- The other four solvers have absolute relative scalar mass drift below3.6e-9. Every solver has MAC divergence RMS below6e-8 and zero wall-normal storage padding in these cases. Padding is a state-format diagnostic, not an independent wall-boundary measurement.
- Refinement status uses the RMS of separately normalized velocity and scalar discrepancies, with a50% gross-disagreement threshold. Physical-law status uses scalar mass drift; energy, divergence, scalar extrema and padding are additional measured diagnostics. Buoyancy may increase kinetic energy.

Focused Slurm validation:169tests at `b7cca43`,41config/chart tests at `4887715`, and9plot/chart tests at `1146803`. Canonical unit CI at `4887715` passed728tests on each of Python3.12 and3.14 (34skips,14subtests). Full expanded benchmark status will be recorded once all suites finish; these forward results do not certify unfinished suites.
