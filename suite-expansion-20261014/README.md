# Expanded buoyancy benchmark

Numerical data: [CI run 38577109543](https://github.com/pasteurlabs/mosaic/actions/runs/38577109543), PR head `4887715c119622e48ad360c567a10cffb393919c`, tested merge `66f632ee0f679fd95e77c2fd142de1abe666b049`.

Both CPU and GPU forward suites completed. The figures combine all five solvers using the production result merge and plot pipeline on Slurm job 3082788. Plot source `1146803288f7b6155821933ef031d429bd6eaa18` differs only in small-number axis formatting and its regression test. No simulations were repeated to render these figures.

- All five solvers show decreasing coupled space/time refinement discrepancies from N=16 to N=64. The reference is the same solver at 2N, dt/2 and twice the native steps, at fixed physical time 0.2. This is refinement consistency, not independent ground truth or a convergence-order estimate.
- PhiFlow loses 1.8313% of scalar mass at 128 steps, time 0.64. The unchanged 1% conservation check flags this as an anomaly. Its fields remain finite with small divergence; nonconservative semi-Lagrangian transport is distinct from instability.
- The other four solvers have absolute relative scalar mass drift below 3.6e-9. Every solver has MAC divergence RMS below 6e-8 and zero wall-normal storage padding in these cases. Padding is a state-format diagnostic, not an independent wall-boundary measurement.
- Refinement status uses the RMS of separately normalized velocity and scalar discrepancies, with a 50% gross-disagreement threshold. Physical-law status uses scalar mass drift; energy, divergence, scalar extrema and padding are additional measured diagnostics. Buoyancy may increase kinetic energy.

Focused Slurm validation: 169 tests at `b7cca43`, 41 config/chart tests at `4887715`, and 9 plot/chart tests at `1146803`. Canonical unit CI at `4887715` passed 728 tests on each of Python 3.12 and 3.14 (34 skips, 14 subtests). The expanded numerical benchmark completed: **54 ok, 1 anomaly, 0 failed, 0 missing, 0 stale cells**, across all 55 solver–experiment combinations. The sole anomaly is PhiFlow scalar mass drift. Both unit CI and the benchmark workflow, including documentation rendering, passed on `4887715`. [Official report](ci/status-report.md) and [measurements](ci/summary.json) preserve the detailed evidence.

The later `1146803` commit changes only chart-axis formatting and its test. Its nine focused tests and rendering of the actual combined five-solver data passed separately on Slurm. Its automatic CI rerun is separate from the completed numerical validation recorded here.
