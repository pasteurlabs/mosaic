# Policy 1: Exponax dealiasing

**Status:** decided 2026-10-27: back to the default 2/3 rule, references regenerated (#280).

## The case

The Exponax tesseract passes `dealiasing_fraction=1.0` to its Navier–Stokes steppers (intervention `no-dealiasing`, "to match other solvers", present since c47498a with no measurement behind it). The library default is 2/3. Mosaic generates the ns-3d-grid `forward/agreement` and `forward/baseline` references by running this same tesseract at N = 64 and truncating spectrally (`benchmarks/cli/reference.py`, `_generate_converged`), so the setting also enters every peer's forward score. Strictly, 1.0 does not switch dealiasing off: it still zeros the Nyquist mode in the nonlinear term (`nonlin_fun/_base.py` L59-71).

## What Exponax says (v0.2.0, `github.com/Ceyron/exponax/blob/v0.2.0/...`)

- **API docs.** Every Navier–Stokes stepper documents "The default 2/3 corresponds to Orszag's 2/3 rule. To fully eliminate aliasing, use 1/2" (`exponax/stepper/_navier_stokes.py` L82-85, rendered at fkoehler.site/exponax/api/stepper/nonlinear/navier_stokes/).
- **Tutorials.** Only `docs/examples/creating_your_own_solvers_1d.ipynb` discusses the choice. For strongly nonlinear terms it suggests 1/2 as "a good first guess". None of the 2D or 3D Navier–Stokes examples (`solver_showcase_2d.ipynb`, `kolmogorov_flow_3d_with_spectral_analysis.ipynb`) changes the default, and I found no example that lowers dealiasing.
- **Paper.** APEBench (arXiv:2411.00180, Appendix B, "Dealiasing for the Nonlinear Part") motivates dealiasing by aliases "potentially corrupting the solution and leading to instabilities". Every APEBench scenario keeps 2/3 (`Ceyron/apebench`, `scenarios/*`).
- **Smooth, well-resolved flows.** I found no guidance on these.

## What policy 1 implies

The documentation would never tell a user to turn dealiasing off. It only ever advises stronger dealiasing, so the default stays. The record's `inaccurate-forward` consequence also has the wrong sign. On these smooth setups, turning dealiasing off makes Exponax more accurate (below), so this change engineers away an artefact of the method, which rule 3 forbids.

## Measured effect

Exponax 0.2.0, JAX 0.10.0 on CPU, in float32 like the tesseract, compared against a float64 N = 128 run with a time step 4 to 8 times smaller. The scripts and `results.json` are in `diag/exponax/` (next to this note). My 1.0 runs reproduce the checked-in references to 7e-7.

| Exponax error     | 1.0 (now)    | 2/3 (default) | Peer range       |
| ----------------- | ------------ | ------------- | ---------------- |
| agreement, N = 16 | 5e-6 to 9e-6 | 5e-4 to 9e-4  | 3e-3 to 4e-2     |
| baseline, N = 8   | 1.8e-2       | 0.24          | 0.08 to 0.17     |
| baseline, N = 16  | 3.5e-4       | 8.9e-3        | 0.016 to 0.048   |
| baseline, N = 32  | 2.8e-5       | 7.1e-5        | 4.5e-3 to 1.2e-2 |

- **Reference.** References built with 1.0 and 2/3 differ by 5e-7 to 6e-7. Both sit within 1.2e-5 of the float64 run. Every peer's error changes only in the fifth significant digit or beyond. At N = 64 the TGV holds about 1e-12 of its energy above |k| = 20, so aliasing has nothing to act on.
- **Exponax's own rows.** The loss is effective resolution. Exponax keeps |k| ≤ 20 of 32 at N = 64, ≤ 4 at N = 16 and ≤ 1 at N = 8. That is one mode fewer than the usual statement of the rule (|k| < N/3). At N = 8 the TGV nonlinear term vanishes entirely, and the run matches pure viscous decay to 4e-7. Exponax drops from best to worst at N = 8 and stays best elsewhere.
- **Long horizon** (horizon_sweep setup, N = 20, ν = 0.001). With 1.0, 42% of the energy sits at max|k| ≥ 6 at t = 16, and the kinetic energy is half that of the 2/3 run. I did not compute a resolved solution, so I cannot say which run is closer.

## Measured on the full ns-3d-grid suite (A100, `main` vs 2/3)

Every cell stays `ok`. Forward errors grow (agreement 7.7e-6 → 7.9e-4, baseline 3.5e-4 → 8.9e-3; still the lowest of all solvers). The gradient side improves: `fd_check` 2.6e-4 → 1.2e-4, `jacobian_svd_steps40` condition number 10.0 → 4.6, `recovery` final IC error 1.6e-2 → 1.0e-4, and `horizon_sweep_limits` gradient norms stay bounded where 1.0 blows up (2.8e6 vs 409 at 320 steps). The references move by 5e-7 to 6e-7.

## Decision

Revert to 2/3. Under the amended policy 1 a non-default scheme needs a measured gain an experienced user would recognise; turning dealiasing off only helps the forward error of a smooth initial condition and costs gradient quality and long-horizon stability, which is the standard reason spectral codes dealias.
