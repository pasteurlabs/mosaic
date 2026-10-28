# Policy 1: jax-cfd central advection with RK4

**Status:** decided, no code change.

## The case

The jax-cfd tesseract passes `convect=advection.convect_linear` and `time_stepper=time_stepping.classic_rk4` to `semi_implicit_navier_stokes` (intervention `central-advection-rk4`, PR #225). It runs only the periodic ns-grid setups (TGV, multimode); `JAX_CFD_NO_OBSTACLE` excludes the cylinder. Per PR #225, the default gave TGV errors 5 to 6 times the peer median, order 1.25 and fd_check error 5.5e-4. The change gives 1.0 times the median, order 2.00 and 9.3e-6, with a forward pass 2 to 3 times slower.

## What jax-cfd says (tag 0.2.1, `github.com/google/jax-cfd/blob/0.2.1/...`)

- **Code defaults.** `convect=None` falls back to `advect_van_leer_using_limiters` (`jax_cfd/base/equations.py` L89-92), and `time_stepper` defaults to `forward_euler` (L129). There is no hosted API documentation that I could find.
- **README.** It lists "Currently only first-order temporal discretization" and "We implement 2nd order accurate 'Van Leer' schemes". It does not mention the RK steppers, although `time_stepping.py` already ships them in 0.2.1.
- **Demos.** `notebooks/demo.ipynb` and the staggered half of `collocated_demo.ipynb` (both 2D decaying turbulence) call `semi_implicit_navier_stokes` with defaults. `collocated_demo.ipynb` has a comment showing how to switch to `convect_linear`, with no advice on when to do so. `channel_flow_demo.ipynb` uses `advect_van_leer` with forward Euler. Kolmogorov flow appears only in `spectral_forced_turbulence.ipynb`, which uses a different solver (`spectral.time_stepping.crank_nicolson_rk4`).
- **Paper.** Kochkov et al. 2021 (arXiv:2102.01010, Appendix A) state that the baseline DNS uses Van Leer advection and "explicit time integration using Euler discretization". The DNS gin config (`jax_cfd/ml/models_configs/implicit_diffusion_dns_config.gin`) uses Lax-Wendroff with a TVD limiter.
- **Tests.** `jax_cfd/base/validation_test.py` (L34-71) checks every Taylor-Green case against the analytic solution with `convect_linear`, never with Van Leer. Its RK4 case has a tolerance of 8e-7, compared with 6e-6 for forward Euler, at CFL 0.1.
- **Upstream contact.** Issue google/jax-cfd#259 (Mosaic, July 2026) asked the maintainers which advection scheme they would pick. It has no replies, and the README now says the project is no longer maintained.

## Decision (2026-10-27)

Keep central advection with RK4. Policy 1 now allows a scheme the library offers when an experienced user would choose it for the problem and a measurement shows the gain, recorded at the level of its mechanism. Both functions are public jax-cfd API, so the record is L0, necessity `choice`, tuned, with #225's before/after numbers as evidence (record updated in `jax-cfd/tesseract_config.yaml`). No code change.
