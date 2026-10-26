# Policy 1: jax-cfd central advection with RK4

## The case

The jax-cfd tesseract passes `convect=advection.convect_linear` and `time_stepper=time_stepping.classic_rk4` to `semi_implicit_navier_stokes` (intervention `central-advection-rk4`, PR #225). It runs only the periodic ns-grid setups (TGV, multimode); `JAX_CFD_NO_OBSTACLE` excludes the cylinder. Per PR #225, the default gave TGV errors 5 to 6 times the peer median, order 1.25 and fd_check error 5.5e-4. The change gives 1.0 times the median, order 2.00 and 9.3e-6, with a forward pass 2 to 3 times slower.

## What jax-cfd says (tag 0.2.1, `github.com/google/jax-cfd/blob/0.2.1/...`)

- **Code defaults.** `convect=None` falls back to `advect_van_leer_using_limiters` (`jax_cfd/base/equations.py` L89-92), and `time_stepper` defaults to `forward_euler` (L129). There is no hosted API documentation that I could find.
- **README.** It lists "Currently only first-order temporal discretization" and "We implement 2nd order accurate 'Van Leer' schemes". It does not mention the RK steppers, although `time_stepping.py` already ships them in 0.2.1.
- **Demos.** `notebooks/demo.ipynb` and the staggered half of `collocated_demo.ipynb` (both 2D decaying turbulence) call `semi_implicit_navier_stokes` with defaults. `collocated_demo.ipynb` has a comment showing how to switch to `convect_linear`, with no advice on when to do so. `channel_flow_demo.ipynb` uses `advect_van_leer` with forward Euler. Kolmogorov flow appears only in `spectral_forced_turbulence.ipynb`, which uses a different solver (`spectral.time_stepping.crank_nicolson_rk4`).
- **Paper.** Kochkov et al. 2021 (arXiv:2102.01010, Appendix A) state that the baseline DNS uses Van Leer advection and "explicit time integration using Euler discretization". The DNS gin config (`jax_cfd/ml/models_configs/implicit_diffusion_dns_config.gin`) uses Lax-Wendroff with a TVD limiter.
- **Tests.** `jax_cfd/base/validation_test.py` (L34-71) checks every Taylor-Green case against the analytic solution with `convect_linear`, never with Van Leer. Its RK4 case has a tolerance of 8e-7, compared with 6e-6 for forward Euler, at CFL 0.1.
- **Upstream contact.** Issue google/jax-cfd#259 (Mosaic, July 2026) asked the maintainers which advection scheme they would pick. It has no replies, and the README now says the project is no longer maintained.

## What policy 1 implies

Read strictly, the documentation and tutorials would not tell a user to change the scheme. They present the default and claim it is second-order. The only material that points to central advection for TGV is the library's validation test suite, and that suite does not single out RK4. The O(dt) Lax-Wendroff time correction is a property of the default method, so rule 3 would keep it in the results as an explained anomaly, as was done for PICT's `PICT_PISO_SPLITTING_FLOOR`.

## Options

1. **Revert to the default and add an explained anomaly.** This matches the demos, the paper and the treatment of PICT. TGV agreement returns to 🟠 (k = 5 to 6), the order to 1.25 and fd_check to 5.5e-4. VJP peak memory at N = 128 and 1000 steps rises from 1.6 GB to 6.2 GB. The forward pass gets faster.
2. **Keep the change as L0 and treat the validation tests as documentation.** Results stay as they are. This widens policy 1 to cover test suites, so PICT and other solvers would need the same check for consistency.
3. **Run both as separate entries** (rule 4). This shows the default and the better-performing variant side by side, at the cost of a second CI entry.

## Recommendation

**Option 1, possibly together with option 3 if the variant should stay visible.** I am fairly confident about what the docs and demos say. I am less sure whether "documentation" should include test suites, and that question decides the case.

## Open questions

- Do library test suites count as documentation under policy 1?
- The demos run at `cfl_safety_factor = 0.5` and the tests at 0.1. I have not checked how the default's error at those CFLs compares with Mosaic's setup.
