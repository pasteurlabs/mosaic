# Policy 1: PhiFlow self-advection and time stepper

**Status:** decided 2026-10-28 from first principles and measurement (Dion left it open): periodic boxes run central differences under `incompressible_rk4`, the channel runs the default semi-Lagrangian advection (#288).

## The case

The PhiFlow tesseract (phiflow 3.4.0, JAX) advects velocity with `vel + dt * advect.differential(vel, vel)`, which is second-order central differences with explicit Euler, followed by `diffuse.explicit` and the projection (intervention `differential-advection`, scope periodic, periodic-3d and cylinder). The only reason on record is a comment from the initial commit (70063a8) that semi-Lagrangian advection caused "VJP gradient magnitude bias" and NaN. No numbers survive. The recorded cost is that central differences with Euler become unstable (ns-3d-grid NaN at 160 steps).

## What PhiFlow says (tag 3.4.0)

- **Default.** `advect.advect` sends grids to `semi_lagrangian` (`phi/physics/advect.py` L73). Every operator-splitting grid example self-advects velocity with `semi_lagrangian`, including `examples/grids/Taylor_Green.ipynb`, `Wake_Flow.ipynb` (cylinder) and `docs/Fluid_Simulation.ipynb`. `Rotating_Bar` and `Moving_Obstacles` use `mac_cormack`. In our issue tum-pbs/PhiFlow#214, the maintainer called semi-Lagrangian advection with explicit diffusion "the default scheme since it's very easy to run it in a stable manner". He seems to have assumed our tesseract already used it, so his reply does not cover the current code.
- **Accuracy.** The "Higher-Order Simulations" section of https://tum-pbs.github.io/PhiFlow/Fluid_Simulation.html says "The operator splitting approach is not compatible with more accurate numerical schemes." It then puts `advect.finite_difference` inside `fluid.incompressible_rk4` for "much more accurate results". https://tum-pbs.github.io/PhiFlow/Taylor_Green_Comparison.html (linked from the docs index and from #214) runs the Taylor–Green vortex with finite differences of order 2, 4 and 6 under `incompressible_rk4`. It compares them with semi-Lagrangian advection, which it labels order 1. `HigherOrder_Demo` and `Higher_order_Kolmogorov` use the same pairing. `finite_difference` is an alias of `differential` (advect.py L133). Every use of it in the docs runs under `incompressible_rk4`, and I found none with forward Euler. These notebooks still carry the warning from version 2.3 that higher-order solvers are "experimental" and periodic-only.
- **Obstacles.** `incompressible_rk4` (`phi/physics/fluid.py` L291) takes no obstacles, and no example uses it with one.
- **Gradients.** I found no guidance on semi-Lagrangian gradients. The optimization material differentiates through `semi_lagrangian` and mentions no caveat: `docs/Fluids_Tutorial.ipynb`, `demos/Top_Opt/Top_Opt3D.py` and the PBDL book's `diffphys-code-ns`. None of them runs a finite-difference (FD) gradient check. From reading phiml 1.14.1 (phiflow requires `phiml>=1.14.0`), the JAX backend interpolates with weights built from `coordinates % 1`. Autodiff therefore appears to include the derivative with respect to the backtraced position, which jumps where the backtrace crosses a cell boundary. I have not tested this.

## Decision

Keep the documented schemes: central differences under `incompressible_rk4` in the periodic boxes, default semi-Lagrangian advection with operator splitting in the channel. The reasoning, from first principles:

- **The benchmark should measure PhiFlow as its documentation tells a practitioner to use it.** Central differences with forward Euler are a Mosaic construction; `finite_difference` appears in the docs only under `incompressible_rk4`. Keeping them would benchmark Mosaic's scheme under PhiFlow's name.
- **Forward Euler with central advection is only conditionally stable.** Central advection puts the eigenvalues on the imaginary axis, outside forward Euler's stability region, and only viscosity damps them. A linear analysis (no projection, no nonlinearity) gives |G| ≈ 1 + (|u| k dt)²/2 − ν k² dt per step, which grows unless dt ≤ 2ν/|u|². The limit goes to zero as ν → 0. RK4 is stable on the imaginary axis up to |λ dt| ≈ 2.8; the benchmark's CFL numbers are at most about 0.5.
- **Today's checks don't hit that limit.** The cost runs (ν = 0.01, dt = 0.01, limit 0.02) are inside it. The 2-D checks at ν ≤ 1e-3 and dt = 0.05 are outside it but run at most 320 steps. In the probe below, main took 800–1500 steps to blow up at N = 32, and the linear growth rate scales with k², so the N = 16 checks should take longer. So the benchmark does not show the instability today; longer low-viscosity rollouts, such as those planned for the Kolmogorov flow, would.
- **The cost is PhiFlow's.** Four pressure projections per step is what PhiFlow charges for its second-order periodic scheme. Main's one projection per step reports a cost for a scheme PhiFlow does not offer.
- **The 3-D horizon does not change.** Both schemes go non-finite between 120 and 160 steps at a CFL number of about 0.16, deep inside RK4's stability region, so the time stepper is not what fails there (below).
- **Channel:** `incompressible_rk4` takes no obstacles, so the documented choice is the default, as in `Wake_Flow.ipynb`. It is also 5× closer to the DFG drag.

The default everywhere (option 1) is also documented, but the docs label it first order and point users who want accuracy on periodic flow away from it, and it measures as such.

## Measured

`mosaic run -s PhiFlow` on an A100, three variants of the tesseract against the same peers: main (central + Euler everywhere), RK4 (central under `incompressible_rk4`, `pressure_order=2`, in the periodic boxes) and SL (semi-Lagrangian everywhere). Then the branch, which combines RK4 in the periodic boxes with SL in the channel; it reproduces both arms' numbers.

|                                                    | main                  | RK4                   | SL                                             | branch                                |
| -------------------------------------------------- | --------------------- | --------------------- | ---------------------------------------------- | ------------------------------------- |
| ns-grid `baseline` error                           | 5.94e-3               | 5.75e-3               | anomaly (0.198 at the coarsest N, first order) | 5.75e-3                               |
| ns-grid `agreement/tgv` error                      | 2.41e-3               | 2.39e-3               | anomaly                                        | 2.39e-3                               |
| ns-grid `analytic_check` rel. error                | 0.019                 | 0.019                 | 0.27 (anomaly)                                 | 0.019                                 |
| ns-grid `fd_check` rel. error (ε = 1)              | 9.1e-6                | 8.6e-6                | 2.7e-4                                         | 1.3e-5                                |
| ns-grid `cylinder` error (Cd vs DFG)               | 0.0538                | 0.0538                | 0.0111                                         | 0.0111                                |
| ns-grid `drag_opt` drag reduction                  | 46.8% (57 it.)        | 46.7%                 | 47.6% (65 it.)                                 | not rerun (same channel code as SL)   |
| ns-3d-grid `baseline` error                        | 0.041                 | 0.0398                | anomaly                                        | 0.0398                                |
| ns-3d-grid `fd_check` rel. error (ε = 1)           | 3.2e-5                | 2.7e-5                | anomaly                                        | 1.2e-5                                |
| ns-3d-grid `horizon_sweep_limits`, last finite VJP | 80 steps (NaN at 160) | 80 steps (NaN at 160) | 1280 steps                                     | not rerun (same periodic code as RK4) |

RK4 changes the periodic errors by a few percent at most; the spatial error dominates on these setups. SL is first order on the periodic box and about 10× less accurate there, as the docs say.

**Kinetic energy over long rollouts** (`diag/phiflow/energy.py`, forward only, Taylor–Green initial conditions, KE relative to the start; the branch's pressure carry restarts every 40 or 100 steps in this probe). Periodic incompressible flow can only lose energy, so a rise is numerical.

| case                                                 | main                                   | branch                             |
| ---------------------------------------------------- | -------------------------------------- | ---------------------------------- |
| 2-D, N = 32, ν = 0, dt = 0.05                        | 0.87 at 700 steps, NaN at 800          | 0.68 at 2000 steps, monotone decay |
| 2-D, N = 32, ν = 1e-3, dt = 0.05                     | 0.63 at 1200, 2.1 at 1400, NaN at 1500 | 0.46 at 2000, monotone             |
| 2-D, N = 64, ν = 0, dt = 0.1                         | NaN by 100                             | 0.93 at 2000, monotone             |
| 3-D, N = 20, ν = 1e-3, dt = 0.05 (the horizon setup) | 1.45 at 120, 26 at 160, NaN at 200     | 1.47 at 120, NaN at 160            |
| 3-D, N = 20, ν = 0, dt = 0.05                        | 2.0 at 120, NaN at 160                 | 2.1 at 120, NaN at 160             |

In 2-D, main fails where the linear analysis says it should and the branch does not. In 3-D both gain energy and fail at the same point. The 3-D Taylor–Green vortex at this viscosity cascades to scales far below a 20³ grid, and central differences in advective form do not conserve kinetic energy discretely, so aliasing at the grid scale can feed energy in; semi-Lagrangian advection damps it, which fits SL's 1280 steps. That is my reading; I have not isolated it, for example with a skew-symmetric form of the advection term.

**Cost** (main and the branch back to back, one job on the GPU; main's times are within 10% of the reference run in most cells, median 2%, with outliers up to 50% on the smallest cases): branch / main, per sweep point of `spatial_cost`, `temporal_cost` and `vjp_cost` (`by_N`, `by_steps`).

|         | ns-grid (N = 64 to 256, 10 to 1000 steps)    | ns-3d-grid (N = 16 to 64, 10 to 100 steps) |
| ------- | -------------------------------------------- | ------------------------------------------ |
| forward | ×2.6–3.0 (N = 256, 100 steps: 14.5 → 41.0 s) | ×2.4–3.9 (N = 64: 3.96 → 14.1 s)           |
| VJP     | ×3.1–3.4 (N = 256: 29.5 → 96.9 s)            | ×3.3–4.4 (N = 64: 8.1 → 35.9 s)            |

Other jobs shared the GPU during an earlier pass of some of these cells; those numbers were discarded and the cells rerun (`diag/phiflow/phiflow-quiet2.sh`).

`incompressible_rk4` in 3.4.0 is third order, not fourth (`upstream-phiflow-rk4.md`); irrelevant here because the spatial error dominates.

## Open questions

- Whether to report the RK4 stage bug upstream (draft awaiting approval).
- `pressure_order=2` follows the comparison notebook's 2nd-order run; the function default is 4. Not measured.
- MacCormack advection with operator splitting (`Rotating_Bar`, `Moving_Obstacles`) would cost one projection per step and may be second order in space on smooth flow. The docs do not offer it for accurate periodic flow, and I have not measured it.
- The Kolmogorov plan adds the forcing to PhiFlow's periodic step; with this change it goes into `momentum`, so every RK4 stage sees it.
