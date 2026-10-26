# Policy 1: PhiFlow self-advection and time stepper

## The case

The PhiFlow tesseract (phiflow 3.4.0, JAX) advects velocity with `vel + dt * advect.differential(vel, vel)`, which is second-order central differences with explicit Euler, followed by `diffuse.explicit` and the projection (intervention `differential-advection`, scope periodic, periodic-3d and cylinder). The only reason on record is a comment from the initial commit (70063a8) that semi-Lagrangian advection caused "VJP gradient magnitude bias" and NaN. No numbers survive. The recorded cost is that central differences with Euler become unstable (ns-3d-grid NaN at 160 steps).

## What PhiFlow says (tag 3.4.0)

- **Default.** `advect.advect` sends grids to `semi_lagrangian` (`phi/physics/advect.py` L73). Every operator-splitting grid example self-advects velocity with `semi_lagrangian`, including `examples/grids/Taylor_Green.ipynb`, `Wake_Flow.ipynb` (cylinder) and `docs/Fluid_Simulation.ipynb`. `Rotating_Bar` and `Moving_Obstacles` use `mac_cormack`. In our issue tum-pbs/PhiFlow#214, the maintainer called semi-Lagrangian advection with explicit diffusion "the default scheme since it's very easy to run it in a stable manner". He seems to have assumed our tesseract already used it, so his reply does not cover the current code.
- **Accuracy.** The "Higher-Order Simulations" section of https://tum-pbs.github.io/PhiFlow/Fluid_Simulation.html says "The operator splitting approach is not compatible with more accurate numerical schemes." It then puts `advect.finite_difference` inside `fluid.incompressible_rk4` for "much more accurate results". https://tum-pbs.github.io/PhiFlow/Taylor_Green_Comparison.html (linked from the docs index and from #214) runs the Taylor–Green vortex with finite differences of order 2, 4 and 6 under `incompressible_rk4`. It compares them with semi-Lagrangian advection, which it labels order 1. `HigherOrder_Demo` and `Higher_order_Kolmogorov` use the same pairing. `finite_difference` is an alias of `differential` (advect.py L133). Every use of it in the docs runs under `incompressible_rk4`, and I found none with forward Euler. These notebooks still carry the warning from version 2.3 that higher-order solvers are "experimental" and periodic-only.
- **Obstacles.** `incompressible_rk4` (`phi/physics/fluid.py` L291) takes no obstacles, and no example uses it with one.
- **Gradients.** I found no guidance on semi-Lagrangian gradients. The optimization material differentiates through `semi_lagrangian` and mentions no caveat: `docs/Fluids_Tutorial.ipynb`, `demos/Top_Opt/Top_Opt3D.py` and the PBDL book's `diffphys-code-ns`. None of them runs a finite-difference (FD) gradient check. From reading phiml 1.14.1 (phiflow requires `phiml>=1.14.0`), the JAX backend interpolates with weights built from `coordinates % 1`. Autodiff therefore appears to include the derivative with respect to the backtraced position, which jumps where the backtrace crosses a cell boundary. I have not tested this.

## What policy 1 implies

For the periodic setups, the docs would tell a user who wants accuracy on a smooth periodic flow such as the Taylor–Green vortex to switch, and they name the switch: finite differences under `incompressible_rk4`. That change is allowed as L0. Central differences with forward Euler appear nowhere in the docs. For the cylinder, the docs offer only the default. The docs give no basis for the gradient reason, so any semi-Lagrangian gradient bias would be an explained anomaly.

## Options

1. **Default everywhere.** This matches the demos and the maintainer. Periodic forward errors would likely grow (order 1 per the comparison notebook), and FD checks may show the interpolation kinks. The 3D NaN may go away. None of this has been measured.
2. **Periodic: `finite_difference(order=2)` under `incompressible_rk4`. Cylinder: the default.** Periodic accuracy and stability should improve. Each step then needs four pressure solves instead of one, and the adjoint needs four too.
3. **Keep the current scheme.** It is undocumented, so it needs a level above L0 with a measured bias, or a separate entry (rule 4).

## Recommendation

**Option 2, with moderate confidence.** I am fairly sure what the docs say. I am less sure how much weight the "experimental" warning should carry. A quick FD check with `semi_lagrangian` would show the size of the cylinder anomaly before the switch.

## Open questions

- In 3.4.0 (and on `develop`), stage 4 of `incompressible_rk4` uses `rhs2` (fluid.py L325) where classical RK4 uses `rhs3`. My check of the order conditions puts it at third order for the unconstrained ODE. Should this be reported upstream?
- Which `pressure_order` should we use? The function default is 4, and the comparison notebook's "2nd order" run uses 2.
