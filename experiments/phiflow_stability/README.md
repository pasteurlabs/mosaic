# Periodic PhiFlow forward stress test

This change is motivated by a newly tested, long forced-flow regime; it does not
claim that the existing benchmark suite fails. The solver PR is stacked on #121
so both baseline and candidate retain the same recurrent staggered state instead
of repeatedly resampling from collocated velocity.

The candidate applies a discrete MAC Fourier pressure projection and projected
SSPRK3 to obstacle-free periodic flow in 2D and 3D. Every Runge–Kutta stage adds
advection and diffusion evaluated at the same stage input. Walls, obstacles, and
inflow retain their existing paths. Explicit advection/diffusion timestep limits
still apply. The external Strang forcing is second order; this experiment does
not claim third-order accuracy for that forced composition.

## Focused validation

Run only on an allocated GPU inside the PhiFlow dependency container:

```sh
python validate.py --adapter baseline.py --resolution 64 --ndim 2 --out baseline-2d-64
python validate.py --adapter candidate.py --resolution 64 --ndim 2 --out candidate-2d-64
python checks.py --adapter candidate.py --out checks.json
```

Repeat the paired forward comparison at 192², 32³, and 64³. Each pair runs on the
same GPU, in separate Python processes. The baseline is the unchanged adapter at
`38e07fbe6cc0266526f0eb1d25ab5afc29bd5630` (#121). The script records the exact
adapter SHA, device, JAX version, first-call time, total time, energy, native MAC
divergence, maximum speed, and the last finite physical time. Full initial and
final (or last failed) fields are retained. Finite checks occur every 0.04 physical
time units, so a reported failure time is an interval, not an exact onset.

Both variants use viscosity 0.001, periodic extent 2π, force `(sin(6y),0[,0])`,
Strang force splitting, and physical duration 75. The 2D timesteps are 0.01 at
64² and 0.01/3 at 192². Initial 2D velocity is the fixed seed-0 multimode field
plus 0.1 times the force. The 3D timestep is 0.01; its initial velocity is a
z-dependent Taylor–Green field plus 0.1 times the same force. These are prescribed
stress cases, not a general stability guarantee.

`checks.py` additionally checks rectangular 2D and 3D projector divergence,
idempotence and adjoint symmetry; an analytically advected/diffused shear wave;
and an eight-step velocity VJP against centered finite differences (primary
perturbation 0.001). The existing benchmark CI covers the established cases.

Results and artifact links will be added after the allocated jobs finish.
