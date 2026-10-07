# PhiFlow PR 202: forward stability evidence

These are new forced-flow stress cases, not a claim that the old benchmark suite
fails. `provenance.json` binds code, exact adapters, jobs and the dependency image;
`SHA256SUMS.json` binds every published data file. Numerical work ran on allocated
RTX 5090 GPUs; plotting and numerical summaries ran on Slurm CPU allocations.

The baseline is #121, with the same native recurrent-state API. Both sides use
identical initial arrays and external Strang forcing; no repeated canonical
reinitialization. Periodic physical time is 75, nu=.001, extent 2π, forcing
(sin(6y),0[,0]); timestep .01 at 64²/32³/64³ and .01/3 at 192². Fixed seed 0
multimode 2D initial flow and z-dependent Taylor–Green 3D initial flow are explicit
in the archived `validate.py`.

`raw/` preserves baseline, SSPRK-only, fourfold-smaller-step, skew-advection, and
initial dtype-harness-failure attempts. `fields.tar.gz` contains complete initial
and final/failed fields; reports contain all scalar trajectories. The 64³ baseline
sparse-index failure occurs before stepping and is not a stability comparison.

The skew candidate completes all four T 75 cases; it is energy-neutral in the
periodic semidiscrete inner product but does not exactly preserve momentum.
The measured mean-velocity drift norms are about 2.8e-6/2.6e-6 in 2D and
.01019/.00310 in 3D. These coarse forced 3D fields are stability demonstrations,
not evidence of grid-converged turbulent accuracy. Explicit timestep limits remain.

`nonperiodic/` holds the final adapter, nine focused regression tests, and the
closed-wall 2D/3D T 2 refinement experiment. Native boundary-aware SSPRK 3 improves
time integration there. The legacy prescribed-inflow algorithm is unchanged;
there is no general open-boundary or inflow energy-stability claim.

Reproduce inside the image identified by `image.sha256` (dependencies only),
with PYTHONPATH=/tesseract and explicit frozen adapter paths:

```sh
python validate.py --adapter baseline.py --resolution 64 --ndim 2 --out before
python validate.py --adapter candidate.py --resolution 64 --ndim 2 --out after
python checks.py --adapter candidate.py --out checks.json
python closed_walls.py
```

The standalone closed-wall script expects `/validation/baseline.py` and
`/validation/candidate.py` and writes into that mount. Runtime numbers are
warmed four-native-step calls,30 repetitions, same GPU/inputs, compilation
reported separately. Plotting time is not included in those kernel timings.
