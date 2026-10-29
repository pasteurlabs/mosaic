# Independent buoyancy reference

Native OpenCFD v2412 `buoyantBoussinesqPimpleFoam` numerical reference,
not an exact solution or solver consensus. The closed no-slip square has
zero normal scalar derivative, viscosity and diffusivity 0.0005, buoyancy 1,
and the seed-0 `interacting_plumes` initial condition. Final time: 0.4.

The 128×128 run at dt=0.00125 is restricted by face/volume averages to
16×16 (`reference_0`) and 32×32 (`reference_1`). JSON records native image,
source and artifact hashes, physical diagnostics, and refinement measurements.
Comparing 64×64 and 128×128 changes restricted velocity by at most 1.39%
and scalar by 0.39%. Halving the fine timestep changes velocity by at most
0.032% and scalar by 0.022%. These are sensitivity estimates, not error
bounds; rankings within these differences are unresolved. Native mass drift
is below 5e-10 and divergence RMS below 1.5e-7. A separate cosine diffusion
check gave relative scalar error 1.738e-5 and exactly zero velocity.

Temperature is `T = 300 K + theta K`. Beta and gravity map variable buoyancy
to `+theta e_y`; constant gravity is absorbed in pressure. Laminar `nu/Pr`
is the requested diffusivity. Velocity is recovered from corrected native
face fluxes. This is forward-only, without exact restart or AD claims.

Generated and audited on Slurm October 29, 2026 using the external
`buoyancy-openfoam-reference-20261029/generate.py` campaign harness.
Do not overwrite with the generic consensus generator. Regeneration requires
native spatial and temporal refinement audits.

The generation script is shipped alongside the native adapter as
`mosaic/tesseracts/buoyancy-grid/openfoam/generate_reference.py`.
Inside the pinned OpenFOAM worker image with this repository's Python
requirements installed, run its `run` command three times with `--repo REPO`:

- `--n 64 --dt .0025 --steps 160 --output /scratch/coarse`
- `--n 128 --dt .00125 --steps 320 --output /scratch/fine`
- `--n 128 --dt .000625 --steps 640 --output /scratch/audit`

Then use `assemble --coarse /scratch/coarse --fine /scratch/fine
--audit /scratch/audit --output /scratch/candidate`. Review the diagnostics
and convergence audit before admission, and record the actual worker image
digest in the manifest. Assembly intentionally does not write to the repository.
