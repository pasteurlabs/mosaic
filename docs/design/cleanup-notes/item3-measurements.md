# Backlog item 3: measurements, 2026-10-26

All runs on the sandbox A100 box (24 cores), images built from `main` at 6517f75 unless stated. Wall times in the OpenFOAM table were measured while other jobs ran on the box, so treat them as indicative; the planned quiet re-run (`diag/quiet_phase.sh`) did not happen.

## Done, with PRs

| Id                                    | Result                                                                                                                                                                                      | PR                                                                                                               |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `compliance-grad-field` (TopOpt.jl)   | Zygote gradient bit-identical to `comp.grad` (nx = 8, 16); gradient and optimization suites identical                                                                                       | branch `refactor/topopt-jl-zygote-gradient`, PR not opened (stacked on #260, `vjp_cost` timing not yet measured) |
| `channel-cg-scoped-device` (Warp-NS)  | no-op: `_warp_device()` is always Warp's default device; velocity/pressure bit-identical, force within atomics noise                                                                        | #264                                                                                                             |
| `courant-substepping` (OpenFOAM)      | never triggers: Co = 0.869 / 0.867 at 10 / 20 cells per diameter; bit-identical after removal                                                                                               | #263                                                                                                             |
| `poisson-fd-eigenvalues-3d` (Warp-NS) | not an exact inverse: one projection leaves 11% of the divergence (smooth field, N = 16), 68% (random); fd_check 1.5e-5 with these eigenvalues, 2.9e-5 with continuous ones. Comments fixed | #264                                                                                                             |

## XLB channel on BGK

The docstring's claim holds on the current channel (`drag_opt` setup, float64, two smooth inlet-profile directions over ±5% of U_m):

| Collision    | Cd(profile)                                         | VJP vs central FD |
| ------------ | --------------------------------------------------- | ----------------- |
| BGK          | smooth (quartic-fit residual 4e-8 over a 0.03 span) | 4e-6 to 1.5e-5    |
| stock KBC    | jumps by 0.1 to 0.3 (residual 0.18 over 0.31)       | NaN               |
| smoothed KBC | residual 0.06 over 0.17                             | off by 10³ to 10⁶ |

No code change. #194's smooth KBC gradients were measured on an older channel. Script: `diag/xlb/`.

## OpenFOAM channel settings (`forward/cylinder`)

Each variant reverts one setting to the `offsetCylinder` tutorial (OpenFOAM 12) or to the simpler choice. Main: Cd 5.56395 / 5.57661 at 10 / 20 cells per diameter, 93 s for both.

| Variant                                | Cd at 10 / 20       | Wall (both) | Reading                                                                                       |
| -------------------------------------- | ------------------- | ----------- | --------------------------------------------------------------------------------------------- |
| U tolerance 1e-5 (tutorial)            | same to 5 digits    | 89 s        | 1e-8 not needed for Cd; keep or loosen                                                        |
| p GAMG tol 1e-6, relTol 0.1 (tutorial) | same to 5 digits    | 56 s        | 40% faster, same answer: a tuning candidate                                                   |
| p PCG/DIC instead of GAMG              | same                | 213 s       | GAMG justified (2.3x)                                                                         |
| no non-orthogonal corrector            | 5.56396 / **crash** | —           | required: diverges at 20 (max Co 11.9, FPE in the momentum solve)                             |
| serial instead of MPI                  | same                | 90 s        | no measurable benefit at these mesh sizes (≤ 36k cells, 3 processes)                          |
| no grading                             | —                   | —           | not measured: my variant set the wake grading to 1, which divides by zero in the mesh formula |

Script: `diag/of_variants.py`.

## Built, not run (stopped at the #259 gate)

Variant images exist on the box for: FEniCS structural without `mumps-adjoint-monkeypatch` (`fenics_structural_mesh:nopatch`); deal.II structural uncapped threads (`ds-uncap`) and CG+SSOR (`ds-cg`); deal.II heat thread-capped (`dh-cap`) and CG tolerance 1e-8 (`dh-tol8`); JAX-FEM both with the default `jax_solver` (`jaxsolver`). `diag/quiet_phase.sh` runs them all serially; images vanish on a box reset, `diag/dealii_variants.py` and the worktree recipes rebuild them.

## Side findings

- Warp-NS runs are not bit-reproducible: forces use `wp.atomic_add`; two runs of `main` differ by 1e-7 in the force history, 7e-7 in the 3-D VJP, and 4e-4 in `drag_opt --debug`'s final Cd.
- The XLB image resolved an unpinned warp-lang 1.18.0, and the PhiFlow image an unpinned phiml 1.16.1.
- ns-3d-grid's `lbm_N_base` scaling lacks the 2-D `min(1, ·)` clamp (see `policy3-setup-deviations.md`).
