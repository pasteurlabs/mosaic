# Backlog item 3: measurements, 2026-10-26

All runs on the sandbox A100 box (24 cores), images built from `main` at 6517f75 unless stated. Timing comparisons ran serially on an otherwise idle box (`diag/quiet_phase.sh`, `diag/followup*.sh` in `diag/`).

## Outcomes

| Id                                              | Result                                                                                                                                                                                                                                     | PR                                |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------- |
| `compliance-grad-field` (TopOpt.jl)             | Zygote gradient bit-identical to `comp.grad`; suites identical; VJP time the same over three runs                                                                                                                                          | #270 (stacked on #260)            |
| `channel-cg-scoped-device` (Warp-NS)            | no-op; fields bit-identical, forces within atomics noise                                                                                                                                                                                   | #264                              |
| `poisson-fd-eigenvalues-3d` (Warp-NS)           | not an exact inverse (11% divergence left on a smooth field at N = 16); no gradient bias (fd_check 1.5e-5 vs 2.9e-5 with continuous eigenvalues). Comments fixed                                                                           | #264                              |
| `courant-substepping` (OpenFOAM)                | never triggers (Co 0.869 / 0.867); removed, bit-identical                                                                                                                                                                                  | #263                              |
| OpenFOAM channel settings                       | see table below                                                                                                                                                                                                                            | #269 (stacked on #263)            |
| XLB channel on BGK                              | docstring confirmed (table below)                                                                                                                                                                                                          | none                              |
| `mumps-adjoint-monkeypatch` (FEniCS structural) | needed for speed: VJP 11.1 s vs 14.6 s at nx = 128, 3.2 vs 4.7 s at 64, no difference on small meshes; gradients identical (float32). The patch is live: stock `linalg_solve` receives `solver_parameters` but drops them and runs UMFPACK | none; record as `too-slow`, ~1.3x |
| `umfpack-direct-solve` (deal.II structural)     | CG+SSOR gives the same compliance but is 1.7x slower at nx = 128 (4.9 vs 2.9 s)                                                                                                                                                            | none; keep, record                |
| `cg-ssor-tight-tolerance` (deal.II heat)        | 1e-8 gives identical forward metrics, 15% faster at nx = 1024 (51 vs 60 s)                                                                                                                                                                 | #266                              |
| deal.II threading                               | structural cap has no effect (2.93 vs 2.89 s at nx = 128); capping heat costs 8%                                                                                                                                                           | #271 drops the structural cap     |
| `umfpack-solver-option` (JAX-FEM, both)         | trade-off, see `jaxfem-linear-solver.md`                                                                                                                                                                                                   | decision for Dion                 |

## XLB channel on BGK

`drag_opt` setup, float64, two smooth inlet-profile directions over ±5% of U_m:

| Collision    | Cd(profile)                                         | VJP vs central FD |
| ------------ | --------------------------------------------------- | ----------------- |
| BGK          | smooth (quartic-fit residual 4e-8 over a 0.03 span) | 4e-6 to 1.5e-5    |
| stock KBC    | jumps by 0.1 to 0.3 (residual 0.18 over 0.31)       | NaN               |
| smoothed KBC | residual 0.06 over 0.17                             | off by 10³ to 10⁶ |

#194's smooth KBC gradients were measured on an older channel. Script: `diag/xlb/`.

## OpenFOAM channel settings (`forward/cylinder`, two repeats)

| Variant                                | Cd at 10 / 20 cells per diameter                                   | Wall (both)        |
| -------------------------------------- | ------------------------------------------------------------------ | ------------------ |
| `main`                                 | 5.563948193 / 5.576608235                                          | 82–83 s            |
| p GAMG tol 1e-6, relTol 0.1 (tutorial) | 5.563948158 / 5.576608228                                          | 50 s               |
| p PCG/DIC instead of GAMG              | 5.563948175 / 5.576608231                                          | 193 s              |
| serial instead of MPI                  | identical                                                          | 82–83 s            |
| U tolerance 1e-5 (tutorial)            | same to 5 digits                                                   | ~same (single run) |
| no non-orthogonal corrector            | 5.56396 / crash (max Co 11.9)                                      | —                  |
| uniform mesh (grading 0.9999 / 1.0001) | 5.567318 / 5.578718 (Cd closer to 5.5795; Cl, Δp slightly further) | 111 s (single run) |
| #269 (tutorial p tolerance, serial)    | 5.563948158 / 5.576608228                                          | 53 s               |

Script: `diag/of_variants.py`.

## Side findings

- Warp-NS runs are not bit-reproducible: forces use `wp.atomic_add`; two runs of `main` differ by 1e-7 in the force history, 7e-7 in the 3-D VJP, and 4e-4 in `drag_opt --debug`'s final Cd.
- The XLB image resolved an unpinned warp-lang 1.18.0, and the PhiFlow image an unpinned phiml 1.16.1.
- ns-3d-grid's `lbm_N_base` scaling lacks the 2-D `min(1, ·)` clamp (see `policy3-setup-deviations.md`).
