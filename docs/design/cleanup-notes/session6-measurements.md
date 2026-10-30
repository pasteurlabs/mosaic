# Measurements from the sixth session (2026-10-29)

Local A/B runs on the sandbox's A100 box (24 cores), with `~/localfiles/runs/cleanup2/ab.sh`; outputs in `/nvme/scratch/ab/<arm>`, scripts in `~/localfiles/runs/cleanup3/`. Other jobs shared the box throughout (the Kolmogorov session's PICT container, this session's own GPU arms), so timings are indicative unless a table says otherwise.

## deal.II heat: preconditioner (#305)

`spatial_cost`, mean per call. Forward metrics identical in all arms.

| nx   | SSOR (`main`) | UMFPACK | AMG (Trilinos ML) |
| ---- | ------------- | ------- | ----------------- |
| 16   | 0.014 s       | 0.014 s | 0.020 s           |
| 128  | 0.406 s       | 0.421 s | 0.414 s           |
| 256  | 1.74 s        | 1.77 s  | 1.58 s            |
| 512  | 9.24 s        | 7.67 s  | 6.28 s            |
| 1024 | 61.1 s        | 35.6 s  | 25.6 s            |

Trilinos needs MPI; with the persistent worker (#297) it initializes once per process, so the ~1.35 s per-call start-up that #245 removed does not return.

## OpenFOAM periodic: pressure solver and decomposition (#306)

Arms: `main` (PCG/DIC, serial), GAMG, MPI decomposition with `min(cores, cells / 10k)` ranks, both. Forward errors move in the fifth significant digit at most.

| Cell                    | PCG    | GAMG   | PCG + MPI | GAMG + MPI |
| ----------------------- | ------ | ------ | --------- | ---------- |
| 2-D spatial N = 256     | 37.4 s | 16.4 s | 11.4 s    | 8.40 s     |
| 2-D temporal 1000 steps | 44.7 s | 27.8 s | 44.1 s    | 27.7 s     |
| 3-D spatial N = 48      | 11.2 s | 10.9 s | 6.80 s    | 7.04 s     |
| 3-D spatial N = 64      | 28.7 s | 27.1 s | 150 s     | 279 s      |
| 3-D temporal 100 steps  | 19.9 s | 19.7 s | 7.93 s    | 8.61 s     |

3-D N = 64 runs 24 ranks on 24 cores. A follow-up probe (one apply, 3-D N = 64, 50 steps, ranks limited by `--cpuset-cpus`, box load 6–27): serial 28–30 s at any load; 8 ranks 18 and 64 s; 12 ranks 15–16 s; 16 ranks 29 and 73 s; 24 ranks 147 and 164 s. The ranks busy-wait, so decomposition collapses when other processes hold cores. GAMG adopted; decomposition not, since benchmark runs share the machine with GPU solvers whose runtimes spin CPU cores. #269 removes the channel's decomposition for a related reason (no gain at its sizes).

## FEniCS and Firedrake heat: MPI

`mpirun -n` on a standalone script, nx = 1024 (1024 × 512 × 1 hexahedra, P1, MUMPS as in both tesseracts); third repeat of the solve.

| ranks | FEniCS | Firedrake |
| ----- | ------ | --------- |
| 1     | 12.6 s | 10.4 s    |
| 2     | 19.4 s | 8.83 s    |
| 4     | 27.2 s | 6.04 s    |
| 8     | 39.6 s | 5.08 s    |
| 16    | 77.0 s | 4.56 s    |

FEniCS (dolfin 2019, MUMPS) slows with every rank: stays serial. Firedrake's solve halves by 8 ranks, but a served tesseract is one Python process: `mpirun` per call costs 3.7 s of start-up at 8 ranks plus 4–8 s of form compilation in a fresh process, which cancels the gain. Using it would need a persistent MPI worker pool with pyadjoint's tape across ranks. Not done; an option for Dion.

## INS.jl

- **#292's change** (`empty!` on its setup cache) leaves cost unchanged: every cell within 3% of the pre-#292 image (ns-grid by_N 4.31 → 4.27 s, ns-3d-grid by_steps 24.6 → 25.3 s). The CI shift was baseline drift.
- **Julia threads** (`JULIA_NUM_THREADS=auto`): alone, the served container segfaults (45 dropped connections, every cost cell failed). juliacall starts Julia with `--handle-signals=no`, and multithreaded Julia stops threads for GC through signals; juliacall's own warning says to set `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`. With both settings: no failures, forward metrics identical, cost 1.5–4.4× lower (ns-grid by_N 4.27 → 1.82 s, ns-3d-grid by_steps 25.3 → 5.74 s). The GPU backend below is faster again, so this is recorded as the CPU option, not proposed.
- **GPU backend** (pending PR): the same mechanism crashed the CUDA build. CUDA.jl 6 hands GPU waits to Julia threads it adopts (GPUToolbox `cooperative_wait`); `Threads.maxthreadid()` goes from 1 to 2 after a few VJPs, and a GC during such a wait kills the server silently (core file, no OOM). The `CUDACore` preference `nonblocking_synchronization = false` keeps waits on the calling thread: `maxthreadid` stays 1, 300 served 3-D 40-step VJPs ran without a restart (the old image died by call 10), and a served call takes 1.1 s instead of 3 s.

## torch-fem setup cache: not adopted

A cache of the `SolidHeat` model and boundary tensors, keyed on the mesh, BCs and material constants. Interleaved per-call timing against `main`, one served container each:

| nx  | call  | `main` | cache hit | cache miss |
| --- | ----- | ------ | --------- | ---------- |
| 16  | apply | 101 ms | 109 ms    | 308 ms     |
| 64  | apply | 284 ms | 344 ms    | 345 ms     |
| 16  | VJP   | 160 ms | 159 ms    | 159 ms     |
| 64  | VJP   | 644 ms | 644 ms    | 644 ms     |

The setup is a negligible share of a call, and a hit is slower for `apply` (not investigated further). Branch `perf/torch-fem-setup-cache` stays unpushed.

## JAX-FEM: PETSc solvers (#309)

`main` passes JAX-FEM a jitted, Jacobi-preconditioned BiCGSTAB through `custom_solver`. The arms swap in JAX-FEM's own `petsc_solver` with CG and BoomerAMG (PETSc's default rtol and 1e-10) or GAMG. Mean `cost_time_s` over each sweep, failed per-sweep entries in brackets:

| Cell                 | BiCGSTAB (`main`) | CG + hypre | CG + hypre, 1e-10 | CG + GAMG   |
| -------------------- | ----------------- | ---------- | ----------------- | ----------- |
| structural spatial   | 0.30 s            | 0.79 s     | 0.52 s            | failed (8)  |
| structural VJP by N  | 0.74 s            | failed (1) | failed (1)        | failed (10) |
| thermal spatial      | 0.30 s            | 0.46 s     | 0.29 s            | 0.40 s      |
| thermal VJP by N     | 0.63 s            | failed (1) | 0.62 s            | failed (5)  |
| failed entries, both | 0                 | 2          | 1                 | 24          |

Every failure is JAX-FEM's own check after the PETSc solve (absolute residual < 0.1). JAX-FEM zeroes the Dirichlet rows of the matrix but not the columns, so the system is not symmetric and CG is not designed for it; this may be the cause, not verified. `main`'s solver stays. The same reading of `jax_fem.solver` showed that the `numerics` label ("Direct (SuperLU)") and the JVP docstring (UMFPACK) were wrong; #309 corrects them.

## PICT periodic: pressure solver options

Served-equivalent calls in each arm's image (in-process `apply` and VJP, TGV plus noise), two interleaved rounds, median of the warm repeats. GPU shared with the Kolmogorov session's PICT job and this session's Warp-NS A/B.

| Case                     | CG (`main`) | BiCG   | BiCG + precond.      | CG, `pressure_reuse_result` |
| ------------------------ | ----------- | ------ | -------------------- | --------------------------- |
| 2-D N=128, 100 st. apply | 3.64 s      | 3.76 s | 10.7 s               | 4.34 s                      |
| 2-D N=256, 100 st. apply | 4.30 s      | 3.56 s | NaN residual, step 1 | 4.26 s                      |
| 2-D N=256 VJP            | 7.59 s      | 7.96 s | —                    | 7.60 s                      |
| 3-D N=48, 20 st. apply   | 2.15 s      | 4.06 s | —                    | 2.38 s                      |
| 3-D N=48 VJP             | 3.70 s      | 4.72 s | —                    | 4.84 s                      |

BiCG changes outputs (2-D apply up to 4e-3 relative, viscosity gradient up to 100% at N=256 2-D) and is slower in the 3-D cases; in 2-D its times fall inside CG's own spread (3.07–5.30 s over the six N=256 repeats). The preconditioned BiCG breaks down at N=256 (`finalResidual=nan` after 520 iterations). `pressure_reuse_result` leaves `apply` bit-identical. CG stays; no PR.

Two side findings:

- PICT's VJP is not bit-reproducible: CG against CG across rounds differs by up to 1.8% in the 2-D N=256 viscosity gradient (3e-4 in the velocity gradient).
- 90% of a periodic `apply` is `PISOtorch.SolveLinear` (cProfile, 2-D N=128): 300 solves per 100 steps, about 32 ms each, and the whole apply grows only from 6.3 s at N=32 to 13 s at N=256 (under contention). The cost is per-solve latency, not grid work. This is the likely source of PICT's ns-grid cost anomaly (32× the GPU median in CI, not explained on `main`); not examined further.

## CI findings on this session's PRs

- **#296, #308:** the bot's PhiFlow baseline predates #288, so its PhiFlow cost "regressions" are #288's RK4 cost. #308's multimode change (4.7e-4 → 3.5e-3) is not from #308: `forward/agreement/multimode` on `main` and on #308 agrees to 2e-4 relative.
- **#307:** CI reports a new ceiling in ns-3d-grid `gradient/horizon_sweep_limits`: the 5120-step VJP fails on the CI GPU runner (the stored traceback is truncated before the exception type), while the A100 completes every horizon up to 10 240 steps.

## #291: XLA's default GEMM autotuning

Images for jax-cfd, PhiFlow and XLB built from `main` with #291 (`x291main`) and with #291's flags reverted (`x291off`). `forward/agreement/multimode` error per ν, old flags against `main`:

| Solver  | largest relative difference                 |
| ------- | ------------------------------------------- |
| jax-cfd | 0 (bit-identical)                           |
| XLB     | 0 (bit-identical)                           |
| PhiFlow | 5.6e-4 (ν = 0.05; 1e-7 to 2e-5 at the rest) |

The full forward, gradient and cost suites on `main` for the three solvers have no failed entries beyond those of the b6eb311 reference run, with one exception: XLB's ns-3d-grid `horizon_sweep_limits` VJP is NaN at 320 steps (the reference reached 5120). That is BGK's limit, recorded in #282, and it reproduces with the old flags (same gradient norms at 40–160 steps). So the flags do not explain CI's multimode shifts on #291, consistent with #298's finding that they came from scoring against the consensus of the solvers present in each CI job. The flags were not run on the CI's T4.

## PICT channel geometry cache: not adopted

A cache of the DFG channel's wall geometry and transfer weights, keyed on the mesh. Construction time per call, median of three repeats after a first call, in the pressure-tolerance image (`ppcg`) and with the cache (`pcache`):

| res | mesh    | without | with  |
| --- | ------- | ------- | ----- |
| 10  | 220×41  | 14 ms   | 5 ms  |
| 20  | 440×82  | 50 ms   | 6 ms  |
| 40  | 880×164 | 199 ms  | 40 ms |

`forward/cylinder` outputs are bit-identical. The benchmark never repeats a mesh: `forward/cylinder` runs res 10 and 20 once each, and PICT is excluded from `drag_opt`, so the cache would not hit. Branch `perf/pict-channel-geometry-cache` stays unpushed.
