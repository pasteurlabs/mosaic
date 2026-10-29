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
