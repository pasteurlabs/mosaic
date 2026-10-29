# PhiFlow 2-D cost and pressure-solve tolerance

**Status:** 2026-10-29. The cost is PhiFlow's method and gets an explained anomaly. The periodic tolerance goes back to phiml's default (#296).

## Question

After #293, PhiFlow's ns-grid cost is 21–42× the GPU median, and #288's RK4 step (four projections per step) adds 2.6–3×. The handoff asked for the cause to be found before it is explained: CPU against GPU, the unpreconditioned CG and per-step host syncs.

## Findings

All timings on the A100 with no other job on the GPU (checked with `docker events`), unless marked CPU. Scripts in `diag/phiflow/`.

- **GPU against CPU.** The GPU is not slower at any N. On the CPU, one projection alone takes longer than main's whole GPU step: 5.8 ms against 4.9 ms at N = 64, 51 ms against 22 ms at N = 128. #233's "2-D is 3× slower on the GPU" came from a shared GPU.
- **Cost per CG iteration** (`cg-probe.py C`, time at a 1000-iteration cap minus 100, divided by 900): 37 / 66 / 113 / 184 µs at N = 64 / 128 / 192 / 256. Roughly 25 µs fixed plus 2.4 ns per cell. Only N = 64 is near the latency floor of the ~10 kernels and the host-side convergence check per iteration (`cg-probe.py H`).
- **The sparse matrix-vector product is the cost** (`spmv.py`, same shape as phiml's matrix, 7 entries per row including the rank-deficiency row and column): 19 / 47 / 94 / 159 µs, 51–86% of an iteration. The same Laplacian as a `jnp.roll` stencil takes 14 µs at N = 256. phiml solves with an explicit sparse matrix whenever the backend supports sparse tensors (`phiml/math/_optimize.py` L629, `prefer_explicit`), and `make_incompressible`'s Laplacian is a `@jit_compile_linear` function, so the user has no matrix-free option. JAX lowers the BCOO product to a gather and a scatter-add.
- **Iterations per projection** (first projection from x0 = 0): about 100 / 245 / 350 / 520 at 1e-7 and 73 / 141 / 215 / 296 at 1e-5. On the GPU the solve converges under the 1000 cap at every N; on the CPU at N = 256 it does not reach 1e-7 (residual 1.9e-7 at 1000 iterations, 2.9e-6 at 5000, from float32 drift).
- **Preconditioners** (`cg-probe.py D`): `'ilu'` is unsupported under JAX (no sparse triangular solve; under jit phiml falls back to SciPy). `'cluster'` crashes on the periodic matrix: its clusters have N² entries, the vector N² + 1 from the rank-deficiency row. This is a phiml bug and a candidate upstream issue (not drafted, not approved). With `rank_deficiency=0` it runs but needs more iterations than plain CG.

## Tolerance

The tuning rule asks for the loosest tolerance at which every check is unchanged. The A/B (`tol-fast.sh`, `tol-compare.py`; images built from #288's head) ran ns-grid forward without the cylinder, fd_check and analytic_check, and ns-3d-grid forward, fd_check and the recovery. 1e-6 and 1e-5 against 1e-7:

- forward errors and gradient checks within 0.15%;
- recovery final IC error 0.00308 / 0.00347 / 0.00301 at 1e-7 / 1e-6 / 1e-5, no trend;
- ns-grid `divergence_rms` up to +65% at 1e-5, around 1e-7; no check uses it.

The ns-3d stall that motivated 1e-7 came from the central-difference Euler step and does not occur with RK4. 1e-5 is phiml's float32 default, so the default comes back. One projection gets 1.4–1.8× faster. The end-to-end cost, the Jacobian, horizon and parameter sweeps and the cylinder are left to #296's benchmark run.

## Open

- The phiml `'cluster'` issue, if Dion wants it filed.
- `rank_deficiency=0` cuts iterations at N ≤ 128 (77 against 88, 161 against 238) but leaves the rank-deficient system unaugmented; results not checked.
- XLA while-loop command buffers could remove the per-iteration host sync; that is an XLA flag, not PhiFlow usage, and only matters at small N.
