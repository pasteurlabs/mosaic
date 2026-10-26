# Policy 3: setup deviations to record

**Status:** draft for Dion. No code changed.

Policy 3 lets a solver run a documented substitute for a setup it can't express exactly, shown next to its results. Three cases are known.

## PICT: advective outflow instead of p = 0

- **Prescribed:** DFG 2D-1 outlet with zero pressure (`cylinder.BOUNDARY_CONDITIONS`, `x_hi: neumann`).
- **Substitute:** PICT's convective (advective) outflow, as in its own samples (`convective-outflow` in the spike record). PICT has no pressure outlet.
- **Why:** `cannot-express`.
- **Effect:** not isolated. #239 reports PICT's Cd 3% high and flat under refinement. The outflow, the orthogonal-scheme choice on the O-grid (`orthogonal-scheme-on-o-grid`) and the staircase cylinder all plausibly contribute; none has been measured separately.

## TopOpt.jl: load split evenly over face nodes

- **Prescribed:** a uniform traction F/(Ly·Lz) on the right face (`structural_mesh/physics.py`, `_cantilever_bcs`).
- **Substitute:** the harness gives TopOpt.jl the total force divided equally among the right-face nodes, as concentrated loads. TopOpt.jl's INP reader only takes normal surface pressure (`DLOAD`), and the load here is tangential to that face (−z on the x = Lx face).
- **Why:** `cannot-express`.
- **Effect:** not measured. An equal split puts the same force on corner, edge and interior nodes, whereas a consistent bilinear traction gives edge nodes half and corner nodes a quarter of an interior node's share. The total force and its resultant line of action are the same, so compliance should differ mainly through local deformation near the face. The size of that difference at the benchmark resolutions is unknown. One way to measure it: run a traction solver (e.g. JAX-FEM) with the same concentrated nodal loads.
- **Where it lives:** harness code (`_traction_solvers` set in `_cantilever_bcs`), so it can't carry a code marker. The record belongs on the TopOpt.jl entry.

## XLB: harness-side time-step scaling

- **Prescribed:** the experiment's `dt` and `steps`, the same for every solver.
- **Substitute:** for XLB only, the harness scales `dt` by `lbm_N_base / N` and `steps` by the inverse when `lbm_N_base` is set (`make_inputs` in both `navier_stokes_grid/physics.py` and `navier_stokes_3d_grid/physics.py`). The simulated time is unchanged.
- **Why:** LBM ties the time step to the grid (acoustic scaling), so a fixed physical dt means a different lattice Mach number at each N.
- **Inconsistency found:** the 2-D helper clamps the factor (`min(1, lbm_N_base / N)`), the 3-D helper does not. In ns-3d-grid `forward/physical_laws/vs_N`, N = 8 with `lbm_N_base = 16` gives XLB dt = 0.02 and 10 steps, twice the prescribed step, while every other solver runs dt = 0.01 and 20 steps. Whether this is intended I can't tell from the code or history.
- **Overlap:** the XLB tesseract also sub-steps internally to a target lattice Mach number (`_sub_steps`, `_MA_TARGET`). That part lives in the tesseract and is the solver's own business. Whether the harness scaling is still needed on top of it is an open question; if not, the deviation can be removed instead of recorded.

## Decisions needed

1. Record the three as deviations in the format of `interventions.md`, or treat XLB's scaling as part of the setup for LBM solvers?
2. Fix the 2-D/3-D clamp inconsistency (that would change XLB's 3-D `vs_N` result at N = 8), or confirm it.
3. Measure the TopOpt.jl load-split effect before recording it, or record it as `effect: unknown` first?
