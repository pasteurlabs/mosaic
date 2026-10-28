# Policy 3: setup deviations

**Status:** decided 2026-10-27. XLB: harness scaling removed (#284). TopOpt.jl: consistent nodal loads (#281). PICT: recorded as a deviation.

Policy 3 lets a solver run a documented substitute for a setup it can't express exactly, shown next to its results. The question each case answers for a practitioner is "does it solve my problem as specified?", and for a developer "were the rules the same for everyone?". That gives a preference order:

1. If the solver can express the setup, give it the setup; a deviation Mosaic introduced itself is a fairness problem, not a property of the solver.
2. If it can't, the substitute lives in the solver's tesseract, where it is the solver's own handling, and the harness gives every solver the same inputs.
3. A recorded deviation states its effect with a number where one can be had.

## XLB: harness-side time-step scaling

- **Was:** for XLB only, the ns-3d-grid harness scaled `dt` by `16 / N` and `steps` by the inverse (`lbm_N_base`). The 2-D helper had the same code with a `min(1, ·)` clamp but no experiment used it. In 3-D the missing clamp gave XLB twice the prescribed step at N = 8.
- **Why it existed:** LBM ties the time step to the grid through the lattice Mach number, so a fixed physical `dt` means a larger Ma at larger N.
- **But:** the XLB tesseract already sub-steps every prescribed step down to a target Mach number (`_sub_steps`), and ns-grid relies on that alone. The harness scaling was a second mechanism for the same constraint, specific to one solver and one problem.
- **Decision:** remove it from both helpers (rule 2 above; it also removes the clamp question). The sub-stepping stays and is recorded on XLB as its deviation: `cannot-express` (LBM couples dt to dx through Ma), k = ⌈Ma / Ma_target⌉ LBM steps per prescribed step, never coarser than prescribed.
- **Effect** (`forward/physical_laws/vs_N`, A100, branch `refactor/xlb-drop-harness-dt-scaling`): every status unchanged. N = 8 analytic error 0.064 → 0.079 and kinetic energy 0.108 → 0.104; N = 16 identical; N = 32 divergence RMS 3.0e-4 → 5.6e-4. `forward/agreement` (N = 16) is unaffected. Cost at N = 32 to 64 should drop because XLB no longer runs a smaller step than its sub-stepping needs; not timed on a quiet GPU, the `benchmark:domain` run on #284 will show it.

## TopOpt.jl: load split evenly over face nodes

- **Prescribed:** a uniform traction F/(Ly·Lz) on the right face (`structural_mesh/physics.py`, `_cantilever_bcs`).
- **Was:** the harness gave TopOpt.jl the total force divided equally among the right-face nodes, as concentrated loads, because its INP reader takes only normal surface pressure (`DLOAD`) and this load is tangential.
- **Decision:** rule 1 above. TopOpt.jl's `CLOAD` takes arbitrary per-node values, so the harness can pass the consistent nodal loads of the uniform traction on the trilinear face mesh (corner ¼, edge ½, interior 1 of an interior node's share), which is the discretization a traction-capable FE code produces. No deviation remains. Agreement and baseline errors 5.5e-3 → 0 (TopOpt.jl now assembles the same trilinear load as the traction solvers that set the reference), compliance 3.99731e-3 → 3.97532e-3, gradients and optimization unchanged (#281).

## PICT: advective outflow instead of p = 0

- **Prescribed:** DFG 2D-1 outlet with zero pressure (`cylinder.BOUNDARY_CONDITIONS`, `x_hi: neumann`).
- **Substitute:** PICT's convective (advective) outflow, as in its own samples (`convective-outflow`). PICT has no pressure outlet.
- **Why:** `cannot-express`, so rule 2; record it on PICT.
- **Effect:** not isolated. #239 reports PICT's Cd 3% high and flat under refinement; the outflow, the orthogonal-scheme choice on the O-grid (`orthogonal-scheme-on-o-grid`) and the staircase cylinder all plausibly contribute. The outlet is 2.0 m (20 D) downstream of the cylinder, so I would expect the outflow's share to be small, but I have not measured it. Measuring it needs a solver that supports both outlet types; OpenFOAM does, which would make it one A/B run.
