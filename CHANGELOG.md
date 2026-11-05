# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-11-05

### Features

- *(solver)* Update torch-fem to 0.12.1, solve with CG+Jacobi on GPU (#187)
- *(solver)* Add torch-fem backend for structural-mesh (#188)
- *(harness)* Turn forward/baseline into a resolution-convergence test (#214)
- *(harness)* Score the ns-grid cylinder against Schäfer–Turek DFG 2D-1 in every obstacle solver (#239)
- *(harness)* Time the VJP request alone in `vjp_cost` (#241)
- *(harness)* Check ns-grid gradients against the exact Taylor–Green gradient (#250)
- *(domain)* Run jax-cfd on the ns-3d-grid experiments (#283)
- *(harness)* Record the cost warm-up call as first_call_s (#285)
- *(harness)* Record solver interventions, piloted on four solvers (#302)
- *(harness)* Let an experiment generate its own converged reference (#317)
- *(solver)* Record the interventions of every remaining solver (#303)
- *(solver)* Run INS.jl on the DFG 2D-1 cylinder channel (#325)
- *(domain)* Check dJ/dν against the exact TGV derivative (#333)
- *(harness)* Record core counts, thread settings and driver versions (#332)
- *(domain)* Time 10 trials in the mesh cost suites (#331)
- *(domain)* Extend the ns-3d cost and horizon sweeps (#330)
- *(domain)* Add forced Kolmogorov flow to ns-grid, with forward and chained-gradient checks (#272)
- *(harness)* Record the iteration of the best objective in optimization results (#341)
- *(harness)* Record when the loss levels off instead of the iteration of the lowest loss (#346)
- *(harness)* Record the time until an optimization's loss levels off (#358)

### Bug Fixes

- CI runners are dispatched even if matrix is empty (#161)
- *(harness)* Keep one solver's NaN out of the consensus reference (#164)
- *(harness)* Gate the three problems that had almost no status checks (#157)
- *(solver)* Seed the Firedrake source adjoint with the nodal functional (#162)
- *(harness)* Only score an IC against an analytic reference that solves it (#163)
- *(harness)* Make median_k's majority mean what it says (#174)
- Extend cost sweep to the paper's nx=128 resolution (#179)
- *(solver)* Seed the FEniCS source adjoint with the nodal functional (#167)
- *(solver)* Seed the Firedrake rho identification adjoint with the nodal functional (#184)
- Tune FEniCS Tesseracts (better linear solver, avoid redundant forward calls, only setup once) (#182)
- Stop flagging float32 round-off as forward outliers on mesh problems (#191)
- *(harness)* Read the served container name through container_info() (#192)
- *(solver)* Route XLB and Exponax VJPs through jax_vjp (#193)
- *(domain)* Apply the full prescribed flux in the thermal reference solve (#145)
- *(harness)* Import resource only when the L-BFGS diagnostic is on (#207)
- *(harness)* Read solver configs and templates as UTF-8 (#206)
- *(harness)* Fail non-finite results, report limit-sweep frontiers and per-solver hardware (#230)
- *(harness)* Explained anomalies annotate, never change status or score (#213)
- *(solver)* Correct ns-grid solver accuracy and run the JAX solvers on the GPU (#234)
- *(harness)* Publish results correctly when a PR redefines an experiment (#240)
- *(harness)* Build images with `python -m tesseract_core.sdk.cli` (#238)
- *(harness)* Stop reporting the path-dependent field error of thermal conductivity recovery (#242)
- *(solver)* Stop initializing MPI in the deal.II structural solver (#245)
- *(harness)* Rerun the --only cells under --continue (#252)
- *(harness)* Keep solvers that finished before a crash on --continue (#253)
- *(harness)* Score an --only rerun against its peers' cached outputs (#251)
- *(harness)* Stop writing empty cost entries for solvers that did not run (#254)
- *(harness)* Run the L-BFGS optimization experiments with SciPy's L-BFGS-B (#249)
- *(solver)* Compute the firedrake-heat identification error in float64 (#262)
- *(solver)* Key the JAX-FEM structural setup cache on the mesh coordinates (#261)
- *(solver)* Pin warp-lang for XLB instead of patching its source at build time (#265)
- *(solver)* Pin phiml in the PhiFlow tesseract (#279)
- *(solver)* Run Exponax with its default 2/3-rule dealiasing (#280)
- *(domain)* Give TopOpt.jl the consistent nodal loads of the structural traction (#281)
- *(solver)* Tighten PICT's periodic linear-solve tolerance to 1e-6 (#287)
- *(harness)* Compare cost within a device class in the cost checks (#293)
- *(solver)* Use XLA's default GEMM autotuning in jax-cfd, PhiFlow and XLB (#291)
- *(solver)* Make PICT's VJP linear in the cotangent (#294)
- *(harness)* Classify CUDA.jl's out-of-GPU-memory error as OOM (#312)
- *(harness)* Score ns-grid's multimode agreement against a checked-in consensus (#298)
- *(solver)* Key TopOpt.jl's setup cache on its inputs, not an XOR of their hashes (#300)
- *(solver)* Keep only the current grid's setup data in Warp-NS's caches (#311)
- *(domain)* Run Exponax on the periodic ns-grid experiments (#268)
- *(harness)* Apply the cost peer check to mesh resolution sweeps (#318)
- *(solver)* Checkpoint XLB's scan blocks, not only its steps (#329)
- *(harness)* Record VRAM in serial runs on single-GPU hosts (#327)
- *(solver)* Give every JAX solver the same GPU memory settings (#328)
- *(harness)* Measure peak memory in the cost suite (#326)
- *(harness)* Leave driver-reserved memory out of VRAM peaks (#336)
- *(domain)* Run PICT on drag_opt (#334)
- *(harness)* Record why a solver container died and restart it mid-sweep (#337)
- *(harness)* Chart 1-D agreement profiles, and name every benchmark label in the PR verdict (#345)
- *(harness)* Scaffold schema stubs into mosaic_shared (#342)
- *(cli)* Strip spaces around mosaic reference -e entries (#339)
- Skip problems without the -s solvers in mosaic build (#338)
- Reject unknown status formats and keep JSON output clean (#340)
- *(harness)* Count L-BFGS iterations without the initial point (#356)
- *(harness)* Rescale the CFL base grid in --debug runs (#353)
- *(harness)* Keep both runs' solver failures when merging results (#352)
- *(harness)* Add the IC helpers that mosaic ics imports (#351)
- *(harness)* Reject an unknown mosaic status --suites (#349)
- *(harness)* Refuse to overwrite an existing domain in new-domain (#350)

### Refactor

- *(solver)* Share the ns-grid BC checks and cylinder mask across tesseracts (#243)
- *(solver)* Remove unused Warp-NS adjoint clipping and grid_fluid helpers (#259)
- *(solver)* Move the INS.jl and TopOpt.jl solver code into their tesseracts (#260)
- *(solver)* Drop OpenFOAM's Courant sub-stepping in the cylinder channel (#263)
- *(solver)* Drop a no-op device scope and correct Warp-NS projection comments (#264)
- *(solver)* Run the deal.II structural solver with deal.II's default threading (#271)
- *(solver)* Run XLB's periodic cases with BGK collision (#282)
- *(solver)* Use PhiFlow's documented schemes for the periodic box and the channel (#288)
- *(solver)* Keep one setup in the JAX-FEM, TopOpt.jl and INS.jl caches (#292)
- *(solver)* Take TopOpt.jl's compliance gradient from its Zygote rule (#270)
- *(domain)* Stop rescaling XLB's ns-3d-grid time step in the harness (#284)

### Documentation

- Altair result plots — PNG and interactive chart from one definition (#204)
- Correct results-page text, exclusion reasons and solver metadata (#231)
- Explain the method-intrinsic anomalies of the full benchmark run (#244)
- Explain PICT's ns-grid cost anomaly (#313)
- Describe JAX-FEM's linear solver as it runs (#309)
- Show solver effort and setup deviations on the results pages (#304)
- Add a solver capability matrix to the Solver Reference (#319)
- Redesign docs site with solver browser, experiment explorer and per-setup results checks (#203)
- Drop the unused staggered round-trip explanation on ns-3d-grid (#335)
- Explain PICT's ns-grid VJP cost anomaly (#354)
- Explain OpenFOAM's TGV error at low viscosity (#357)
- *(harness)* Stop claiming that CI cost rankings carry over to other GPUs (#363)
- Restructure the docs around one positioning, with one results page per setup (#343)

### Performance

- *(solver)* Cache JAX-FEM thermal problem setup across calls (#195)
- *(solver)* Faster torch-fem load assembly, keep gradients at rho bounds (#200)
- *(solver)* Replay a cached adjoint tape in the Firedrake tesseracts (#229)
- *(solver)* Stop the deal.II heat CG solve at a relative residual of 1e-8 (#266)
- *(solver)* Solve JAX-FEM's linear systems with a BiCGSTAB compiled once per mesh (#286)
- *(solver)* Serve every deal.II request from one solver process (#297)
- *(solver)* Run INS.jl on its CUDA backend (#307)
- *(solver)* Stop torch-fem's GPU CG at a relative residual of 1e-6 (#301)
- *(solver)* Precondition the deal.II heat CG with algebraic multigrid (#305)
- *(solver)* Use the tutorial pressure tolerance and run OpenFOAM's channel serially (#269)
- *(solver)* Loosen the channel pressure CG of Warp-NS and PhiFlow (#308)
- *(solver)* Solve OpenFOAM's periodic pressure with GAMG (#306)
- *(solver)* Use phiml's default CG tolerance in PhiFlow's periodic pressure solve (#296)
- *(solver)* Decompose OpenFOAM's periodic cases over the physical cores (#314)

### Testing

- Check partial reruns against a full run, and fix reruns of experiments with callable params (#256)

## [0.2.0] - 2026-08-20

### Features

- Surface numeric metric regressions and scope PR benchmark diffs (#137)
- Implement checkpointing for XLB (#135)
- Implement auto-rescaling to fix anisotropy of thermal mesh (#140)
- Compare release PRs against the previous release (#153)

### Bug Fixes

- Stop overlapping text in topopt_3d figure (#91)
- External contributor friendly doc builds (#111)
- Update installation instructions (#114)
- Validate -s solver names against all problems, not just -p (#131)
- Implement warp suggestions (#132)
- *(harness)* Diagnose common Docker setup failures instead of dumping stderr (#126)
- *(harness)* Report the anomaly status from min_cosine (#141)
- Use proper reference for 3D TGV (#136)
- Publish race (#146)
- Result merge and plotting (#151)
- *(harness)* Apply the forward suite's status checks (#148)
- Status-check overrides replace suite defaults (#152)

### Documentation

- Link the Mosaic arXiv paper in the README (#88)
- Fix solver reference pages dropping all canonical IO fields (#84) (#90)
- Add which solver is best section to readme (#101)
- Surface numerical ranking for 3D NS optimization (#86) (#92)
- Correct INS.jl solver description and exclusion rationale (#102)
- Add disclaimer on llm usage (#134)

### Performance

- Upgrade INS.jl tesseract to v5.0.0 (native ν gradient, in-place forward, float32) (#103)

## [0.1.1] - 2026-06-25

### Bug Fixes

- Fix jax-fem; make sure full tracebacks get forwarded to PR status reports (#81)

### Refactor

- Upgrade tesseract core to v1.10.0 (#75)

### Documentation

- Better readme + docs (#74)

### Testing

- Trigger jax-fem build (#82)

## [0.1.0] - 2026-06-17

### Features

- *(run)* Add --continue to resume after crashes (#25)
- *(run)* Result_partial.json checkpointing + fd_check PoC (#26)

### Bug Fixes

- Cap drag_opt max_iters to keep optimization runner ≤ 6h (#56)
- Fix all tesseract builds (#62)

### Refactor

- Delete code ratio logic and `tesseracts` command (#47)
- Keep only one copy of shared code (#44)
- Simpler result format + artifact-based result storage (#65)

### Documentation

- Better docs rendering (#72)

### Performance

- *(lbfgs)* Jit the L-BFGS update+linesearch step (#19)
- Port _project_divergence_free to jax.numpy (#20)
- *(pict, drag_opt)* Halve PISO correctors and drag_opt horizon (#57)

### Testing

- Better test coverage (#52)

<!-- generated by git-cliff -->
