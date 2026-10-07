# XLB correction and local collision diagnostics

These are diagnostic measurements, not a new confirmed training result or proof that XLB autodiff is incorrect. All numerical work ran on Slurm B200 allocations. Existing failed admission results and thresholds remain unchanged.

The independent audit starts from the exact saved canonical forced-flow field `train[0,1]` of the XLB reference dataset, regenerates five native states with 15 internal substeps, and retains those states. The original dataset SHA is recorded in `independent/outcome.json`; the small initial field is included as `initial.npy`. These regenerated states are not claimed to be the exact prior failing native trajectory.

For the default float64 internal path, zero-correction reconciliation changes internal populations by at most 5.55e-17; serialized float32 populations remain exactly unchanged. Reassociating the equilibrium subtraction makes the internal no-op exact. A prescribed small correction conserves density to 4.44e-16 and matches the requested lattice velocity change to 9.37e-17. This does not support reconciliation roundoff as the main cause of the observed long-horizon gradient failure.

The full local collision Jacobian recomputes density, velocity and equilibrium from every perturbed population vector. At fixed cell (17,23) after the first regenerated frame, KBC's Jacobian has spectral norm 28.84 and eigenvalue magnitude 15.74. Restricted to zero-mass/zero-momentum perturbations, its norm is 20.05 and the same eigenvalue persists. BGK's restricted norm/eigenvalue is 0.99917. These are local kinetic-conditioning measurements, not eigenvalues of the full spatial/timestep trajectory.

Pure-float64 KBC AD agrees with sufficiently small finite differences: the restricted check at epsilon1e-9 has relative error2.60e-6. A much larger epsilon can instead show a large discrepancy because the local map is highly nonlinear on that scale. These extra epsilon sweeps diagnose conditioning; they do not replace or relax the registered model-parameter admission check at epsilon0.001.

The separate collision controls keep the same 15-substep physical evolution and zero model initialization. Reassociated KBC still has a NaN h16 gradient. BGK has a finite h16 gradient and passes that diagnostic's original epsilon0.001 check (relative error0.0001918); its 48-frame native error is0.04324. This is promising evidence for a distinct collision variant, not full solver admission: it still requires its own frozen image, fresh same-variant reference audits, formal training-window check, and long forced-forward validation.

Sources, raw outputs, configurations, hardware/logs and SHA256 manifests are retained. No gradient clipping or stop-gradient workaround was used. Jobs: independent audit2911806; restricted kinetic audit2911818; collision controls2911795/2911796; source snapshot2911824.
