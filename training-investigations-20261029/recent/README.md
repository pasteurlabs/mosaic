# Completed follow-up research

These native Slurm investigations supplement PR #211. They do not replace its 256-update benchmark, canonical-container receipts, or baseline/SOL gallery. Solver/data/source identities, protocols, Slurm accounting and report bytes are retained in the subdirectories and `publication_manifest.json`. Large model/field NPZ files are intentionally omitted. No figures were recomputed for publication.

| Investigation | Finding | Evidence |
|---|---|---|
| Universal learning rate | Retain Adam .003. The equal-backend geometric mean of selected-validation loss ratios (.001/.003) is 1.11557 across all five completed seed-0 pairs. JAX-CFD and Oceananigans favor .001; the aggregate does not. | [Report](lr-transfer/report.json), [plot](lr-transfer/validation.png) |
| Longer recurrent windows | H16 beats H8 on validation and held-out loss for all three observed network initializations. Geometric-mean ratios are .8325 and .7721. | [Confirmation](h16-confirmation/confirmation.json), [paired plot](h16-confirmation/paired_ratios.png), [initial H8/H16/H32 screen](long-horizons/validation.png) |
| Architecture | The small U-Net is competitive, but its validation loss is 1.0388× the width-8 CNN's; its lower held-out loss is not a selection criterion. Deeper/wider CNNs did not improve this screen. | [Report](architectures/report.json), [plot](architectures/validation.png) |
| Horizon curriculum | Worse selected validation and held-out error than the matched fixed-H8 60-second trial. | [Report](curriculum/report.json), [plot](curriculum/validation.png) |
| L-BFGS | The fixed-batch trial slightly improves on matched fixed-batch Adam, but does not establish an improvement over stochastic training (different objective/windows). | [Report](lbfgs/report.json), [plot](lbfgs/validation.png) |
| Hybrid curriculum/L-BFGS | The completed hybrid is worse than fixed-H8 Adam and selected its checkpoint before L-BFGS. The fixed-H8 L-BFGS trial failed the unchanged CFL check at .50894; the failure is retained. | [Report including failed cell](hybrid/report.json), [plot](hybrid/validation.png) |
| Restricted conditioning | JAX-CFD at N64: solver, corrector and composition derivatives on a fixed 72-dimensional physical input subspace. Corrector effects are small and recipe-dependent; these are not full-Jacobian condition numbers or proof of improved AD accuracy. | [Comparison](conditioning/comparison.json), [conditions](conditioning/restricted_conditioning.png), [spectra](conditioning/restricted_spectra.png) |

## Interpretation and provenance

The universal LR decision uses validation only. Its aggregation rule was recorded after JAX-CFD and PhiFlow outcomes were visible, before the other three numerical training outcomes; it is not a blind preregistration. Each pair has matched data, initialization and schedule; the transfer uses dense 16-update validation, not the benchmark's 64-update schedule.

The horizon/architecture experiments use a 60-second active training budget on RTX hardware, with compilation reported separately, and are not 256-update CI validations or T4 timing predictions. H16 confirmation covers three network initializations, common window starts 0–68, and the same single validation and held-out physical trajectories; it does not establish population-level generalization. Slurm cgroup memory peaks (about 15 GiB in some runs) differ from process-sampler RSS; retained accounting must not be replaced by the smaller number. Shared/cached reference preparation also prevents interpreting these full-method times as cold CI runtimes.

The studies use native adapters/bridges, not a new canonical all-backend container campaign. Historical direct JAX calls bypassed per-call Pydantic checks; later trial instrumentation and original acceptance failures are preserved in their reports. The main benchmark gallery retains its own documented sources and references. New performance implementation work is still in progress and has no measured result in this publication.
