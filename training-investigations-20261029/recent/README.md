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


## Recurrent sensitivity and robustness follow-ups

[Full report and precision limits](recurrent-sensitivity/README.md) supplement the
original local 72-dimensional study with a separate **24-dimensional physical
input restriction at actual N64**, matched H8/H16 selected checkpoints, two ICs,
and genuinely different mode subsets. They are JAX-CFD-only diagnostics.
The earlier local study evaluates derivatives at common states; these recurrent
maps start together then evolve autonomously along different states. Neither
study measures neural parameter-gradient alignment.

- [Absolute singular gains and restricted ratios](recurrent-sensitivity/long_rollout_sensitivity.png)
  cover horizons 1, 4, 8, 16 and 32. At the core H32 point, solver-only σmin/σmax
  are .01090/2.840, versus .01306/2.862 for H8 and .01264/2.873 for H16.
  Both corrected paths modestly improve the restricted ratio while slightly
  increasing maximum gain. Physical high-mode damping remains strong.
- [Perturbation growth by spatial scale](recurrent-sensitivity/physical_mode_growth.png)
  and [prediction error by spatial scale](recurrent-sensitivity/spectral_prediction_error.png)
  must be read together: the correctors slightly increase high-band perturbation
  gain while reducing forecast error here. Extra damping does not explain that
  improvement. This is not a causal claim about training.
- [State/IC/subspace robustness](recurrent-sensitivity/state_basis_robustness.png)
  retains all five state/basis cases. Only 24 of 8065 physical state directions
  are sampled; raw singular minima and ratios are preserved separately from the
  descriptive rank cutoff. Public JVP precision is Float32, so the cutoff is not
  a certified accuracy floor. Finite-amplitude checks on one low-band velocity
  direction at two amplitudes differ from tangents by at most .67%; sampled
  endpoint CFL stays below .232. These checks do not independently validate
  every direction and do not waive any training admission gate.


## Parameter-gradient alignment follow-up

[Full parameter-gradient report](parameter-alignment/README.md) addresses a
different quantity from the state-Jacobian studies above: raw neural parameter
gradients of trajectory prefixes against the H100 objective at common initial
states and checkpoints. This is not the actual shuffled-window gradient
sampling distribution. The [alignment plot](parameter-alignment/alignment.png)
and [finite descent probes](parameter-alignment/descent.png) show that short
prefix directions can increase the long objective in some cases, but results
are not monotonic across horizons/checkpoints. The original two initial-model
H100 FD failures at epsilon.003 remain in the [epsilon profile](parameter-alignment/h100_fd.png);
smaller-epsilon agreement does not override prior admission.

All four new follow-ups are now complete: parameter alignment, long-rollout
state sensitivities, physical-scale growth versus prediction error, and
state/IC/subspace robustness. They remain JAX-CFD-only, separate from both the
original local72D conditioning study and the canonical all-backend CI gallery.
The restarted forecast chunks use fine-reference states at frames25/50, whereas
the training comparisons evaluate full trajectories from time0; these are
different objectives. H8 often improves the measured local/recurrent restricted
ratio more than H16 despite H16's better full-rollout training result. Conditioning
therefore does not explain the training ranking causally.
