### Final bounded neural-corrector comparison

**Solver-in-the-loop beats tuned supervision in this registered setting.**


The matched ablation supports a benefit from solver derivatives at the selected recipe.

| Method | Mean rollout error ↓ |
|---|---:|
| Full solver gradients | 2.131% |
| Matched stopped gradients | 2.378% |
| Tuned stopped gradients | 2.378% |
| Tuned supervision | 3.035% |
| Solver only | 5.639% |

Full/supervised error ratio: **0.702**, paired 95% interval [0.649, 0.761].

Matched stopped-gradient ratio: **0.896**, paired 95% interval [0.868, 0.924].

Selection used 16 separate validation flows and three model seeds. Confirmation uses eight fresh model seeds and 32 untouched flows. Intervals resample paired model seeds and shared flows, not time frames. The positive threshold was a ≥5% mean reduction and a 95% interval excluding no improvement.

The same INS solver supplies the 192² reference restricted to 64². The physical task, reference checks and model architecture are unchanged. Search covers learning rates, unroll lengths, update budgets and supervised warm starts; it is not an exhaustive impossibility test.

![Rollout and paired results](ARTIFACT_URL/report/rollout-and-paired.png)

![Full-domain fields](ARTIFACT_URL/report/fields.png)

![Spatial velocity errors](ARTIFACT_URL/report/field-errors.png)

Mean training cost per model (including warm starts and supervised-pair generation):

- Full solver gradients: 168.4 minutes

- Matched stopped gradients: 93.1 minutes

- Tuned stopped gradients: 93.1 minutes

- Tuned supervision: 8.8 minutes

![Training costs](ARTIFACT_URL/report/training-cost.png)

Shared reference generation, evaluation and hyperparameter search are separate costs; equal updates do not imply equal compute.

| Selected recipe | Learning rate | Unroll | Updates | Supervised start |
|---|---:|---:|---:|---|
| Full solver gradients | 0.0001 | 16 | 3000 | None |
| Matched stopped gradients | 0.0001 | 16 | 3000 | None |
| Tuned stopped gradients | 0.0001 | 16 | 3000 | None |
| Tuned supervision | 1e-05 | 8 | 1000 | None |

Recorded tuning fit time: 60.18 GPU-hours; shared preparation, diagnostics and evaluation are additional.

[Frozen protocol, every candidate, failures, costs and numerical results](ARTIFACT_TREE). Earlier negative correction and neural-control results remain retained in this PR; direct control optimization is a separate non-neural result.
