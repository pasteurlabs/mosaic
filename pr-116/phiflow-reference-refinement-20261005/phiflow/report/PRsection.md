### Frozen corrector transfer: phiflow

The INS-selected model settings are reused unchanged. All three methods use B200 GPUs, the same 64² solver and solver-specific 192² references. Temporal refinement is solver-specific to meet the common 0.5% reference-admission criterion. Every training and test IC must pass admission.

Both supervision and matched stopped-gradient comparisons favor full solver gradients.

| Method | Mean held-out relative L2 error |
|---|---:|
| full | 0.022445 |
| stopped | 0.025048 |
| supervised | 0.032690 |
| native | 0.058509 |

Paired model/IC bootstrap ratios below 1 favor full gradients:

- Full/supervised: 0.6866 (95% CI 0.6579–0.7202).
- Full/stopped: 0.8961 (95% CI 0.8542–0.9430).
- Full/native: 0.3836 (95% CI 0.3650–0.4034).

![Paired rollout errors](ARTIFACT_URL/report/rollout-and-paired.png)
![Full vorticity fields](ARTIFACT_URL/report/fields.png)
![Full velocity errors](ARTIFACT_URL/report/field-errors.png)
![Same-hardware training cost](ARTIFACT_URL/report/training-cost.png)

These are within-solver, same-hardware comparisons. They do not establish equal DNS accuracy across solvers or compare B200 runtimes with the earlier RTX5090 INS result.
