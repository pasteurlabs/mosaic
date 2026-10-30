# Finer-time reference sensitivity

The same45frozen predictions were scored against the preexisting N128 dt0.00125 audit; no model or native solver was called, and no choice changed. Both targets are restricted to N64. This checks temporal-reference sensitivity, not spatial-reference convergence.

| Group | Standard mean paired ratio | Finer-time audit ratio | Seed-mean winner flips | Case winner flips |
|---|---:|---:|---:|---:|
| joint | 0.866420 | 0.884273 | 0 | 1 |
| joint_zero_mean | 0.926967 | 0.930952 | 0 | 0 |

Ratios average three paired seed ratios; each seed ratio uses mean loss on the same five ICs. All45primary metrics recomputed identically to the original report within1e-12 relative.

![Temporal-reference sensitivity](reference_sensitivity.png)
