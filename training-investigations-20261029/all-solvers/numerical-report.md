# All-solver diagnostic comparison

Five own-solver maps; common H8/256-update Adam .003 checkpoints with validation every 16 updates. No retraining.

H denotes a rollout interval of eight native timesteps at dt=0.005. H100 therefore spans physical time 4. Training uses four ICs, N64 and own-solver N128 references. All existing checkpoints use one nominal model seed; validation cadence 16 differs from public cadence 64.

These diagnostics do not change the public recipe or admission thresholds. The earlier JAX-only 60-second H8/H16 models are a separate study.

## Completion

- JAX-CFD parameter: completed=True, rows=6, job=3620642; failure=none reported
- JAX-CFD operator: completed=True, rows=486, job=3620861; failure=none reported
- PhiFlow parameter: completed=True, rows=6, job=3620829; failure=none reported
- PhiFlow operator: completed=True, rows=486, job=3620870; failure=none reported
- INS.jl parameter: completed=True, rows=6, job=3620744; failure=none reported
- INS.jl operator: completed=True, rows=486, job=3620918; failure=none reported
- Oceananigans parameter: completed=True, rows=6, job=3620745; failure=none reported
- Oceananigans operator: completed=True, rows=486, job=3620930; failure=none reported
- WaterLily parameter: completed=True, rows=6, job=3620924; failure=none reported
- WaterLily operator: completed=True, rows=486, job=3620954; failure=none reported

## Selected checkpoint, held-out IC100: controlled parameter prefixes

Cosines compare raw parameter gradients against H100. Actual step changes use parameter L2=1e-4; negative means a lower H100 loss. These are three fixed horizon directions at one checkpoint and IC per solver, not a population estimate or a new recipe selection.

| Solver | H8 cosine | H16 cosine | H32 cosine | H8 step (%) | H16 step (%) | H32 step (%) |
|---|---:|---:|---:|---:|---:|---:|
| JAX-CFD | 0.3154 | 0.1704 | 0.05819 | -0.5302 | -0.2288 | 0.001943 |
| PhiFlow | 0.1839 | 0.2111 | 0.8606 | -0.1139 | -0.1308 | -0.5405 |
| INS.jl | -0.005565 | -0.09269 | -0.4115 | 0.07422 | 0.333 | 1.284 |
| Oceananigans | -0.08181 | -0.01457 | 0.2059 | 0.9775 | 0.4063 | -1.387 |
| WaterLily | -0.6615 | -0.7017 | 0.2316 | 0.5375 | 0.5775 | -0.1656 |

## Selected checkpoint, developed held-out state: H32

All entries are relative to the same backend’s uncorrected path, IC100/frame50/basis0. The spectrum is output-restricted (24 cotangents); a smaller restricted ratio is not proof of a better-conditioned full solver. Forecast error uses the backend’s own fine reference.

| Solver | Largest gain ratio | Smallest gain ratio | Restricted-ratio ratio | Forecast MSE ratio |
|---|---:|---:|---:|---:|
| JAX-CFD | 1.005 | 0.972 | 1.034 | 0.5227 |
| PhiFlow | 1.05 | 1.564 | 0.6709 | 0.3114 |
| INS.jl | 0.9907 | 0.9727 | 1.019 | 0.5895 |
| Oceananigans | 1.002 | 1.052 | 0.9519 | 0.943 |
| WaterLily | 1.006 | 1.021 | 0.9852 | 0.8895 |

## Three-state robustness at H32

Basis0 only; every value is relative to the corresponding uncorrected path. The two other primary-state bases do not create additional forecast IC samples.

| Solver | State | Model | Largest gain ratio | Smallest gain ratio | Restricted-ratio ratio | Forecast MSE ratio |
|---|---|---|---:|---:|---:|---:|
| JAX-CFD | Held-out100 / frame50 | selected | 1.005 | 0.972 | 1.034 | 0.5227 |
| JAX-CFD | Held-out100 / frame50 | final | 0.9995 | 0.9844 | 1.015 | 0.5572 |
| JAX-CFD | Held-out100 / frame25 | selected | 1.006 | 1.132 | 0.8889 | 0.5573 |
| JAX-CFD | Held-out100 / frame25 | final | 0.9983 | 1.138 | 0.8769 | 0.6802 |
| JAX-CFD | Validation1 / frame50 | selected | 1.005 | 1.261 | 0.7964 | 0.5586 |
| JAX-CFD | Validation1 / frame50 | final | 0.9995 | 1.146 | 0.8722 | 0.6 |
| PhiFlow | Held-out100 / frame50 | selected | 1.05 | 1.564 | 0.6709 | 0.3114 |
| PhiFlow | Held-out100 / frame50 | final | 1.035 | 1.477 | 0.7012 | 0.3038 |
| PhiFlow | Held-out100 / frame25 | selected | 1.05 | 1.457 | 0.7204 | 0.3313 |
| PhiFlow | Held-out100 / frame25 | final | 1.035 | 1.433 | 0.722 | 0.4048 |
| PhiFlow | Validation1 / frame50 | selected | 1.055 | 1.299 | 0.8126 | 0.2991 |
| PhiFlow | Validation1 / frame50 | final | 1.041 | 1.332 | 0.7812 | 0.2722 |
| INS.jl | Held-out100 / frame50 | selected | 0.9907 | 0.9727 | 1.019 | 0.5895 |
| INS.jl | Held-out100 / frame50 | final | 0.993 | 0.9836 | 1.01 | 0.6433 |
| INS.jl | Held-out100 / frame25 | selected | 0.987 | 0.9619 | 1.026 | 0.6295 |
| INS.jl | Held-out100 / frame25 | final | 0.9882 | 0.9775 | 1.011 | 0.6559 |
| INS.jl | Validation1 / frame50 | selected | 0.9897 | 0.9685 | 1.022 | 0.7798 |
| INS.jl | Validation1 / frame50 | final | 0.9926 | 0.9912 | 1.001 | 0.8167 |
| Oceananigans | Held-out100 / frame50 | selected | 1.002 | 1.052 | 0.9519 | 0.943 |
| Oceananigans | Held-out100 / frame50 | final | 1.002 | 1.054 | 0.9509 | 0.7877 |
| Oceananigans | Held-out100 / frame25 | selected | 1.004 | 1.123 | 0.8941 | 0.8342 |
| Oceananigans | Held-out100 / frame25 | final | 1.002 | 1.118 | 0.8961 | 0.9047 |
| Oceananigans | Validation1 / frame50 | selected | 1.001 | 1.071 | 0.9344 | 0.9861 |
| Oceananigans | Validation1 / frame50 | final | 1 | 1.066 | 0.938 | 0.8982 |
| WaterLily | Held-out100 / frame50 | selected | 1.006 | 1.021 | 0.9852 | 0.8895 |
| WaterLily | Held-out100 / frame50 | final | 1.004 | 1.036 | 0.9691 | 0.8492 |
| WaterLily | Held-out100 / frame25 | selected | 1.005 | 1.039 | 0.9673 | 0.8279 |
| WaterLily | Held-out100 / frame25 | final | 1.004 | 1.032 | 0.9723 | 0.8161 |
| WaterLily | Validation1 / frame50 | selected | 1.006 | 1.062 | 0.9472 | 0.9178 |
| WaterLily | Validation1 / frame50 | final | 1.005 | 1.044 | 0.9625 | 0.8589 |

## Descriptive finite-difference flags

These counts retain both epsilon points rather than selecting the favorable one. They are not pass rates for a representative population and do not establish whether an individual mismatch is derivative error, finite-step variation, or numerical precision.

| Solver | H100 parameter points >5% / total | State dot checks >5% / total | Two-ε vectors >5% / total |
|---|---:|---:|---:|
| JAX-CFD | 2/18 | 38/756 | 28/378 |
| PhiFlow | 0/18 | 6/756 | 36/378 |
| INS.jl | 0/18 | 6/756 | 21/378 |
| Oceananigans | 1/18 | 7/756 | 23/378 |
| WaterLily | 0/18 | 3/756 | 29/378 |

## Interpretation limits

- One model seed; two reference ICs for parameter analysis; five developed-state/basis cases for operator analysis.
- Parameter prefixes start at frame0; these are not the sampled reference-reset training-window gradient distribution or clipped Adam updates.
- Each backend has its own numerical reference and RMS coordinate normalization; raw singular values are not a cross-solver accuracy ranking.
- 24-dimensional output-cotangent restriction; no full-Jacobian condition number or certified smallest singular value.
- Physical forward vectors are centered finite differences, not JVPs. All epsilon disagreements and low-response dot discrepancies retained.
- CFL observations cover sampled interval/provisional/corrected/perturbed endpoints, not every native microstep.
- Initial hidden weights may differ at CPU/GPU roundoff; measured discrepancies retained.
- Earlier JAX H16/60-second experiments are excluded from this common256 comparison.
- Operator trajectories restart once from a developed reference state and then evolve autonomously; their 32-interval errors differ from the full 100-interval benchmark objective.
- DCT energies use packed component arrays. Low-band energy includes DC (radius<=4), while all selected probing modes exclude DC; broad energy bands are not the exact finite probe subspaces.
- There are three distinct developed states. Cases0/1/2 share the same forecast trajectory and differ only in probing basis; they are not independent IC samples.

All scalar finite-difference values, absolute discrepancies, denominators, and two-epsilon vector disagreements are retained in report.json. Five-percent plotting flags are descriptive, not retrospective admission rules.

Blank plots mean unavailable data, not zero sensitivity. The raw directory retains every latest outcome; report-input.json identifies preserved failed attempts.

## Same-probe local gradient decomposition

[Full per-probe values](local_gradient_decomposition.json) and [figure](local_gradient_changes.png) quantify CᵀQ−Q and SᵀCᵀQ−SᵀQ from the saved arrays, without new solver calls. These local derivatives use the same probes; they are not added to obtain the composed gradient. Norm changes and directional cosines are descriptive, not gradient correctness tests. The velocity-only corrector leaves the immediate temperature output unchanged, so local temperature-probe changes should be near zero; recurrent temperature effects can still arise through subsequent solver steps.
