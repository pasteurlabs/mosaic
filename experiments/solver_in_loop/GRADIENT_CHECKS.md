# Checking the initial-model derivative

A directional finite difference at one perturbation can mix derivative error,
nonlinearity over the perturbation and floating-point cancellation. Keep the
original check and inspect a refinement curve when its discrepancy is large.

The Warp refinement results are recorded in `WARP_REPLICATION.md`. Those probes
check the initial, untrained model. Repeating supervised pretraining separately
is different: the JAX-CFD warm-start seed-1 repeats did not reproduce the original
parameter hash, despite matching source, image, configuration and dataset hash.
Within each run, all three methods still share the same immutable pretrained
model. Across these repeats, the errors do not isolate the finite-difference
step size:

| Run                 | Epsilon | Relative discrepancy | Checkpoint matches original |
| ------------------- | ------: | -------------------: | --------------------------- |
| Original warm-start |    0.01 |              0.22253 | yes                         |
| Independent repeat  |   0.003 |              0.24120 | no                          |
| Independent repeat  |   0.001 |              0.12720 | no                          |

The new `training.fd_epsilons` option evaluates additional perturbations at the
**same initial model, sampled window, gradient and random direction**, before
any optimizer update. It records each epsilon, autodiff directional derivative,
finite difference and relative discrepancy in `end_to_end_fd_checks_by_seed`.
The existing scalar metric still uses `training.fd_epsilon`; additional checks
do not replace it with the most favorable result. The default remains 0.01.

A test with a known nonlinear loss verifies convergence and confirms that the
extra checks do not change the optimizer update. Cluster validation passed
693 tests, with three skips. Numerical work stays on the cluster.

The first fixed-model study is job 2858869 in
`pr116-fixed-gradientcheck-20261002`: JAX-CFD, model seed 1, 3,000 supervised
pretraining updates and one subsequent update, with epsilon 0.01 plus
0.003, 0.001, 0.0003 and 0.0001. It is a new repeat of the protocol, not a
reconstruction of the missing original checkpoint. This diagnostic is not a
training-quality comparison.

The fixed-model study completed. The autodiff directional derivative was
-0.0021608025 at every step; the finite differences did **not** converge:

| Epsilon | Finite difference | Relative discrepancy |
| ------: | ----------------: | -------------------: |
|    0.01 |     -0.0019237399 |              0.05804 |
|   0.003 |     -0.0014677644 |              0.19100 |
|   0.001 |     -0.0152811399 |              0.75223 |
|  0.0003 |      0.0000496705 |              1.00000 |
|  0.0001 |     -0.0544637442 |              0.92368 |

The primary check exceeds the unchanged 5% threshold. The erratic smaller-step
values do not establish derivative correctness or identify the source of the
mismatch. Keep this run and the original warm-start failure flagged. The Warp
results are independent of this JAX-CFD diagnostic.
