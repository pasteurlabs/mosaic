# ns-grid converged references

## `forward_kolmogorov.npz`

The ground truth of the Kolmogorov-flow experiments: the flow on [0, 2π)² driven by the force (sin 4y, 0) against linear drag 0.1 at viscosity 2·10⁻³ (`kolmogorov.py` holds the setup).

| Array                                                        | Contents                                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| `initial_<seed>`                                             | Burned-in state, (128, 128, 1, 2)                                              |
| `truth_<seed>`                                               | Trajectory at t = 1, 2, 3, 4, restricted to 64², for the test seeds 5, 6 and 7 |
| `convergence_correlation_<seed>`, `convergence_error_<seed>` | 512² run against 1024², every 0.5 time units up to 12                          |
| `horizon_times`, `convergence_times`                         | The times of the trajectories                                                  |

Seeds 0–3 start the training trajectories of `optimization/kolmogorov_sil`, seed 4 its validation trajectory, and seeds 5–7 its test trajectories and those of `forward/kolmogorov`.

Each seed starts from a random field and spins up for 100 time units at 512² before it is restricted to 128². That keeps the file small and drops only modes deep in the dissipation range, which the flow regenerates within a fraction of a time unit. Solvers start from the state spectrally prolonged to their grid, and the truth starts from exactly that field at 1024². All runs are Exponax as registered in ns-grid (ETDRK2 with 2/3-rule dealiasing) at a CFL number of about 0.1.

The truth is converged over the scored horizon. The 512² run from the same state differs from it by 0.3% after one time unit and by 2.3–3.2% after four, at a vorticity correlation of 0.997 or more, and it stays above 0.95 correlation for 6.5 to 7.5 time units. Independently, JAX-CFD converges to the truth at observed orders 1.6 and 1.9 between 64² and 256² (relative L2 at t = 1: 0.086, 0.029 and 0.0077) until it reaches that 0.3% floor at 512².

Regenerate with `mosaic reference -p ns-grid -e forward/kolmogorov` (Docker and an NVIDIA GPU; about half an hour on an A100).
