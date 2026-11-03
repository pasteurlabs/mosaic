# Solver-in-the-loop training on Kolmogorov flow

This directory and the code it runs reproduce a study of training a learned corrector _through_ each differentiable ns-grid solver. The study did not become a benchmark. Every solver that can run the flow passed every training check, so the setup did not tell solvers apart beyond the forward cost that `forward/kolmogorov` already measures. The one defect it surfaced, PICT's adjoint tolerance, is caught as well by `gradient/chain_fd_check`, which needs neither a corrector nor native solver state.

Warp and PhiFlow are not part of the study. Warp diverges on the turbulent flow at every grid. PhiFlow's 256² run, which would be its reference, turns non-finite before t = 4 from two of the three test states in `forward/kolmogorov`.

## What runs

- **Native solver state** (JAX-CFD, INS.jl, XLB): the canonical input gains `state` and `return_state`, so a chain of calls continues a staggered or lattice solver exactly instead of restarting it from the cell-centred velocity. `closure.py` measures what that changes.
- **`optimization/kolmogorov_sil`**: a small periodic CNN corrects the solver's coarse run (64², dt = 0.01) every 0.1 time units. It is trained against that solver's own 256² reference in three arms with identical windows, updates and initialisation:
  - _supervised_: one-interval pairs, with no solver in the graph;
  - _stopped_: a corrected K-interval rollout, with the gradient stopped at every solver output;
  - _sil_: the same rollout, differentiated through the solver.

  It gates on a reference audit against a 512² run, closure of chained calls, the gradient of the training loss through 16 intervals against finite differences, and every arm completing with finite losses. It records the wall time per update of each arm. This is the size meant for CI: K = 4, 32 updates, one seed.

- **`optimization/kolmogorov_sil_stability`**: the same experiment at K = 16, 256 updates and three seeds, scored on three test states. Every trained corrector, as selected by validation and after its last update, is then rolled out for 200 intervals. It takes about half an hour per solver on an A100, except PICT, which takes most of a day. Run it explicitly with `-e`.

## Reproducing

```bash
# CI-size feasibility and cost (minutes per solver)
python -m mosaic.benchmarks.cli run -o results/sil -p ns-grid \
  -e optimization/kolmogorov_sil -s 'jax-cfd,INS.jl,XLB,PICT,Exponax'

# Long training and stability rollouts, one solver per run
for s in jax-cfd Exponax XLB INS.jl PICT; do
  python -m mosaic.benchmarks.cli run -o results/stability-$s -p ns-grid \
    -e optimization/kolmogorov_sil_stability -s $s --no-plots
done

python studies/kolmogorov-sil/summarize.py results/sil results/stability-*

# Closure of chained calls, restarting from the velocity and with native state
python studies/kolmogorov-sil/closure.py latest jax_cfd ins xlb pict exponax
python studies/kolmogorov-sil/closure.py latest jax_cfd ins xlb --state
```

`mosaic run` replaces a `result.json` rather than merging solvers into it, so run the stability experiment into one directory per solver.

To see the PICT defect, build PICT without #294 (`git revert 3d480d0`). Then the FD gate of `optimization/kolmogorov_sil` fails, and `gradient/chain_fd_check` reports a median error near 0.9 with a cosine near 0.5.

## Results

All results are from one A100 80 GB, with images built from this code merged with `main` on 2026-10-28 and PICT with #294.

### Closure

These are relative L2 distances (median / max) over the three test states, between a chain of one-interval calls and one call over the same time.

| Solver  | Restart from velocity: 2 intervals | 16          | 40         | Native state: 2 | 16   | 40          |
| ------- | ---------------------------------- | ----------- | ---------- | --------------- | ---- | ----------- |
| JAX-CFD | 1.0% / 1.0%                        | 13% / 14%   | 55% / 64%  | 0               | 0    | 0           |
| INS.jl  | 1.0% / 1.0%                        | 13% / 14%   | 55% / 63%  | 0               | 0    | 0           |
| XLB     | 0.53% / 0.55%                      | 2.3% / 2.7% | 9.7% / 11% | 1e-6            | 1e-6 | 3e-6 / 5e-6 |
| PICT    | 0                                  | 0           | 0          | —               | —    | —           |
| Exponax | 1e-6                               | 7e-6        | 5e-5       | —               | —    | —           |

PICT and Exponax are collocated and carry no history between steps, so they restart exactly without native state.

### `optimization/kolmogorov_sil` (2026-10-28, idle GPU)

| Solver  | Audit ratio | Closure | FD error, zero / nonzero readout | s per update, supervised / stopped / sil |
| ------- | ----------- | ------- | -------------------------------- | ---------------------------------------- |
| JAX-CFD | 0.074       | 0       | 1.7e-4 / 2.4e-4                  | 0.01 / 0.17 / 0.22                       |
| Exponax | 0.003       | 4.5e-5  | 1.5e-3 / 1.9e-4                  | 0.01 / 0.17 / 0.22                       |
| XLB     | 0.032       | 7.5e-6  | 1.9e-3 / 1.9e-4                  | 0.01 / 0.19 / 0.30                       |
| INS.jl  | 0.075       | 0       | 1.7e-4 / 2.4e-4                  | 0.01 / 0.22 / 0.49                       |
| PICT    | 0.093       | 0       | 1.2e-4 / 2.3e-4                  | 0.01 / 1.21 / 2.66                       |

All five pass every gate, with gradient errors far below the 5% threshold.

### `optimization/kolmogorov_sil_stability` (2026-10-29 to 30)

Mean relative test error over 40 intervals per model seed, with the uncorrected solver's in brackets. Each arm has 18 long rollouts: three seeds, three test states, selected and final model.

| Solver  | Supervised        | Stopped           | SIL               | Uncorrected | Non-finite rollouts | Peak energy / uncorrected mean |
| ------- | ----------------- | ----------------- | ----------------- | ----------- | ------------------- | ------------------------------ |
| JAX-CFD | 0.117 0.100 0.108 | 0.151 0.161 0.167 | 0.193 0.151 0.189 | 0.243       | 0 of 54             | 1.07–1.34                      |
| INS.jl  | 0.117 0.100 0.108 | 0.151 0.161 0.167 | 0.193 0.151 0.189 | 0.243       | 0 of 54             | 1.06–1.35                      |
| Exponax | 0.068 0.075 0.067 | 0.063 0.061 0.065 | 0.076 0.068 0.063 | 0.091       | 0 of 54             | 1.06–1.37                      |
| XLB     | 0.069 0.061 0.063 | 0.078 0.085 0.068 | 0.084 0.073 0.102 | 0.086       | 0 of 54             | 1.14–1.50                      |
| PICT    | 0.145 0.143 0.147 | 0.186 0.169 0.150 | 0.228 0.161 0.192 | 0.277       | 0 of 54             | 1.10–1.36                      |

- All 45 trainings complete their 256 updates. SIL's gradient norms are 10–20× the stopped arm's but stay bounded.
- No arm makes a corrector less stable than the uncorrected solver, whose own peak energy is 1.09–1.45 times its mean over the same rollouts.
- Averaged over the three seeds, supervised training gives the lowest error on every solver except Exponax, where the stopped arm is lower, and training through the solver is never better than stopping the gradient at the solver. On single seeds of JAX-CFD, INS.jl, PICT and Exponax it is.
- Seconds per update at K = 16 (supervised / stopped / sil): JAX-CFD 0.02 / 0.69 / 1.00, Exponax 0.02 / 0.75 / 1.02, XLB 0.02 / 0.85 / 1.38, INS.jl 0.02 / 0.87 / 2.35, PICT 0.02 / 4.73 / 26.8. These runs partly shared the GPU with other jobs, so treat them as rough.

JAX-CFD and INS.jl agree to every printed digit throughout, as they do in the forward checks.
