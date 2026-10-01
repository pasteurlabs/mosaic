# Literature-guided solver-in-the-loop experiments

The question is whether full solver gradients improve held-out rollouts over
both a strong supervised baseline and recurrent training with stopped gradients.
A favorable domain is useful; an outcome-dependent metric or omitted baseline is
not. All previous campaigns and unsuccessful configurations remain reportable.

## Evidence and differences from this benchmark

List et al., *Differentiability in Unrolled Training of Neural Physics Simulators
on Transient Dynamics*, CMAME 433 (2025), 117441:
https://doi.org/10.1016/j.cma.2024.117441
Accessible methods: https://arxiv.org/html/2402.12971v2
Code: https://github.com/tum-pbs/unrolling

Their ONE/NOG/WIG distinction corresponds to our supervised/stopped/full arms.
The wake experiments use a 1→4→16 curriculum with learning-rate reduction.
Kolmogorov flow uses sustained sinusoidal forcing, burn-in, a 1→2→4 curriculum,
rates 1e-4→1e-5→1e-6, and 144,000 updates per stage. Consequently our short,
unforced-flow runs are not replications of their successful fluid experiments.
Their ablations also show sensitivity to the learning-rate schedule and that
longer horizons can become counterproductive. Physical time matters as well as
step count. See Sections 3.2–3.3, 4.1 and Appendix F of the linked preprint.

The authors' January 2025 follow-up reports stronger KS correction results after
longer training with a plateau scheduler. This is an author-posted follow-up,
not a separate peer-reviewed study:
https://ge.in.tum.de/2025/01/15/unrolling-neural-operators-with-and-without-gradients/

The turbulence study below additionally supports choosing an optimization
horizon relative to the flow's characteristic timescale:
https://doi.org/10.1017/jfm.2022.738

## Frozen next experiment: curriculum versus a fixed long horizon

Campaign: `pr116-curriculum-20261001`. This tests an adaptation of the training
recipe on our existing flow; it is not a reproduction of the paper's physics.

- Solvers: JAX-CFD and INS-JL; model seeds 0, 1, 2.
- Resolution: coarse 64², same-solver reference 192², time-only reference audit.
- Preserve current ICs, physical coefficients, four native steps per correction,
  24-interval training trajectories and 48-interval held-out evaluation.
- Pretrain each model for 3,000 supervised updates, eight independent pairs per
  update, lr=1e-4. Use the same pretraining recipe for both schedules.
- Curriculum: 1,000 fine-tuning updates at horizon 4, lr=1e-5, then 1,000 at
  horizon 16, lr=1e-6.
- Control: 1,000 fine-tuning updates at horizon 16, lr=1e-5, then 1,000 at
  horizon 16, lr=1e-6.
- Within each job, clone the pretrained model into continued supervision,
  stopped-gradient recurrence and full recurrence. Reset Adam at the branch
  point; preserve its state across curriculum stages. All arms get the same
  rate schedule, sampled windows, examples per update and update budget.
- Twelve GPU jobs total. The pool's four-hour limit bounds this exploratory
  budget; it is much shorter than the published convergence studies.

The control isolates gradual horizon growth at equal optimizer updates and
learning-rate schedule. It does NOT equalize target examples or compute between
schedules: curriculum uses 20,000 targets per arm, fixed horizon uses 32,000.
Both schedules are reported; no winner is selected using held-out test data.
We also report the pretrained and native baselines, error curves, fields, cost,
finite-difference gradient checks, all model seeds and failures. Longer runs
are needed before making a convergence claim.

Submission after freezing source and passing cluster validation:

```sh
# Common flags: --solvers jax-cfd,ins-jl --regimes 64:4:16 --seeds 0,1,2
# --pretrain-updates 3000 --pretrain-unroll 8 --updates 2000
# Curriculum:
--curriculum 4:1000:1e-5,16:1000:1e-6
# Matched fixed-horizon control:
--curriculum 16:1000:1e-5,16:1000:1e-6
```

## Next physical benchmark, if a separate replication is needed

A forced periodic Kolmogorov case is the closest published Navier–Stokes target.
Implement the force consistently in reference and coarse stepping, check native
state reconciliation, regenerate burned-in trajectories, then admit each solver
through closure and reference checks. Keep 64²→192² for all solvers as requested;
this differs from the paper and must be labeled an adaptation. A separate KS
positive control can use the authors' code to test the published learning setup
without claiming it establishes an advantage for our Navier–Stokes adapters.
Neither physical extension is part of this frozen campaign.
