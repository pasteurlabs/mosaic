# Supervised pretraining followed by solver-in-the-loop fine-tuning

Question: can differentiating through the solver improve a useful supervised
corrector, beyond continued supervision or recurrent training without solver
gradients?

The first campaign uses JAX-CFD and INS-JL, three model seeds (0, 1, 2), and
fine-tuning horizons of 8 and 16 correction intervals: 12 independent GPU jobs.
Each job first trains with supervision for 3,000 updates using eight pairs per
update. All three branches receive that exact immutable model, reset Adam, and
train for 1,000 additional updates with matched sampled windows. Pretraining
uses RNG seed 2025; fine-tuning uses 2026. The pretraining horizon stays eight
for both fine-tuning horizons. There is no test-based checkpoint selection.

The physical task remains unchanged: coarse 64², own-solver reference 192²,
coarse dt 0.02, reference dt/3, four coarse steps per correction, viscosity
0.001, multimode amplitude 0.5 and k0=4. Eight training ICs have 24 intervals;
four held-out ICs have 48. The reference audit compares 192² at dt/3 and dt/6;
it checks time accuracy of the finite-resolution target, not spatial convergence.

The same three-layer periodic CNN and divergence-free correction are used in
all arms. Loss is the mean squared relative velocity error over each window,
normalized by a shared native-solver baseline. Adam uses lr=1e-4 and global
norm clipping at 5. The three arms are continued fixed-pair supervision,
recurrent training with stopped solver gradients, and full recurrent training.

Artifacts record pretraining losses, held-out pretrained rollouts, checkpoint
hashes, fine-tuning losses and all-arm held-out rollouts. Training cost includes
shared pretraining for each method when comparing total cost; raw common and
incremental costs are also retained. Budgets match updates and examples within
each horizon, not wall time, and horizon 16 sees more targets per update than 8.

Primary comparisons are held-out mean relative L2 over all noninitial frames:
full versus continued supervision and full versus stopped gradients. Native
and pretrained rollouts show whether fine-tuning improves the starting point.
The existing report pools model seeds and bootstraps paired seeds and shared
ICs. All seeds and failures must be reported. These are exploratory runs, not
an independent confirmation set.

Run through `submit.py --pretrain-updates 3000 --pretrain-unroll 8 --updates 1000
--regimes 64:4:8,64:4:16 --seeds 0,1,2 --solvers jax-cfd,ins-jl` against a frozen
campaign source, after cluster validation. Numerical tests, training and plots
run on the cluster, not the development machine.
