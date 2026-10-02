# Direct fluid-control optimization

## Question and scope

Do derivatives through the fluid solver improve per-instance control optimization
at the same elapsed optimization budget? This is a separate experiment from
amortized neural-policy training. A positive result does not establish a benefit
for neural training, supervised learning, or deployment amortization. Retain the
previous control and policy experiments, including negative results.

Optimize the 64 bounded actuator coefficients for one initial/target pair. All
methods use the same repaired PhiFlow image, initial state, target, force basis,
control parameterization and objective. Use the existing long-horizon setup:
64² optimization, 192² target generation and evaluation, eight force intervals
with 20 coarse steps per interval, dt=.02 and T=3.2. Fine temporal factors are 3
and 6. Viscosity=.001, bound=.25, velocity scale=.5 and effort weight=.01.
The normalized effort and physical duration remain identical across methods.
Freeze source, image, configuration, task arrays and their hashes in artifacts.

## Methods and initialization

Every method starts from exactly the same linear-controller coefficients,
converted to the same realized bounded parameterization. Save the original and
realized coefficients and any clipping error. No method receives the generating
controls or an optimized demonstration as its initialization. Keep zero and
linear control as visible reference results.

Development configurations are fixed prospectively:

| Method | Settings |
| --- | --- |
| Exact solver-gradient Adam | learning rate .01 or .05 |
| Antithetic action-space SPSA Adam | learning rate .01 or .05 crossed with perturbation .01 or .05; one or four independent directions per update |
| Powell | identity direction set, xtol=.001 and ftol=.00001, latent bounds [−8,8], subject to the same time budget |

Record the precise Powell implementation/version and resolved options before
launch. Optimize identical action latent variables for all methods; perturbations
are in those variables, not in a different physical action space. SPSA uses fixed,
recorded random seeds and independent directions. Preserve every configured run.
The tuning-search cost is reported separately; eight SPSA settings versus two AD
settings is deliberately a stronger search for the gradient-free baseline.

## Time and computational work

Each run has one continuous 120-second optimization budget, with checkpoints at
30, 60 and 120 seconds. A checkpoint contains the best **evaluated** candidate
whose objective evaluation completed by its deadline. Never score an unevaluated
last Adam iterate as if it had already been found. Retain the common initial
candidate if no improvement has completed. If initialization itself has not
completed, mark the checkpoint unavailable rather than assign it a free score.

Charge method-specific engine setup, tracing/compilation, objective queries,
VJPs, optimizer work, synchronization and method-specific transfers to the clock.
Start timing before that work and synchronize before recording elapsed time.
The execution model is a shared GPU service: task preparation and linear control
can warm forward-solver kernels before optimization. Share that preparation across
all methods, perform no deliberate optimizer or AD prewarm, and charge each run's
first AD call and optimizer initialization. Record this as shared/warm-service
performance, **not cold process or cold deployment latency**. Compilation reuse
and order effects remain possible; randomize method order by a fixed recorded
rule and retain the order. Record hardware/node identities and use the same GPU
class. Any subsequent cold-latency claim requires isolated cold confirmation.

An in-flight solver operation may finish after a deadline. Charge its full time
and work, record the overrun, and exclude its candidate from that deadline's
checkpoint. A numerical failure remains visible, stops that run, and cannot
become an automatically retried optimization with a fresh budget.

Report forward solver evaluations and reverse/VJP evaluations separately, along
with optimizer updates and failures. An AD forward plus VJP is **not** one unit
of work equivalent to a black-box forward call. SPSA uses two or eight perturbed forward calls per gradient estimate for
one or four antithetic directions respectively; count any
additional candidate/incumbent scoring calls too. Powell must count every
objective call. Curves against forward/VJP counts supplement the primary
wall-time comparison; do not combine their counts using an assumed equal cost.

Shared dataset generation, goal admission and linear initialization are recorded
as separate common preparation costs, with an end-to-end view adding them to all
methods. Fine evaluation and audits occur only after all optimization runs for
the task are finished. They never provide optimization feedback, and their time
and solver calls are reported separately from the optimization clock.

## Development selection and untouched confirmation

Use development task seeds 3000–3007, distinct from previous gate and policy
splits. Select one configuration per method using mean 192² objective at the
120-second checkpoint across **all eight** tasks. Numerical failure, unavailable
checkpoint or failed fine audit makes a configuration ineligible; do not average
only its successful tasks. Report those failures and all remaining configurations.
For an exact selection tie, use the first configuration in the frozen manifest.
If a method has no eligible configuration, report that result instead of choosing
an unregistered replacement. Launch final confirmation only when all three methods
have an eligible frozen configuration; otherwise stop and retain the development
results without generating or inspecting final-test tasks.

Freeze the selections, engine implementation, numerical thresholds and plotting
choices before generating or evaluating final task seeds 5000–5015. Run only the
selected configuration for each method on these 16 untouched tasks, with the same
120-second budget and checkpoint rules. No retuning from final-test outcomes.
The final task is the statistical unit: report paired task-level differences,
means, medians, win counts and all failures. Describe this as one fixed-seed SPSA
comparison, not a comprehensive estimate of optimizer-seed variability.

## Numerical admission and reporting

Keep existing numerical thresholds: goal and evaluated-terminal temporal
relative discrepancy below .5%, and the existing primary directional-gradient
check below 5% at epsilon .001 in three fixed directions. Epsilons .01/.0001
remain supplementary diagnostics and cannot replace a failed primary check.
Perform fresh directional-gradient checks only after all optimization runs,
alongside fine evaluation, to avoid prewarming AD kernels. Keep these admission
checks outside optimization timing and disclose their cost.
Use the same temporal audit for every method, checkpoint and task. Never silently
drop a task because its baseline or a favored method fails admission.

At each checkpoint report fine terminal mismatch, effort and combined objective;
compare paired methods on the same target arrays. Save complete fields, controls,
optimization traces, solver-work ledgers, elapsed times and archive identities.
Plots show objective against charged elapsed time, paired final-task results and
full fields with shared physical scales. Fix examples prospectively to development
seed3000 and final seeds5000,5001, irrespective of which method wins. Show initial,
target, linear and all selected-method final fields and error fields. Include
zero/linear baseline values and the failed/unavailable counts beside aggregates.
