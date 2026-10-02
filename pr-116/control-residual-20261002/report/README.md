# Developmental flow-control results

Development validation only; unequal, untuned budgets. Descriptive points, no superiority or confidence-interval claim.

Exact dataset pairing: **True**. All dataset hash groups are retained in report.json.

## Learned-policy checkpoints

| Method | Updates | Model seeds | Mean objective | Seed minimum–maximum | All admitted |
|---|---:|---:|---:|---:|---|
| Demonstration imitation | 0 | 3 | 0.026055196 | 0.026055196–0.026055196 | True |
| Demonstration imitation | 250 | 3 | 0.026814307 | 0.02622005–0.027727958 | True |
| Demonstration imitation | 500 | 3 | 0.026909466 | 0.026539952–0.027180011 | True |
| Demonstration imitation | 1000 | 3 | 0.027875172 | 0.027427426–0.028273724 | True |
| Full gradients | 0 | 3 | 0.026055197 | 0.026055196–0.026055198 | True |
| Full gradients | 100 | 3 | 0.026733156 | 0.026310923–0.027053919 | True |
| Full gradients | 300 | 3 | 0.02729366 | 0.02667176–0.027780517 | True |
| Refined-label imitation | 0 | 3 | 0.026055197 | 0.026055196–0.026055198 | True |
| Refined-label imitation | 250 | 3 | 0.026849716 | 0.026174758–0.027794852 | True |
| Refined-label imitation | 500 | 3 | 0.027036773 | 0.026715597–0.027233196 | True |
| Refined-label imitation | 1000 | 3 | 0.028107126 | 0.027625855–0.028544537 | True |
| Action SPSA | 0 | 3 | 0.026055196 | 0.026055196–0.026055196 | True |
| Action SPSA | 100 | 3 | 0.026438175 | 0.025956771–0.026913186 | True |
| Action SPSA | 300 | 3 | 0.026393401 | 0.026220186–0.02663088 | True |

![All checkpoints](pilot_objectives.png)

![Measured final-checkpoint training costs](pilot_costs.png)

## Per-instance warm starts

| Horizon | Tasks | Linear mean | Warm AD mean | Mean reduction | AD optimization seconds/task | All admitted |
|---|---:|---:|---:|---:|---:|---|

Optimization timings exclude construction and fine-grid validation of the initial linear controller.

![Warm starts](warm_start.png)

## Complete cost ledger

| Cell | Common data s | Policy fit s | Extra labels s | Validation s | Expert labels selected/attempted | Status |
|---|---:|---:|---:|---:|---:|---|
| demonstration_imitation-0 | 2020.8 | 23.4 | 0.0 | 0.0 | 0/0 | complete |
| demonstration_imitation-1 | 2020.8 | 24.6 | 0.0 | 0.0 | 0/0 | complete |
| demonstration_imitation-2 | 2020.8 | 22.9 | 0.0 | 0.0 | 0/0 | complete |
| full-0 | 2020.8 | 1417.7 | 0.0 | 0.0 | 0/0 | complete |
| full-1 | 2020.8 | 1381.6 | 0.0 | 0.0 | 0/0 | complete |
| full-2 | 2020.8 | 1372.3 | 0.0 | 0.0 | 0/0 | complete |
| improved_imitation-0 | 2020.8 | 24.5 | 7501.4 | 0.0 | 0/0 | complete |
| improved_imitation-1 | 2020.8 | 24.0 | 7501.4 | 0.0 | 0/0 | complete |
| improved_imitation-2 | 2020.8 | 22.8 | 7501.4 | 0.0 | 0/0 | complete |
| spsa-0 | 2020.8 | 1150.8 | 0.0 | 0.0 | 0/0 | complete |
| spsa-1 | 2020.8 | 1143.9 | 0.0 | 0.0 | 0/0 | complete |
| spsa-2 | 2020.8 | 1178.4 | 0.0 | 0.0 | 0/0 | complete |

All cell statuses: {"demonstration_imitation-0": "complete", "demonstration_imitation-1": "complete", "demonstration_imitation-2": "complete", "full-0": "complete", "full-1": "complete", "full-2": "complete", "improved_imitation-0": "complete", "improved_imitation-1": "complete", "improved_imitation-2": "complete", "prepare-0": "complete", "prepare-1": "complete", "prepare-2": "complete", "prepare-3": "complete", "prepare-4": "complete", "prepare-5": "complete", "prepare-6": "complete", "prepare-7": "complete", "spsa-0": "complete", "spsa-1": "complete", "spsa-2": "complete"}

## Fixed-example full fields

![Long-horizon warm start, task 0](warm-long-0-task-0.png)

![Full-gradient policy seed 0, validation task 2000](full-0-task-2000.png)
