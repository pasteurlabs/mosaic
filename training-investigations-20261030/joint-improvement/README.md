# Joint velocity and temperature correction: fresh-test follow-up

Joint correction now improves forecast error over a separately tuned velocity-only corrector in this JAX-CFD experiment. It still introduces physical defects, so it remains a research candidate.

## Controlled setup

- Same closed-box buoyancy flow, native JAX-CFD source1c714dd, coarse64²/dt0.005/eight native steps per correction interval. Own-solver reference128²/dt0.0025/sixteen steps per interval;100 intervals toT4.
- Width8, three3×3convolutions;881velocity-only or954joint parameters. Adam, clip1, H8 recurrent windows,256updates, same four training trajectories and balanced shuffled-window stream.
- Search54seed0candidates:48joint (raw/centered inputs, unconstrained/zero-mean thermal residual, head scales0.1/0.214/0.5/1, LR0.0003/0.001/0.003) and6velocity-only controls (raw/centered×sameLRgrid). Per-model training budgets match; joint has more hyperparameter-search compute.
- Full-rollout mean loss on validationICs1/5/6 chooses checkpoint0/64/128/192/256 and recipe. Three roles were specified before the search: bestvelocity, bestjointoverall, bestheat-preservingjoint.
- Freeze those recipes, then repeat each with networkseeds1/2:60training runs total, all complete. No oldIC100 or fresh learned-model test scores were accessed during training or selection.
- The nine chosen models were locked with recipe/parameter/source hashes before evaluating freshICs101–105. This is three model initializations crossed with five same-family ICs, not15independent model replicates.

All training differentiates through the native solver and corrector over eight recurrent intervals. The loss equally weights velocity and temperature after fixed first-training-trajectory RMS normalization. There are no supervised or stopped-gradient arms.

## Frozen recipes

| Model | Input centering | Thermal-head scale | Heat-mean constraint | Adam LR |
|---|---|---:|---|---:|
| Velocity only | No | — | Temperature unchanged |0.0003|
| Joint | No |0.1|None|0.001|
| Joint, heat preserving | No |0.5|Zero-mean thermal residual|0.0003|

The heat constraint subtracts the spatial mean of the predicted thermal residual before adding the physical identity path. It preserves integrated heat to roundoff but does not enforce a maximum principle. Centering was tested but not selected by the three-trajectory validation average.

## Fresh results

| Model | Mean combined error | Mean paired improvement vs velocity-only | Finer-time-reference improvement |
|---|---:|---:|---:|
| Velocity only |0.000674718|—|—|
| Joint |0.000587467|13.36%|11.57%|
| Joint, heat preserving |0.000625045|7.30%|6.90%|

Combined error includes both velocity and temperature. Improvements average three paired network-seed ratios, each formed after averaging the same five ICs. They are not the ratio of the grand mean losses in the second column.

The selected joint model improves five-IC mean error in all three initializations (ratios0.8003/0.9170/0.8820), winning13of15seed–IC case comparisons. Its mean paired velocity error is8.75%lower and temperature error16.57%lower; both improve in every initialization.

The heat-preserving model has ratios0.8689/1.0509/0.8611: two initializations improve, one worsens. Mean paired velocity error is2.70%lower and temperature error11.26%lower; it wins12of15case comparisons.

![Fresh paired field errors](fresh/report/fresh_relative_errors.png)
![Fresh full-rollout errors](fresh/report/fresh_error_rollouts.png)

## Physical properties

The unconstrained joint model introduces final heat drift of roughly−1.84%to+1.97% and temperature undershoot up to0.00492. The heat-preserving version retains total heat within about1e−9absolute mean-temperature units, comparable to velocity-only, but still produces undershoot up to0.00716. Velocity-only has no undershoot in these cases; none of these selected models overshoots the initial maximum. All sampled corrected rollouts remain finite, with maximum velocityCFL≤0.235 and divergence≈4.6e−7.

Thus field-error improvement and physical admissibility must be reported separately. No benchmark-default promotion follows from this experiment.

![Heat conservation and temperature bounds](fresh/report/fresh_heat_and_bounds.png)
![Predeclared IC101, networkseed0 example](fresh/report/fresh_fixed_example.png)

## Reference sensitivity

The exact same45frozen predictions were rescored against the preexisting N128reference with a further halved timestep0.00125. No learned or native solver was called and no model choice changed. Joint improvement remains11.57%, with all three initialization means better; one case winner flips. Heat-preserving improvement remains6.90%, with the same two improving initialization means. This checks temporal sensitivity of the reference, not convergence to a continuum solution or spatial-reference convergence.

![Temporal-reference sensitivity](fresh/report/reference_sensitivity.png)

## Validation, compute and controls

The selected nine models used roughly26–33seconds of nativeRTX fit time, including the first fully compiled training update and validation; shared reference/native/gradient-gate preparation and environment setup are excluded. These are not actual container/T4 CI timings. The full54-candidate research search is not proposed as a CI workload. Only256updates are used per trained model.

![Seed0 validation search](training/screen-report/joint_grid.png)
![Validation across three initializations](training/confirmation-report/validation_confirmations.png)
![Separate validation components](training/confirmation-report/component_validation.png)
![Per-model fit timing](training/confirmation-report/fit_timing.png)

Source-equivalent unconstrained forward maps and single-step parameter pullbacks match bitwise. The canonical unscaled velocity-only H8 loss and parameter gradient also match bitwise. An initial overstrict bitwise check on the reparameterized H8 gradient failed: distributing the head factor across interval contributions changes association at float32precision (relative6.82e−8, maxabsolute1.86e−9). The original failure and measured controls are retained; the separate reparameterization control uses1e−6relative agreement after this diagnostic.

All60training runs pass the same research-only convergence check at initial and exact production synthetic-nonzero parameters. It requires a common adjacent epsilon pair from0.003/0.001/0.0003 to satisfy the original5%AD/FD tolerance and5%slope agreement at both points. Every original fixed-epsilon result is retained; the public benchmark gate is unchanged. This is trueAD with an architectural constraint and parameter reparameterization, not a native-gradient accuracy repair or a full half-inverse-gradient implementation.

## Provenance and scope

[Training protocol](training/PROTOCOL.json), [all54validation candidates](training/screen-report/summary.json), [six confirmation runs](training/confirmation-summary.json), [frozen model choices](frozen-selection.json), [fresh scalar outcomes](fresh/evaluation/outcome.json), [independent reduction audit](training/independent-fresh-scalar-audit.json), [finer-time sensitivity](fresh/report/reference_sensitivity.md), and [publication hashes](publication-manifest.json).

Full model/field/reference arrays remain at the recorded cluster paths and hashes. Experiment code and archives were not added to the benchmark source branch. Source/defaults and CI labels remain unchanged. This is JAX-CFD only, one small CNN family and five new same-family ICs; it does not establish a universal optimizer or architecture winner across solvers or flow regimes.
