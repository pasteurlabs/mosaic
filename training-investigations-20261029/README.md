# JAX-CFD corrector investigations

These supporting experiments are separate from the bounded benchmark's default recipe and acceptance results. Code source: `0246f6a`; exact hashes, devices, jobs and report hashes are recorded in `handoff.json`.

## Protocol

The screen tested widths 4/8/16/32 and Adam/AdamW/Muon, with three learning rates per optimizer. Each model used 256 recurrent-window evaluations, the same four training cases, eight-interval windows, and independent full-trajectory validation. Batch-4 fits used 64 optimizer steps: equal window work, different optimizer budget. Checkpoints were evaluated every 16 window evaluations. Stage 1 had 36 eligible fits; stage 2 contained 24 recipes (20 new and 4 reused); confirmation attempted 10 new fits and trained 8, with 2 rejected by the original gradient check before optimization. Seed 0 selected recipes; seeds 1/2 checked model initialization robustness on the same validation/test trajectories. No test scores selected candidates.

All formal screen/followup/confirmation fits used RTX 5090. The separate initial three-arm pilot and early Muon feasibility used B200. Native JAX-CFD requirements were an explicitly recorded overlay on the production environment. Direct native calls bypassed per-call Pydantic checks; all saved selected rollouts met the advective CFL limit, which does not establish every transient training window was admissible.

## Findings and limits

- Width-8 AdamW at LR .001 improved held-out error over the original Adam .003 baseline in all three observed model seeds. Seed 0 Adam .001 and AdamW .001 were almost identical; this does not establish a benefit from weight decay. Matched Adam .001 seeds 1/2 and other-backend transfer checks are being run before changing the default recipe.
- Wider Muon results were mixed across seeds. No universal width/optimizer winner is established.
- Joint velocity/temperature correction reduced error in some runs but introduced heat-integral drift and undershoot. Two seed-1 configurations failed the fixed-epsilon neural gradient check before training. Their failed cells remain explicit; incomplete families have no three-seed aggregate.
- The width-32 failed finite-difference check was reproduced. Error decreased approximately quadratically with epsilon, and higher-precision loss accumulation barely changed it: finite-step truncation is supported, rather than a demonstrated AD defect. This diagnostic does not retroactively override admission.

## Gradient paths

Plots compare solver, corrector and chain-rule-composed state adjoints under matched normalized output probes. Local temperature probes satisfy the velocity-only corrector's identity behavior; recurrent eight-interval probes reveal indirect temperature effects. Higher-frequency velocity sensitivities change more than low-frequency or temperature probes in these examples. Norm/cosine/spectral differences are sensitivity measurements, not evidence of better gradient accuracy or full-Jacobian conditioning. Raw finite-difference checks are retained in the reports; all plots are from one held-out physical trajectory.

`final-report-v3` retains successful and rejected confirmation cells. `gradient-paths-report-v2` contains local, recurrent and scale-resolved operator plots. Full parameters and field arrays remain in external campaign artifacts; they are omitted here to keep this publication compact.
