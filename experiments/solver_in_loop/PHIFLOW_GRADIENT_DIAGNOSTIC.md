# PhiFlow gradient precision diagnostic

The SSPRK3 candidate's earlier energy and random-cotangent finite-difference
checks remain recorded as failures. The investigation below identifies poor
float32 finite-difference resolution, rather than a demonstrated VJP defect.
It does not override those results or change the neural-training gradient gate.

## Preserved evidence

All numerical work ran on B200 Slurm allocations. Results live under
`/data/personal/andrinr/runner/results/mosaic/`:

- `pr116-phiflow-ssprk3-diagnostic-20261004`: original candidate burns,
  temporal/closure checks, failed primary energy FD, and failed supplementary
  cotangent FD. The 64² and 192² burns both complete physical time 75 with
  unchanged forcing and viscosity. Temporal 3/6 relative error is 8.9281e-5;
  native closure error is 0. These were an explicitly recorded adapter overlay.
- `pr116-phi-gradient-precision-20261005`: jobs 2895274/2895275, native
  float32/float64 sweeps across one unforced step and 1/8/64 forced steps.
- `pr116-phi-gradient-parity-20261005`: matched-vector native jobs 2895293/2895294,
  RPC job 2895298, image build 2895304, and numerical comparison 2895309.
  `parity-analysis.json` contains every forward/gradient array comparison;
  source hashes and complete step sweeps are retained. RPC raw arrays are in
  the earlier campaign's `results/gradient-parity-rpc` directory.

## What was tested

The new diagnostic uses the same saved post-burn 64² initial state, viscosity .001,
dt .01, forcing amplitude 1 and wavenumber 6. It checks centered energy and fixed
random linear objectives in fixed smooth and random directions. Centering
subtracts a constant baseline, preserving the mathematical derivative while
reducing cancellation during the scalar reduction.

The entire perturbation sweep is reported: RMS amplitudes .01, .003, .001, .0003,
.0001, .00003, .00001. No best epsilon is chosen. RMS normalization is explicit;
the older unit-L2 direction with epsilon .001 on 8192 components perturbs each
component by only about 1.1e-5 RMS. The native 64-step probe has the temporal
extent of 16 correction intervals, but does not contain a learned corrector.

The first precision experiment used JAX random vectors in each native dtype;
its random slopes therefore cannot be paired between precisions. The second
experiment explicitly generates float32 vectors before casting, saves full
forward and gradient arrays, and compares identical vector inputs.

## Findings

For the full 64-step chain in the first sweep, float32 errors across both
objectives and both directions are at most .001807/.001953/.002600 at RMS
perturbations .01/.003/.001. Errors increase at very small perturbations (up to
.4302 at 1e-5). Float64 errors decrease toward roundoff as perturbations shrink,
with the expected central-difference convergence pattern.

The matched-input comparison shows:

- Every saved RPC forward and gradient array is bitwise identical to its native
  float32 counterpart, including both 64-step objectives.
- The largest float32-versus-float64 forward relative L2 error is 7.0377e-6.
- The largest float32-versus-float64 gradient relative L2 error is 2.55134e-5.
- All saved arrays are finite.

These results support finite-difference roundoff as the cause of the earlier
small-perturbation failures. They do not justify changing the frozen model-
parameter FD epsilon .001 or its .05 error threshold. Fresh reference admission
and the actual 16-interval neural-training gradient check remain prerequisites
for the fixed transfer campaign.

## Standalone candidate image

`/data/personal/andrinr/runner/artifacts/mosaic/phiflow-ssprk3-2895304.sqsh`

SHA256: `01a2b244fef26be086a27d938e12ce564ac8b000e7d759396907beed6181d4ce`

Adapter SHA256:
`3ccc5eeda8a807c328f1b8948815c76c6eba1bde88345af4b55b43fbed8a9127`.
The build verified the original and candidate adapter hashes before replacing
the adapter in an isolated root-remapped container and exporting a new image.
An independent allocation then verified the embedded candidate hash. The
original image remains unchanged. This is an explicit SSPRK3 integrator
variant, not an unchanged native-solver transfer or a third-order claim for the
externally Strang-split forced system.
