# Fixed readout-head sensitivity scaling

This external JAX-CFD diagnostic estimates a **single fixed scalar reparameterization** of the joint velocity/temperature readout. It is not half-inverse-gradient (HIG) or scale-invariant-physics (SIP) training.

The training objective already normalizes physical velocity and temperature by training RMS scales. That normalizes units in the loss; it does not make the neural-network-plus-solver sensitivity to both readout heads equal.

For training windows (cases 0,1,2,3; starts 0,24,48,72; horizon 8), let R(p) be all normalized recurrent output residuals, weighted so that

    0.5 * ||R(p)||² = mean public normalized_loss over all 4 windows and 8 outputs.

We estimate sigma_h² = mean_k ||J_R(p0) d_h,k||² for six matched unit readout-weight directions in each head. The initial readout p0 is zero, so all candidate models start from exactly the native solver. The fixed factors are

    head_scales = [1, sigma_velocity_head / sigma_temperature_head].

The scale acts through a differentiable parameter map:

    effective final-layer weights and biases = raw trainable values * head_scales.

The reporting loss stays unchanged. For SGD this is a scalar block preconditioner in effective readout coordinates. For Adam, cancellation of a constant gradient scale does not generally remove the corresponding change in physical output step size. Neither observation establishes a training benefit.

The same directional-response estimate at the production synthetic nonzero point is **diagnostic only**. It cannot change the factor. No validation or test data enter calibration.

Controls retain:

- Exact normalized-residual/public-loss agreement.
- Full output-vector centered FD versus JVP for one matched direction per head at initial and synthetic nonzero points, using fixed epsilon .003/.001/.0003; at least two vector relative errors must be below 5%.
- Exact zero-readout identity and helper/explicit-parameter-map parity.
- Individual directional responses, timing, source/data hashes and unconstrained temperature-change metrics.

All calibration cost, including compilation and probes, must be charged if judging total wall-clock advantage. This experiment does not whiten all singular directions, condition shared hidden layers, or guarantee preservation of heat or temperature bounds. The zero-readout estimate may become stale during training.

Background: [Scale-Invariance and Inversion](https://physicsbaseddeeplearning.org/physgrad.html) and [Half-Inverse Gradients](https://physicsbaseddeeplearning.org/physgrad-hig.html). The latter uses singular-value decomposition and a truncated half inverse of the combined network/physics Jacobian; this scalar block experiment does not implement that algorithm.

## Completed calibration

| Point | Velocity-head response RMS | Temperature-head response RMS | Thermal scale |
|---|---:|---:|---:|
| initial | 0.32149705 | 1.50251387 | 0.21397276 |
| synthetic_nonzero | 0.32194109 | 1.25599877 | 0.25632277 |

The selected thermal factor is **0.2139727633353161**, from the initial point only. The thermal readout has approximately4.67times the normalized recurrent directional response of the velocity readout at that point. This is evidence for differing local head sensitivity, not a demonstrated explanation of AdamW training outcomes. The diagnostic nonzero point changes the ratio; a fixed estimate can become stale.

All12 full-output vector centered-FD comparisons passed (worst relativeerror0.204%); the two objective definitions agree to1.25e−7 relative. Zero-readout identity and helper/explicit-map parity are exact. Calibration took16.359seconds including first compilation, all probes and checks, excluding environment setup.

The synthetic point can remove mean heat and create negative temperature values. Scaling reduces the magnitude of that particular synthetic correction but does not impose heat conservation or a maximum principle. Training outcomes are a separate experiment.
