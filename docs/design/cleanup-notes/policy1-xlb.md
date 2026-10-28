# Policy 1: XLB collision operator

**Status:** decided 2026-10-27: BGK everywhere, smoothed KBC removed, one XLB entry (#282).

## The case

XLB's stepper defaults to BGK (`collision_type="BGK"`, [xlb/operator/stepper/nse_stepper.py@v0.3.1](https://github.com/Autodesk/XLB/blob/v0.3.1/xlb/operator/stepper/nse_stepper.py)). Mosaic's periodic runs switch to KBC above ω = 1.8 (`kbc-collision-low-viscosity`), and the KBC they switch to is a Mosaic subclass with a smoothed stabiliser (`smooth-kbc-stabiliser`, #194, #232, #234). The channel stays on BGK. Because ω is computed from dt/k (`tesseract_api.py:612-617`), Ma-targeted sub-stepping pushes runs towards ω = 2 and onto KBC.

## What XLB says

- README and docs list "KBC collision model (unconditionally stable for flows with high Reynolds number)" ([README.md@v0.3.1](https://github.com/Autodesk/XLB/blob/v0.3.1/README.md), unchanged on main, also docs/index.md). No ω or Re threshold appears anywhere I searched (README, docs/, CHANGELOG, docstrings, issues and PRs; discussions are disabled).
- Examples use KBC at high Re and ω ≈ 1.96-2.0: `examples/cfd/windtunnel_3d.py` (Re 5e4), `examples/cfd/turbulent_channel_3d.py` (Re_τ 180), `examples/ibm/sphere_ibm.py` (Re 300). They use BGK at low Re: `examples/cfd/lid_driven_cavity_2d.py` (Re 200, ω ≈ 1.14), `examples/cfd/flow_past_sphere_3d.py` (ω 1.6, Re ≈ 10 by my arithmetic).
- No example is periodic decaying turbulence. The XLB paper ([arXiv:2311.16080](https://arxiv.org/abs/2311.16080), §4.1) runs a 2D TGV with both operators and reports KBC "more robust" up to N = 1024, with BGK degrading beyond N = 256. It attributes the BGK degradation to the time step or initialization. Its Re = 100 cylinder (§4.3) uses BGK.
- On differentiation: both gradient examples use BGK (`examples/out_of_core/autodiff_lbm.py`, default τ = 0.53, so ω ≈ 1.89; `examples/cfd/differentiable_lbm.py`, main only). I found nothing on differentiating through KBC. Our question on this ([Autodesk/XLB#165](https://github.com/Autodesk/XLB/issues/165)) has no replies.

## What policy 1 implies

- **Channel (Re 20).** BGK is the default, and nothing in XLB points to KBC at this Re. No intervention is needed. The docstring's reason (KBC makes drag non-smooth) does hold on the current channel, although #194 measured stock-KBC drag gradients within 1.3e-3 of FD on the older setup. Measured 2026-10-26 on an A100, `drag_opt` setup (10 cells per diameter, t_end = 4, float64), Cd along two smooth inlet-profile directions over ±5% of U_m:
  - BGK: Cd smooth (quartic-fit residual 4e-8 over a 0.03 span); VJP vs central FD within 4e-6 to 1.5e-5 at the best ε.
  - stock KBC: Cd jumps by 0.1 to 0.3 between neighboring profiles (residual 0.18 over a 0.31 span); VJP is NaN.
  - smoothed KBC: residual 0.06 over 0.17; VJP off by 10³ to 10⁶.

  The docstring could cite this measurement in addition to the default. Script and outputs: `diag/xlb/` (next to this note).

- **KBC switch.** The README gives a documented reason to use KBC for high-Re flows, so a switch can be L0 with that evidence. Whether small-grid TGV at ν = 1e-3 to 5e-2 counts as "high Re" is a judgment call. The 1.8 threshold is ours, and XLB's own autodiff example runs BGK above it.
- **Smoothed stabiliser.** XLB ships nothing like it, so policy 1 cannot cover it. Rule 4 makes it a separate entry. Rule 3 says stock KBC's gradient blow-up (7.4e9 × FD at N = 32) stays in the XLB results.

## Measured on the full suites (A100; forward, gradient, optimization; float64 as in the tesseract)

Three images: `main` (BGK, smoothed KBC above ω = 1.8), BGK everywhere, and XLB's stock KBC above 1.8. All periodic cells are finite with every operator.

|                                                                      | `main`                   | BGK                    | stock KBC  |
| -------------------------------------------------------------------- | ------------------------ | ---------------------- | ---------- |
| ns-grid `agreement/tgv` error                                        | 1.1e-3                   | 6.2e-4                 | 1.7e-3     |
| ns-grid `baseline` error                                             | 1.5e-3                   | 9.0e-4                 | 1.9e-3     |
| ns-grid `analytic_check` rel. error                                  | 3.1e-2                   | 1.9e-4                 | anomaly    |
| ns-grid `fd_check` rel. error                                        | 9.2e-5                   | 1.1e-5                 | 6.7e-5     |
| ns-3d-grid `agreement` error                                         | 9.2e-3                   | 8.5e-3                 | 9.2e-3     |
| ns-3d-grid `fd_check` rel. error                                     | 4.8e-4                   | 4.1e-6                 | 5.1e-4     |
| ns-3d-grid `recovery` final ratio                                    | 8.6e-7                   | 5.5e-8                 | 8.6e-7     |
| ns-3d-grid `horizon_sweep_limits` (N = 20, ν = 1e-3) last finite VJP | 1280 steps (OOM at 2560) | 160 steps (NaN at 320) | 2560 steps |

The only cell where KBC helps is the horizon sweep, which maps where each solver stops rather than scoring it. There BGK's VJP turns non-finite at 320 steps, the documented BGK instability at ω → 2 in under-resolved flow, and it stays in the results as an artefact of the method. One peer effect: with BGK, XLB's lower ns-grid `fd_check` error moves the peer median from 2.1e-5 to 1.2e-5, and PICT's unchanged 8.8e-4 becomes an anomaly (73× the median, threshold 50×); its cause is not investigated.

## Decision

BGK, XLB's default, for every case; drop the switch and the smoothed subclass. No separate entries, so the smoothed stabiliser (a Mosaic-written operator, rule 4) has no place; stock KBC is the scheme XLB documents for high Re, but here it only extends the horizon sweep and fails the 2-D analytic gradient check, so the amended policy 1 gives no reason to pick it.
