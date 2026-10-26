# Policy 1: XLB collision operator

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

## Options

1. **BGK everywhere.** One operator per entry, and the default. Any non-finite runs at ω near 2 become explained anomalies. I have not measured how many there would be (#38 has weak evidence that BGK stays finite).
2. **Stock KBC for documented high-Re cases (L0).** This follows the README, but gradient cells at N ≥ 32 become anomalies, and the entry stays a hybrid unless the choice is fixed per experiment.
3. Either option, plus a separate "XLB (smoothed KBC)" entry. This keeps the #232 gradient results visible under their own name.

## Recommendation (uncertain)

Option 1 plus option 3. Run BGK on the periodic configurations first. Choose option 2 only if BGK fails there.

## Open questions

- Is a separate entry worth its maintenance cost, or should the smoothing be dropped?
- Should the zero-IC device dependence of stock KBC (#234) be recorded as an anomaly too?
- Main's KBC drops ρ from Δs, so a version bump changes the subclass.
