## Summary

TopOpt.jl's VJP read the undocumented `Compliance.grad` field after a forward call. It now takes the gradient through Zygote, as TopOpt.jl documents (`Zygote.gradient(r -> comp(PseudoDensities(r)), rho)`). TopOpt.jl 0.11.0's `rrule` for `Compliance` returns `copy(comp.grad)` scaled by the cotangent, so the gradient is the same vector. `topopt_forward` now returns only the compliance; the gradient and displacement it also returned were unused apart from the VJP.

Stacked on #260 (the file moves there); retarget to `main` once #260 is merged.

## Type of change

- [ ] New solver backend
- [ ] Solver tuning / improvement
- [ ] New benchmark domain
- [ ] Harness / infrastructure change
- [ ] Documentation
- [ ] Bug fix

Cleanup, no result change intended.

## Benchmark label

`benchmark:solver`, so CI confirms TopOpt.jl results are unchanged.

## Notes

Verified locally (CPU):

- In the image, `comp.grad` and the Zygote gradient are bit-identical on cantilevers at nx = 8 and 16 with random densities.
- `gradient/fd_check`, `gradient/param_sweep` and `optimization/topopt` give identical metrics with images from #260 and from this branch (final compliance 0.0010033302241936326 after 2500 iterations in both).
- `cost/vjp_cost`: VJP_COST_PLACEHOLDER
