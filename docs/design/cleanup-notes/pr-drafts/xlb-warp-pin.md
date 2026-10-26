## Summary

Replaces the XLB build step that deleted the `ibm_stepper` import from the installed XLB source with a pin, `warp-lang==1.12.1`. xlb 0.3.1 declares `warp-lang>=1.10.0` and imports `warp.utils.ScopedTimer` unconditionally, which Warp 1.13 removed; the image was resolving warp-lang 1.18.0. With 1.12.1, `import xlb` works unmodified (Warp prints a deprecation warning). XLB's `main` already imports `from warp import ScopedTimer`, but there is no release after 0.3.1.

## Type of change

- [ ] New solver backend
- [ ] Solver tuning / improvement
- [ ] New benchmark domain
- [ ] Harness / infrastructure change
- [ ] Documentation
- [x] Bug fix

## Benchmark label

`benchmark:solver`, so CI confirms XLB results are unchanged.

## Notes

The tesseract runs XLB on its JAX backend, so Warp is only imported. Verified on an A100: `mosaic run --suites forward,gradient -s XLB` on ns-grid and ns-3d-grid with images from `main` (warp-lang 1.18.0, patched) and from this branch gives bit-identical metrics in all 27 experiments.
