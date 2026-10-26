"""Same mesh shape, different coordinates: does the JAX-FEM setup cache see it?

Usage: python jaxfem_cache_demo.py <image>
Serves <image> once and calls apply for Lx=2 then Lx=4 (warm), then serves a
fresh container per geometry (cold) for the true value.
"""

import sys

import numpy as np
from tesseract_core import Tesseract

from mosaic.benchmarks.problems.structural_mesh.config import _SOLVERS
from mosaic.benchmarks.problems.structural_mesh.physics import make_inputs

image = sys.argv[1]
spec = next(s for s in _SOLVERS.values() if s.dir == "jax-fem")
nx = 8
ic = np.full(nx * 2 * (nx // 2), 0.5, dtype=np.float32)
inputs = {Lx: make_inputs(spec, ic, nx=nx, Lx=Lx) for Lx in (2.0, 4.0)}
assert inputs[2.0]["rho"].shape == inputs[4.0]["rho"].shape


def compliance(t, inp):
    return float(np.asarray(t.apply(inp)["compliance"]))


with Tesseract.from_image(image) as t:
    warm = {Lx: compliance(t, inp) for Lx, inp in inputs.items()}
cold = {}
for Lx, inp in inputs.items():
    with Tesseract.from_image(image) as t:
        cold[Lx] = compliance(t, inp)
for Lx in inputs:
    print(
        f"{image} Lx={Lx}: same container {warm[Lx]:.6g}, fresh container {cold[Lx]:.6g}"
    )
