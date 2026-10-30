# Draft: tum-pbs/PhiML (or tum-pbs/PhiFlow)

**Status:** filed 2026-10-30 (see the handoff for the issue number).

**Title:** `UnexpectedTracerError` in the gradient of `solve_linear` when `Solve.x0` comes from a `jax.lax.scan` carry

## Body

Warm-starting a linear solve from the previous step's solution inside `jax.lax.scan` works in the forward pass but fails under `jax.grad`. Without `x0`, the same code differentiates fine.

```python
import jax
import jax.numpy as jnp
from phi.jax.flow import *

domain = dict(x=16, y=16, extrapolation=extrapolation.PERIODIC, bounds=Box(x=1, y=1))
v0 = StaggeredGrid(Noise(), **domain)

def loss(scale, use_x0):
    def step(p_vals, _):
        p_prev = CenteredGrid(math.tensor(p_vals, spatial("x,y")), **domain)
        solve = Solve("CG", 1e-5, 1e-5, x0=p_prev if use_x0 else None)
        _, p = fluid.make_incompressible(v0 * scale, (), solve)
        return p.values.native("x,y"), None
    p_final, _ = jax.lax.scan(step, jnp.zeros((16, 16), jnp.float32), None, length=3)
    return jnp.sum(p_final**2)

jax.grad(loss)(1.0, False)  # works
loss(1.0, True)             # works
jax.grad(loss)(1.0, True)   # UnexpectedTracerError
```

```
jax.errors.UnexpectedTracerError: Encountered an unexpected tracer. A function transformed by JAX had a side effect,
allowing for a reference to an intermediate value with type float32[16,16] wrapped in a DynamicJaxprTracer to escape
the scope of the transformation.
  File ".../phi/physics/fluid.py", line 156, in make_incompressible
  File ".../phiml/math/_optimize.py", line 694, in solve_linear
  File ".../phiml/math/_functional.py", line 972, in __call__
```

It looks like the custom gradient of `solve_linear` keeps a reference to `x0` from the traced scan body. Since `x0` only affects the iteration count, not the solution, it may be possible to `stop_gradient` it or exclude it from what the backward pass captures.

Versions: phiflow 3.4.0, phiml 1.16.1, jax 0.10.0, CPU (`JAX_PLATFORMS=cpu`). It also fails with the current `develop` branches of PhiML (01e44a5) and PhiFlow (724fac3).

Our workaround: solve for the correction `p - p_prev` from a zero initial guess.

## Notes for Dion (not part of the issue)

- Reproduced 2026-10-26 in the phiflow image (script: `upstream-phiml-repro.py` next to this file, run as `python repro.py x0|none`).
- Related but different: tum-pbs/PhiML#16 (int64, still open) is the other PhiFlow workaround.
- Repo choice: the traceback is in phiml, so PhiML is probably right; PhiFlow is where users meet it.
