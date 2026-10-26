"""PhiFlow: pressure solve warm-started from a jax.lax.scan carry, differentiated with jax.grad.

Usage: python repro.py x0|none
"""

import sys

import jax
import jax.numpy as jnp
from phi.jax.flow import *  # noqa: F403

use_x0 = sys.argv[1] == "x0"
domain = dict(x=16, y=16, extrapolation=extrapolation.PERIODIC, bounds=Box(x=1, y=1))
v0 = StaggeredGrid(Noise(), **domain)


def loss(scale):
    def step(p_vals, _):
        p_prev = CenteredGrid(math.tensor(p_vals, spatial("x,y")), **domain)
        solve = Solve("CG", 1e-5, 1e-5, x0=p_prev if use_x0 else None)
        _, p = fluid.make_incompressible(v0 * scale, (), solve)
        return p.values.native("x,y"), None

    p_final, _ = jax.lax.scan(step, jnp.zeros((16, 16), jnp.float32), None, length=3)
    return jnp.sum(p_final**2)


print("loss", float(loss(1.0)))
print("grad", float(jax.grad(loss)(1.0)))
