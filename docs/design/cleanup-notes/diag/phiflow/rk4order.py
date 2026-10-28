# Temporal order of phi.physics.fluid.incompressible_rk4 on a diffusing shear wave u = (sin y, 0):
# divergence-free and advection-free, so the semi-discrete solution decays exactly as exp(-nu*lam*t),
# lam = (2 - 2 cos dy) / dy^2 for the 2nd-order Laplacian.
import numpy as np
from phi.jax.flow import (
    Box,
    CenteredGrid,
    Solve,
    StaggeredGrid,
    advect,
    channel,
    diffuse,
    extrapolation,
    fluid,
    math,
)

math.set_global_precision(64)
N, L, nu, T = 8, 2 * np.pi, 1.0, 1.0
dy = L / N
lam = (2 - 2 * np.cos(dy)) / dy**2
domain = {
    "x": N,
    "y": N,
    "extrapolation": extrapolation.PERIODIC,
    "bounds": Box(x=L, y=L),
}


def pde(v):
    return advect.finite_difference(v, v, order=2) + diffuse.finite_difference(
        v, nu, order=2
    )


def run(steps, stepper):
    dt = T / steps
    v = StaggeredGrid(
        lambda x: math.stack(
            [math.sin(x.vector["y"]), 0 * x.vector["y"]], channel(vector="x,y")
        ),
        **domain,
    )
    p = CenteredGrid(0, **domain)
    for _ in range(steps):
        v, p = stepper(pde, v, p, dt)
    u = v.vector["x"].values.numpy("x,y")
    y = (np.arange(N) + 0.5) * dy
    exact = np.sin(y)[None, :] * np.exp(-nu * lam * T)
    return np.abs(u - exact).max()


def classic_rk4(pde, velocity, pressure, dt, **kw):
    f = lambda v: pde(v)
    k1 = f(velocity)
    k2 = f(velocity + dt / 2 * k1)
    k3 = f(velocity + dt / 2 * k2)
    k4 = f(velocity + dt * k3)
    return velocity + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), pressure


for name, st in [
    (
        "phiflow incompressible_rk4",
        lambda pde, v, p, dt: fluid.incompressible_rk4(
            pde, v, p, dt, pressure_order=2, pressure_solve=Solve("CG", 1e-12, 1e-12)
        ),
    ),
    ("classical RK4 (reference)", classic_rk4),
]:
    errs = [(s, run(s, st)) for s in (8, 16, 32, 64, 128)]
    print(
        name,
        " ".join(f"steps={s} err={e:.3e}" for s, e in errs),
        " orders:",
        " ".join(
            f"{np.log2(errs[i][1] / errs[i + 1][1]):.2f}" for i in range(len(errs) - 1)
        ),
        flush=True,
    )
