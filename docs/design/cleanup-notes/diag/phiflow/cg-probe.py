# Where PhiFlow's 2-D periodic step spends its time. Run in the tesseract image with
# api.py mounted at /work/api.py; JAX_PLATFORMS=cpu for the CPU arm.
#   A: per-step time of periodic_fwd (the tesseract's own jit), cost-experiment settings.
#   B: CG iterations of one projection, eager, via SolveTape.
#   C: time of one jitted projection vs CG iteration cap -> cost per CG iteration.
#   D: CG iterations with phiml's 'cluster' preconditioner and rank_deficiency settings, eager.
#   H: compiled HLO of one projection, written to /out.
# Usage: cg-probe.py <parts> [N,N,...]
import importlib.util
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from phi.jax.flow import (
    Box,
    CenteredGrid,
    Solve,
    StaggeredGrid,
    advect,
    diffuse,
    extrapolation,
    fluid,
    math,
)

spec = importlib.util.spec_from_file_location("api", "/work/api.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
L = 2 * np.pi
NU, DT = 0.01, 0.01
print("device", jax.devices()[0], "tol", api._CG_TOL, flush=True)


def ic(N, seed=0):
    x = (np.arange(N) + 0.5) * L / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rng = np.random.default_rng(seed)
    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)
    for k in range(2, 6):
        a, b = 0.1 * rng.standard_normal(2) / k
        u += a * np.sin(k * Y + b)
        v += b * np.sin(k * X + a)
    return np.stack([u, v], -1)[:, :, None, :].astype(np.float32)


def timed(f, *a, n=3):
    jax.block_until_ready(f(*a))
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*a))
        ts.append(time.perf_counter() - t0)
    return min(ts)


parts = sys.argv[1] if len(sys.argv) > 1 else "ABC"
Ns = (
    [int(n) for n in sys.argv[2].split(",")]
    if len(sys.argv) > 2
    else [64, 128, 192, 256]
)

if "A" in parts:
    for N in Ns:
        v0 = jnp.asarray(ic(N))
        for steps in (10, 100):
            f = jax.jit(lambda v, s=steps: api.periodic_fwd(v, NU, DT, s, L))
            t = timed(f, v0)
            print(
                f"A N={N} steps={steps} total={t:.4f}s per_step={t / steps * 1e3:.2f}ms",
                flush=True,
            )


def tentative(N):
    """Velocity before projection, one forward-Euler central step from the IC (main's step)."""
    bounds = Box["x,y", 0:L, 0:L]
    c = CenteredGrid(
        math.stack(
            {
                "x": math.tensor(ic(N)[:, :, 0, 0], math.spatial("x,y")),
                "y": math.tensor(ic(N)[:, :, 0, 1], math.spatial("x,y")),
            },
            math.channel("vector"),
        ),
        extrapolation.PERIODIC,
        bounds=bounds,
        x=N,
        y=N,
    )
    vel = StaggeredGrid(c, extrapolation.PERIODIC, bounds=bounds, x=N, y=N)
    vel, _ = fluid.make_incompressible(
        vel, solve=Solve("CG", 1e-7, 1e-7, max_iterations=5000)
    )
    return vel + DT * (
        advect.finite_difference(vel, vel, order=2)
        + diffuse.finite_difference(vel, NU, order=2)
    )


def to_faces(vel):
    return jnp.stack([vel.vector[i].values.native("x,y") for i in range(2)], 0)


def from_faces(f, N):
    t = math.stack(
        {n: math.tensor(f[i], math.spatial("x,y")) for i, n in enumerate("xy")},
        math.dual("vector"),
    )
    return StaggeredGrid(
        t, extrapolation.PERIODIC, bounds=Box["x,y", 0:L, 0:L], x=N, y=N
    )


def projector(N, solve):
    return jax.jit(
        lambda f: to_faces(fluid.make_incompressible(from_faces(f, N), solve=solve)[0])
    )


def info_str(info):
    it = int(math.max(info.iterations))
    conv = bool(info.converged.all)
    res = float(math.max(abs(info.residual.values)))
    return f"iterations={it} converged={conv} residual_max={res:.3g}"


if "B" in parts:
    for N in Ns:
        vel = tentative(N)
        for cap in (1000, 5000):
            solve = Solve(
                "CG", 1e-7, 1e-7, max_iterations=cap, suppress=[math.NotConverged]
            )
            with math.SolveTape() as tape:
                fluid.make_incompressible(vel, solve=solve)
            print(f"B N={N} tol=1e-7 cap={cap} {info_str(tape[0])}", flush=True)
        for tol in (1e-6, 1e-5):
            solve = Solve(
                "CG", tol, tol, max_iterations=5000, suppress=[math.NotConverged]
            )
            with math.SolveTape() as tape:
                fluid.make_incompressible(vel, solve=solve)
            print(f"B N={N} tol={tol} {info_str(tape[0])}", flush=True)

if "C" in parts:
    for N in Ns:
        f0 = jax.jit(lambda n=N: to_faces(tentative(n)))()
        for cap in (10, 100, 1000):
            t = timed(
                projector(
                    N,
                    Solve(
                        "CG", 0.0, 0.0, max_iterations=cap, suppress=[math.NotConverged]
                    ),
                ),
                f0,
            )
            print(f"C N={N} cap={cap} time={t * 1e3:.2f}ms", flush=True)
        for tol in (1e-7, 1e-6, 1e-5):
            t = timed(projector(N, Solve("CG", tol, tol)), f0)
            print(f"C N={N} tol={tol} time={t * 1e3:.2f}ms", flush=True)

if "D" in parts:
    for N in Ns:
        vel = tentative(N)
        for rd in (None, 0, 1):
            for pre in (None, "cluster"):
                kw = {} if rd is None else {"rank_deficiency": rd}
                solve = Solve(
                    "CG",
                    1e-7,
                    1e-7,
                    max_iterations=3000,
                    preconditioner=pre,
                    suppress=[math.NotConverged],
                    **kw,
                )
                try:
                    with math.SolveTape() as tape:
                        fluid.make_incompressible(vel, solve=solve)
                    print(
                        f"D N={N} rank_deficiency={rd} pre={pre} {info_str(tape[0])}",
                        flush=True,
                    )
                except (
                    Exception
                ) as e:  # 'cluster' fails on the rank-deficiency row; record the error
                    print(
                        f"D N={N} rank_deficiency={rd} pre={pre} failed: {type(e).__name__}: {str(e)[:150]}",
                        flush=True,
                    )

if "H" in parts:
    import re

    for N in Ns:
        f0 = jax.jit(lambda n=N: to_faces(tentative(n)))()
        txt = projector(N, Solve("CG", 1e-7, 1e-7)).lower(f0).compile().as_text()
        with open(f"/out/proj_N{N}_{jax.devices()[0].platform}.hlo.txt", "w") as fh:
            fh.write(txt)
        bodies = re.findall(
            r"\n(%?[\w.\-]*body[\w.\-]*) .*?\{\n(.*?)\n\}", txt, re.DOTALL
        )
        for name, body in bodies:
            kinds = re.findall(
                r"= \S+ (fusion|custom-call|scatter|gather|reduce|copy|dot|sort|while|conditional)\(",
                body,
            )
            print(
                f"H N={N} {name[:60]} "
                + " ".join(f"{k}={kinds.count(k)}" for k in sorted(set(kinds))),
                flush=True,
            )
