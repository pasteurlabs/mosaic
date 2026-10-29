# Cost of one CG iteration's pieces on the GPU, matching the shape of phiml's pressure solve:
# a BCOO matvec with a periodic 5-point Laplacian plus the rank-deficiency row and column
# (7 entries per row, as in the HLO), against the same operator as a jnp.roll stencil.
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

REPS = 1000


def laplacian_bcoo(N):
    i = np.arange(N * N).reshape(N, N)
    rows, cols, vals = [], [], []
    for di, dj, v in (
        (0, 0, -4.0),
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
    ):
        rows.append(i.ravel())
        cols.append(np.roll(np.roll(i, di, 0), dj, 1).ravel())
        vals.append(np.full(N * N, v))
    n = N * N
    rows += [np.arange(n), np.full(n, n)]
    cols += [np.full(n, n), np.arange(n)]
    vals += [np.ones(n), np.ones(n)]
    idx = np.stack([np.concatenate(rows), np.concatenate(cols)], 1).astype(np.int32)
    return sparse.BCOO(
        (jnp.asarray(np.concatenate(vals), jnp.float32), jnp.asarray(idx)),
        shape=(n + 1, n + 1),
    )


def stencil(x, N):
    g = x[:-1].reshape(N, N)
    lap = (
        jnp.roll(g, 1, 0)
        + jnp.roll(g, -1, 0)
        + jnp.roll(g, 1, 1)
        + jnp.roll(g, -1, 1)
        - 4 * g
        + x[-1]
    )
    return jnp.concatenate([lap.ravel(), jnp.sum(g).reshape(1)])


def per_call(body, x):
    f = jax.jit(
        lambda x: jax.lax.fori_loop(0, REPS, lambda _, y: body(y) * 1e-3 + x, x)
    )
    jax.block_until_ready(f(x))
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        jax.block_until_ready(f(x))
        ts.append(time.perf_counter() - t0)
    return min(ts) / REPS * 1e6


print("device", jax.devices()[0], flush=True)
for N in (
    [int(n) for n in sys.argv[1].split(",")]
    if len(sys.argv) > 1
    else [64, 128, 192, 256]
):
    A = laplacian_bcoo(N)
    x = jnp.asarray(np.random.default_rng(0).standard_normal(N * N + 1), jnp.float32)
    assert np.allclose(A @ x, stencil(x, N), atol=1e-4)
    t_spmv = per_call(lambda y, A=A: A @ y, x)
    t_sten = per_call(lambda y, N=N: stencil(y, N), x)
    t_dot = per_call(lambda y: y * jnp.vdot(y, y), x)
    print(
        f"S N={N} nnz={A.nse} bcoo_matvec={t_spmv:.1f}us stencil={t_sten:.1f}us dot+scale={t_dot:.1f}us",
        flush=True,
    )
