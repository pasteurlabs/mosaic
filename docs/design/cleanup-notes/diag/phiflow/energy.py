# Kinetic energy over long periodic rollouts: forward only, one api.py per run.
# Periodic incompressible NS can only lose energy, so any rise is numerical.
# Run with the tesseract image, api.py mounted at /work/api.py (see phiflow-quiet.sh).
# The RK4 pressure carry restarts from zero at each chunk.
import importlib.util
import time

import jax.numpy as jnp
import numpy as np

spec = importlib.util.spec_from_file_location("api", "/work/api.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
L = 2 * np.pi


def tgv2d(N):
    x = (np.arange(N) + 0.5) * L / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.stack([np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y)], -1)[:, :, None, :]


def tgv3d(N):
    x = (np.arange(N) + 0.5) * L / N
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return np.stack(
        [np.sin(X) * np.cos(Y) * np.cos(Z), -np.cos(X) * np.sin(Y) * np.cos(Z), 0 * X],
        -1,
    )


def ke(v):
    return float(0.5 * np.mean(np.sum(np.asarray(v, np.float64) ** 2, -1)))


cases = [
    # (label, ic, N, nu, dt, chunk, n_chunks)
    ("3d horizon N20 nu1e-3 dt0.05", tgv3d, 20, 1e-3, 0.05, 40, 16),
    ("3d N20 nu0 dt0.05", tgv3d, 20, 0.0, 0.05, 40, 16),
    ("2d N32 nu0 dt0.05", tgv2d, 32, 0.0, 0.05, 100, 20),
    ("2d N32 nu1e-3 dt0.05", tgv2d, 32, 1e-3, 0.05, 100, 20),
    ("2d N64 nu0 dt0.1", tgv2d, 64, 0.0, 0.1, 100, 20),
]
for label, ic, N, nu, dt, chunk, n in cases:
    v = jnp.asarray(ic(N), jnp.float32)
    e0 = ke(v)
    traj = [(0, 1.0)]
    t0 = time.time()
    for i in range(1, n + 1):
        v = api.periodic_fwd(v, nu, dt, chunk, L)
        e = ke(v) / e0
        traj.append((i * chunk, e))
        if not np.isfinite(e) or e > 1e3:
            break
    print(
        label,
        " ".join(f"{s}:{e:.4g}" for s, e in traj),
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
