"""Mask cutoffs, N=8 linear-decay check, and the N=20 long-horizon forward run."""

import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np

L = 2 * np.pi
for N in (8, 16, 20, 32, 64):
    row = []
    for f in (1.0, 2 / 3):
        m = np.asarray(
            ex.nonlin_fun.ConvectionNonlinearFun(
                1,
                N,
                derivative_operator=ex.spectral.build_derivative_operator(1, L, N),
                dealiasing_fraction=f,
            ).dealiasing_mask
        )[0]
        row.append(int(np.nonzero(m)[0].max()))
    print(
        f"N={N}: highest kept |k| per axis: f=1.0 -> {row[0]}, f=2/3 -> {row[1]} (Nyquist {N // 2}, textbook 2/3 keeps |k|<{N / 3:.2f})"
    )


def tgv(N):
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return jnp.asarray(
        np.stack(
            [
                np.sin(X) * np.cos(Y) * np.cos(Z),
                -np.cos(X) * np.sin(Y) * np.cos(Z),
                0 * X,
            ]
        ).astype(np.float32)
    )


def rollout(N, nu, dt, steps, f, every):
    st = ex.stepper.NavierStokesVelocity(
        3, L, N, dt, diffusivity=nu, order=2, dealiasing_fraction=f
    )

    def blk(u, _):
        u = jax.lax.fori_loop(0, every, lambda i, c: st(c).astype(c.dtype), u)
        return u, u

    _, tr = jax.jit(lambda u: jax.lax.scan(blk, u, None, length=steps // every))(tgv(N))
    return np.asarray(tr)


# N=8 baseline run (dt=0.16, 8 steps, nu=0.05): is the 2/3 run pure viscous decay?
u = rollout(8, 0.05, 0.16, 8, 2 / 3, 8)[-1]
lin = np.asarray(tgv(8)) * np.exp(-3 * 0.05 * 1.28)
print(
    "N=8 f=2/3 vs linear viscous decay, rel L2:",
    np.linalg.norm(u - lin) / np.linalg.norm(lin),
)

# horizon_sweep_limits setup: N=20, nu=0.001, dt=0.05, up to 10240 steps (t=512)
for f in (1.0, 2 / 3):
    tr = rollout(20, 0.001, 0.05, 10240, f, 320)
    ke = 0.5 * np.mean(np.sum(tr**2, axis=1), axis=(1, 2, 3))
    fh = np.fft.fftn(tr, axes=(2, 3, 4))
    k = np.abs(np.fft.fftfreq(20) * 20)
    K = np.maximum.reduce(np.meshgrid(k, k, k, indexing="ij"))
    e = np.sum(np.abs(fh) ** 2, axis=1)
    hi = e[:, K >= 6].sum(-1) / e.sum(axis=(1, 2, 3))
    print(
        f"f={f:.3f}: finite={np.all(np.isfinite(tr))}; KE at t=16,32,64,128,256,512:",
        [f"{ke[i]:.3e}" for i in (0, 1, 3, 7, 15, 31)],
        "; energy frac max|k|>=6 at same t:",
        [f"{hi[i]:.1e}" for i in (0, 1, 3, 7, 15, 31)],
    )
