"""Residual divergence after one Warp-NS 3-D projection step, in float64.

Replicates divergence_3d_to_complex_kernel / pressure_correct_3d_from_complex_kernel
(central differences, inv_2h) and the spectral solve with three choices of 1/lambda.
"""

import numpy as np


def d(f, ax, h):  # central difference, periodic
    return (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / (2 * h)


def project(u, lam):
    h = L / n
    div = sum(d(u[a], a, h) for a in range(3))
    inv = np.zeros_like(lam)
    nz = np.abs(lam) > 1e-9 * np.abs(lam).max()
    inv[nz] = 1 / lam[nz]
    p = np.real(np.fft.ifftn(np.fft.fftn(div) * inv))
    return [u[a] - d(p, a, h) for a in range(3)]


L = 2 * np.pi
rng = np.random.default_rng(0)
for n in (16, 32, 64):
    h = L / n
    k = np.fft.fftfreq(n, d=1.0 / n)
    K = np.meshgrid(k, k, k, indexing="ij")
    lams = {
        "compact 4/h^2 sin^2(pi k/N) (current 3-D)": -(4 / h**2)
        * sum(np.sin(np.pi * q / n) ** 2 for q in K),
        "continuous (2pi/L)^2 k^2 (current 2-D)": -((2 * np.pi / L) ** 2)
        * sum(q**2 for q in K),
        "wide 1/h^2 sin^2(2pi k/N) (= D.G)": -(1 / h**2)
        * sum(np.sin(2 * np.pi * q / n) ** 2 for q in K),
    }
    # Random field and TGV-like smooth field
    fields = {
        "random": [rng.standard_normal((n, n, n)) for _ in range(3)],
        "smooth k<=2": None,
    }
    x = np.arange(n) * h
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    fields["smooth k<=2"] = [
        np.sin(X) * np.cos(2 * Y),
        np.cos(Z) * np.sin(Y),
        np.sin(2 * X + Z),
    ]
    for fname, u in fields.items():
        div0 = np.linalg.norm(sum(d(u[a], a, h) for a in range(3)))
        for lname, lam in lams.items():
            v = project(u, lam)
            div1 = np.linalg.norm(sum(d(v[a], a, h) for a in range(3)))
            print(
                f"N={n:3d} {fname:12s} {lname:45s} |D u_new|/|D u*| = {div1 / div0:.2e}"
            )
