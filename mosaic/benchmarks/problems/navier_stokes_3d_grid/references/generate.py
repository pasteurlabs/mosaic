# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the converged spectral reference for ns-3d-grid ``forward/agreement``.

The 3D Taylor-Green initial condition is *not* advection-free: the nonlinear
term immediately generates a ``w`` component and z-structure (vortex
stretching), so the linearized analytic decay ``u(t) = u(0)·exp(-2νt)`` is only
a short-horizon approximation. Scoring solvers against it measures a large
common-mode bias, not solver accuracy (see issue #123).

This script produces the *converged* reference instead: it integrates the full
nonlinear equations with the spectral ETDRK solver (Exponax, which enforces
incompressibility to machine precision) at a resolution well above the
benchmark grid, then spectrally truncates the final field back to the benchmark
resolution. At the benchmark parameters the field is smooth and the truncation
is exact to ~1e-5 relative, so the downsampled high-N field is a faithful
ground truth for the N=16 accuracy metric.

Run (from the repo root, with ``exponax``/``equinox`` importable)::

    python -m mosaic.benchmarks.problems.navier_stokes_3d_grid.references.generate

It writes ``forward_agreement.npz`` next to this file and prints a
cross-validation table: ``||spectral@N_bench - downsampled(spectral@N_hi)||``
should be tiny (confirming convergence) while
``||downsampled - analytic||`` should be the ~0.11 common-mode bias (confirming
the analytic reference was bias-dominated).
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mosaic.benchmarks.core.reference import save_reference  # noqa: E402
from mosaic.benchmarks.problems.navier_stokes_3d_grid.ics import (  # noqa: E402
    _tgv3d,
    _tgv3d_analytic,
)

# The checked-in Exponax tesseract forward is pure JAX; import it directly so
# the reference is generated with the exact same spectral stepper the benchmark
# ships, at higher resolution.
_EXPONAX_DIR = (
    Path(__file__).resolve().parents[4]
    / "tesseracts"
    / "navier-stokes-grid"
    / "exponax"
)
sys.path.insert(0, str(_EXPONAX_DIR))
from tesseract_api import exponax_fwd  # noqa: E402

# ── forward/agreement experiment parameters (mirror config.py) ──────────────
DOMAIN = "ns-3d-grid"
EXP_KEY = "forward/agreement"
L = 2 * float(np.pi)
N_BENCH = 16
DT = 0.01
STEPS = 50
NU_SWEEP = [0.001, 0.01, 0.05]

# Reference resolution: high enough that spectral truncation to N_BENCH is
# exact to well below the smallest meaningful solver error.
N_HI = 64


def _run(N: int, nu: float) -> np.ndarray:
    """Run the spectral solver on the 3D TGV IC at resolution N (float64)."""
    v0 = _tgv3d(N, L=L).astype(jnp.float64)
    out = exponax_fwd(
        v0=v0,
        dt=DT,
        steps=STEPS,
        viscosity=nu,
        domain_extent=L,
        drag=0.0,
        order=2,
        kolmogorov_forcing=False,
        injection_mode=4,
        injection_scale=1.0,
    )
    return np.asarray(out)


def _spectral_downsample(field: np.ndarray, n_target: int) -> np.ndarray:
    """Spectrally truncate an (N,N,N,3) field to (n_target,)*3 + (3,)."""
    n = field.shape[0]
    fh = np.fft.fftn(field, axes=(0, 1, 2))
    h = n_target // 2
    idx = np.concatenate([np.arange(h), np.arange(n - h, n)])
    fh = fh[np.ix_(idx, idx, idx, range(3))]
    out = np.fft.ifftn(fh, axes=(0, 1, 2)) * (n_target / n) ** 3
    return np.real(out)


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> None:
    """Generate, cross-validate, and write the converged reference NPZ."""
    t_end = DT * STEPS
    refs: dict[int, np.ndarray] = {}

    print(f"forward/agreement converged reference (N_hi={N_HI} -> N={N_BENCH})")
    print(f"t_end = {t_end}, params: dt={DT}, steps={STEPS}, L=2π\n")
    print(f"{'nu':>8}  {'||conv-analytic||':>18}  {'||bench-conv|| (trunc err)':>26}")

    for i, nu in enumerate(NU_SWEEP):
        hi = _run(N_HI, nu)
        conv = _spectral_downsample(hi, N_BENCH).astype(np.float32)
        refs[i] = conv

        bench = _run(N_BENCH, nu)
        analytic = np.asarray(_tgv3d_analytic(_tgv3d(N_BENCH, L=L), nu, t_end, L))
        print(
            f"{nu:>8}  {_rel_err(conv, analytic):>18.4e}  "
            f"{_rel_err(bench, conv):>26.4e}"
        )

    path = save_reference(DOMAIN, EXP_KEY, refs, NU_SWEEP)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
