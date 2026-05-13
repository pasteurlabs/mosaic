"""Input factory and diagnostics for ns-grid."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.config import SolverSpec

from .ics import _uniform_flow

_LBM_SOLVERS = {"xlb"}


def make_inputs(
    spec: SolverSpec,
    ic: jax.Array,
    *,
    nu: float,
    dt: float,
    steps: int,
    domain_extent: float = 2 * jnp.pi,
    lbm_N_base: int | None = None,
    obstacle: dict | None = None,
    U_mean: float = 0.5,
    **_,
) -> dict:
    """Build solver input dict, applying LBM dt-scaling when lbm_N_base is set.

    When ic is 1-D (shape (N,)) it is treated as an inflow profile for drag
    optimisation (v0 = uniform background at U_mean, ic → inflow_profile field).
    """
    if ic.ndim == 1:
        N = ic.shape[0]
        _dt, _steps = dt, steps
        if spec.name in _LBM_SOLVERS and lbm_N_base is not None:
            _dt = dt * min(1.0, lbm_N_base / N)
            _steps = max(1, round(steps * max(1.0, N / lbm_N_base)))
        base = {
            "v0": _uniform_flow(N, U=U_mean),
            "inflow_profile": ic,
            "viscosity": jnp.array([nu], dtype=jnp.float32),
            "dt": jnp.array([_dt], dtype=jnp.float32),
            "steps": _steps,
            "domain_extent": float(domain_extent),
        }
        if obstacle is not None:
            base["obstacle"] = obstacle
            base["boundary_conditions"] = {
                "x_lo": {"type": "periodic"},
                "x_hi": {"type": "periodic"},
                "y_lo": {"type": "no_slip"},
                "y_hi": {"type": "no_slip"},
            }
        return {**base, **spec.input_overrides}

    N = ic.shape[0]
    _dt, _steps = dt, steps
    if spec.name in _LBM_SOLVERS and lbm_N_base is not None:
        _dt = dt * min(1.0, lbm_N_base / N)
        _steps = max(1, round(steps * max(1.0, N / lbm_N_base)))

    base = {
        "v0": ic,
        "viscosity": jnp.array([nu], dtype=jnp.float32),
        "dt": jnp.array([_dt], dtype=jnp.float32),
        "steps": _steps,
        "domain_extent": float(domain_extent),
    }
    if obstacle is not None:
        base["obstacle"] = obstacle
        base["boundary_conditions"] = {
            "x_lo": {"type": "neumann"},
            "x_hi": {"type": "neumann"},
            "y_lo": {"type": "no_slip"},
            "y_hi": {"type": "no_slip"},
        }
    return {**base, **spec.input_overrides}


def _divergence_rms(arr: jax.Array, domain_extent: float = 2 * jnp.pi, **_) -> float:
    """RMS divergence ∇·u for 2D fields (N,N,1,2)."""
    ndim = arr.shape[-1]
    dx = domain_extent / arr.shape[0]
    div = sum(
        (jnp.roll(arr[..., i], -1, i) - jnp.roll(arr[..., i], 1, i))
        for i in range(ndim)
    ) / (2 * dx)
    return float(jnp.sqrt(jnp.mean(div**2)))


def _kinetic_energy(arr: jax.Array, **_) -> float:
    """Mean kinetic energy ½〈|u|²〉."""
    return float(0.5 * jnp.mean(jnp.sum(arr**2, axis=-1)))


def _energy_spectrum(arr: jax.Array, **_) -> dict:
    """Isotropic 1-D energy spectrum E(k) for 2D fields."""
    ndim = arr.shape[-1]
    N = arr.shape[0]
    kn = jnp.fft.fftfreq(N, d=1.0 / N)

    if ndim == 2:
        v_hat = [jnp.fft.fft2(arr[:, :, 0, d]) / N**2 for d in range(2)]
        axes = jnp.meshgrid(kn, kn, indexing="ij")
    else:
        v_hat = [jnp.fft.fftn(arr[..., d]) / N**3 for d in range(3)]
        axes = jnp.meshgrid(kn, kn, kn, indexing="ij")

    E_hat = 0.5 * sum(jnp.abs(vh) ** 2 for vh in v_hat)
    K = jnp.sqrt(sum(ax**2 for ax in axes))
    k_bins = np.arange(1, N // 2)
    E_k = jnp.array(
        [float(E_hat[(k - 0.5 <= K) & (k + 0.5 > K)].sum()) for k in k_bins]
    )
    return {"k": k_bins.tolist(), "E_k": E_k.tolist()}


DIAGNOSTICS = {
    "divergence_rms": _divergence_rms,
    "kinetic_energy": _kinetic_energy,
    "energy_spectrum": _energy_spectrum,
}
