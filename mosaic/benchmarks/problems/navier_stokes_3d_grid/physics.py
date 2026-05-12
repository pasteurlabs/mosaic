"""Input factory and diagnostics for ns-3d-grid."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.config import SolverSpec

_LBM_SOLVERS = {"xlb"}


def build_make_inputs(solvers: list[SolverSpec]) -> Callable:
    """Return a ``make_inputs(solver_name, ic, **physics) → dict`` closure.

    Captures the solver list so per-solver ``input_overrides`` can be merged
    into the final dict without importing :mod:`.config` (which would create
    a cycle, since ``config`` imports from this module).
    """
    spec_by_name = {s.name: s for s in solvers}

    def _make_inputs(
        solver_name: str,
        ic: jax.Array,
        *,
        nu: float,
        dt: float,
        steps: int,
        domain_extent: float = 2 * jnp.pi,
        lbm_N_base: int | None = None,
        **_,
    ) -> dict:
        """Build solver input dict, applying LBM dt-scaling when lbm_N_base is set.

        For standard 3-D periodic runs ic has shape (N, N, N, 3) and is passed
        directly as v0.
        """
        spec = spec_by_name[solver_name]

        N = ic.shape[0]
        _dt, _steps = dt, steps

        if solver_name in _LBM_SOLVERS and lbm_N_base is not None:
            _dt = dt * (lbm_N_base / N)
            _steps = max(1, round(steps * (N / lbm_N_base)))

        base = {
            "v0": ic,
            "viscosity": jnp.array([nu], dtype=jnp.float32),
            "dt": jnp.array([_dt], dtype=jnp.float32),
            "steps": _steps,
            "domain_extent": float(domain_extent),
        }
        return {**base, **spec.input_overrides}

    return _make_inputs


def _divergence_rms(arr: jax.Array, domain_extent: float = 2 * jnp.pi, **_) -> float:
    """RMS divergence ∇·u for 3D (N,N,N,3) fields."""
    dx = domain_extent / arr.shape[0]
    div = sum(
        (jnp.roll(arr[..., i], -1, i) - jnp.roll(arr[..., i], 1, i)) for i in range(3)
    ) / (2 * dx)
    return float(jnp.sqrt(jnp.mean(div**2)))


def _kinetic_energy(arr: jax.Array, **_) -> float:
    """Mean kinetic energy ½〈|u|²〉 for 3D fields."""
    return float(0.5 * jnp.mean(jnp.sum(arr**2, axis=-1)))


def _energy_spectrum(arr: jax.Array, **_) -> dict:
    """Isotropic 1-D energy spectrum E(k) for 3D (N,N,N,3) fields."""
    N = arr.shape[0]
    kn = jnp.fft.fftfreq(N, d=1.0 / N)
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
