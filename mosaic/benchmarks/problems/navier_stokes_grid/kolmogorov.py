# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kolmogorov flow on the periodic box: setup, statistics and ground truth.

The flow obeys ∂ₜu + (u·∇)u = −∇p + ν∇²u + f − αu with the body force
f = (A sin(κy), 0), κ = 2πk/L, and linear drag α. The drag removes the energy
that the 2D inverse cascade piles up at the largest scales, so the flow
settles into a statistically steady state.

Every field here uses the canonical layout of the ns-grid solvers: shape
(N, N, 1, 2) with cell (i, j) centred at ((i + ½)h, (j + ½)h). Resampling
between grids therefore shifts phases by the offset between cell centres,
since a coarse cell centre does not coincide with a fine one.

This module also holds the ``forward/kolmogorov`` kernel, which runs each
solver's coarse and reference configurations from the ground truth's initial
states, and its status check.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.core.reference import reference_path
from mosaic.benchmarks.core.runner import get_last_apply_error, safe_apply
from mosaic.benchmarks.core.status_checks import CheckOutcome, ForwardSummary
from mosaic.benchmarks.core.utils import active_solvers

from .ics import _multimode

logger = logging.getLogger(__name__)

# ── Setup ────────────────────────────────────────────────────────────────────
# Chosen in the calibration of the benchmark (PR #267): a chaotic, statistically
# steady flow whose 64² run misses a 4× finer one by about 2% per correction
# interval, while that finer run converges and stays predictable for twice
# the evaluation horizon.

DOMAIN_EXTENT = 2 * np.pi
WAVENUMBER = 4
AMPLITUDE = 1.0
DRAG = 0.1
VISCOSITY = 2e-3

# Coarse configuration, the length of one correction interval, and the
# refinement of the reference configuration in space and time.
COARSE_N = 64
COARSE_DT = 0.01
INTERVAL = 0.1
REFERENCE_FACTOR = 4

# Forward times scored against the ground truth, within the time over which
# two converged runs from the same state stay correlated.
HORIZON_TIMES = (1.0, 2.0, 3.0, 4.0)

# Correlation of the vorticity with the reference above which a rollout counts
# as tracking it.
HIGH_CORRELATION = 0.95

# Relative error assigned to a state that is no longer finite. Two
# uncorrelated fields of equal energy are √2 apart, so 2 says the run has left
# the attractor.
DIVERGED_ERROR = 2.0

# Physics keys that switch on the forcing in ``make_inputs``.
FORCING_PHYSICS = {
    "forcing_wavenumber": WAVENUMBER,
    "forcing_amplitude": AMPLITUDE,
    "linear_drag": DRAG,
}


# ── Fields ───────────────────────────────────────────────────────────────────


def cell_centres(n: int, domain_extent: float = DOMAIN_EXTENT) -> np.ndarray:
    """Cell-centre coordinates (j + ½)h of an ``n``-cell periodic axis."""
    return (np.arange(n) + 0.5) * domain_extent / n


def forcing_field(
    n: int,
    domain_extent: float = DOMAIN_EXTENT,
    wavenumber: int = WAVENUMBER,
    amplitude: float = AMPLITUDE,
) -> np.ndarray:
    """The Kolmogorov force (A sin(κy), 0) at the cell centres, (n, n, 1, 2)."""
    kappa = 2 * np.pi * wavenumber / domain_extent
    field = np.zeros((n, n, 1, 2), dtype=np.float32)
    field[..., 0, 0] = amplitude * np.sin(kappa * cell_centres(n, domain_extent))
    return field


def laminar_velocity(
    n: int,
    t: float,
    *,
    viscosity: float,
    drag: float = DRAG,
    domain_extent: float = DOMAIN_EXTENT,
    wavenumber: int = WAVENUMBER,
    amplitude: float = AMPLITUDE,
) -> np.ndarray:
    """Exact flow at time ``t`` after starting from rest, (n, n, 1, 2).

    The force only depends on y and points along x, so the flow stays a shear
    flow u(y, t) whose advection term vanishes. The forced mode then relaxes
    at the rate λ = νκ² + α: u = (A/λ)(1 − e^{−λt}) sin(κy), v = 0.
    """
    kappa = 2 * np.pi * wavenumber / domain_extent
    rate = viscosity * kappa**2 + drag
    scale = (1.0 - np.exp(-rate * t)) / rate
    field = forcing_field(n, domain_extent, wavenumber, amplitude) * scale
    return field.astype(np.float32)


def laminar_reference(
    ic: Any,
    t: float,
    L: float,
    nu: float,
    forcing_wavenumber: int,
    forcing_amplitude: float,
    linear_drag: float,
) -> np.ndarray:
    """Analytic reference of ``forward/kolmogorov_laminar``: the flow from rest."""
    return laminar_velocity(
        np.asarray(ic).shape[0],
        t,
        viscosity=nu,
        drag=linear_drag,
        domain_extent=L,
        wavenumber=forcing_wavenumber,
        amplitude=forcing_amplitude,
    )


laminar_reference.supported_ics = frozenset({"rest"})


def rest(N: int, **_: Any) -> np.ndarray:
    """The fluid at rest, (N, N, 1, 2)."""
    return np.zeros((N, N, 1, 2), dtype=np.float32)


def _wavenumbers(n: int, domain_extent: float) -> np.ndarray:
    return 2 * np.pi * np.fft.fftfreq(n, d=domain_extent / n)


def resample(field: Any, n: int, domain_extent: float = DOMAIN_EXTENT) -> jax.Array:
    """Band-limited resampling of a periodic field onto an ``n``-cell grid.

    ``field`` has the two spatial axes first, (N, N, ...), sampled at the cell
    centres. Coarsening keeps the modes below the target Nyquist frequency,
    refining pads with zeros; the Nyquist modes of both grids are dropped,
    since a real field cannot carry the phase shift between the two sets of
    cell centres on them. Differentiable.
    """
    field = jnp.asarray(field)
    size = field.shape[0]
    if size == n:
        return field
    k = _wavenumbers(size, domain_extent)
    # Cell centres of the two grids are offset by (h_n - h_N)/2.
    shift = np.exp(1j * k * 0.5 * (domain_extent / n - domain_extent / size))
    trailing = (1,) * (field.ndim - 2)
    m = min(size, n) // 2
    src = np.r_[0:m, size - m + 1 : size]
    dst = np.r_[0:m, n - m + 1 : n]
    hat = jnp.fft.fft2(field, axes=(0, 1))
    hat = hat * jnp.asarray(shift).reshape((-1, 1, *trailing))
    hat = hat * jnp.asarray(shift).reshape((1, -1, *trailing))
    out = jnp.zeros((n, n, *field.shape[2:]), dtype=hat.dtype)
    out = out.at[np.ix_(dst, dst)].set(hat[np.ix_(src, src)])
    out = jnp.fft.ifft2(out, axes=(0, 1)) * (n * n) / (size * size)
    return jnp.real(out).astype(field.dtype)


def vorticity(v: Any, domain_extent: float = DOMAIN_EXTENT) -> jax.Array:
    """Spectral vorticity ∂ₓv − ∂ᵧu of a (N, N, 1, 2) field, shape (N, N)."""
    v = jnp.asarray(v)[:, :, 0, :]
    k = _wavenumbers(v.shape[0], domain_extent)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    hat = jnp.fft.fft2(v, axes=(0, 1))
    return jnp.real(jnp.fft.ifft2(1j * kx * hat[..., 1] - 1j * ky * hat[..., 0]))


def vorticity_correlation(
    a: Any, b: Any, domain_extent: float = DOMAIN_EXTENT
) -> jax.Array:
    """Pearson correlation of the vorticities of two fields."""
    wa = vorticity(a, domain_extent)
    wb = vorticity(b, domain_extent)
    wa = wa - wa.mean()
    wb = wb - wb.mean()
    return jnp.sum(wa * wb) / jnp.sqrt(jnp.sum(wa * wa) * jnp.sum(wb * wb))


def relative_l2(a: Any, b: Any) -> jax.Array:
    """‖a − b‖ / ‖b‖."""
    a, b = jnp.asarray(a), jnp.asarray(b)
    return jnp.sqrt(jnp.sum((a - b) ** 2) / jnp.sum(b**2))


def energy_spectrum(v: Any) -> np.ndarray:
    """Kinetic energy per integer wavenumber shell, ½Σ|û|² over |k| ∈ [s−½, s+½).

    Returns shells 0 … N/2 − 1, normalised so that their sum is the mean
    kinetic energy ½⟨|u|²⟩ (Parseval, Nyquist excluded).
    """
    v = np.asarray(v, dtype=np.float64)[:, :, 0, :]
    n = v.shape[0]
    kint = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(kint, kint, indexing="ij")
    shell = np.rint(np.sqrt(kx**2 + ky**2)).astype(int)
    hat = np.fft.fft2(v, axes=(0, 1)) / n**2
    density = 0.5 * np.sum(np.abs(hat) ** 2, axis=-1)
    spectrum = np.bincount(shell.ravel(), weights=density.ravel())
    return spectrum[: n // 2]


def spectrum_error(states: Any, references: Any) -> float:
    """Mean |log₁₀| ratio of time-averaged energy spectra over shells 1 … N/3.

    ``states`` and ``references`` are stacks of fields, (T, N, N, 1, 2).
    Shells above N/3 are left out: there the canonical interfaces of the
    staggered solvers damp the energy by design, not by error.
    """
    a = np.mean([energy_spectrum(s) for s in states], axis=0)
    b = np.mean([energy_spectrum(s) for s in references], axis=0)
    top = max(2, len(a) * 2 // 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.abs(np.log10(a[1:top] / b[1:top]))
    return float(np.mean(err)) if np.all(np.isfinite(err)) else float("inf")


def high_correlation_time(
    correlations: Any, interval: float, threshold: float = HIGH_CORRELATION
) -> float:
    """Time until the correlation first drops below ``threshold``.

    ``correlations[i]`` belongs to frame i, at time i·interval. A rollout that
    never drops below scores one interval past its last frame, the first time
    at which it could have.
    """
    c = np.asarray(correlations, dtype=np.float64)
    below = np.nonzero(~(c >= threshold))[0]
    frame = below[0] if below.size else len(c)
    return float(frame * interval)


def first_unstable_frame(errors: Any, threshold: float = DIVERGED_ERROR) -> int:
    """First frame whose relative error is non-finite or above ``threshold``.

    A stable rollout scores its frame count.
    """
    e = np.asarray(errors, dtype=np.float64)
    bad = np.nonzero(~(e <= threshold))[0]
    return int(bad[0]) if bad.size else len(e)


def capped_error(state: Any, target: Any) -> float:
    """Relative L2 error, :data:`DIVERGED_ERROR` for a non-finite state."""
    state = np.asarray(state)
    if not np.isfinite(state).all():
        return DIVERGED_ERROR
    return min(float(relative_l2(state, target)), DIVERGED_ERROR)


# ── Ground truth ─────────────────────────────────────────────────────────────

# Initial states by role. The forward check and the test split of the
# solver-in-the-loop benchmark share the test states.
GROUND_TRUTH_SEEDS = {
    "train": (0, 1, 2, 3),
    "validation": (4,),
    "test": (5, 6, 7),
}

# Exponax resolutions and spin-up: a burn-in at BURN_IN_N, then the truth at
# TRUTH_N with CONVERGENCE_N alongside to show convergence. The initial states
# are kept band-limited to INITIAL_N, deep in the dissipation range of this
# flow, which regenerates the smallest scales within a fraction of a time unit.
INITIAL_N = 128
TRUTH_N = 1024
CONVERGENCE_N = 512
BURN_IN_N = 512
BURN_IN_TIME = 100.0
# Convergence of the truth is followed for three horizons.
CONVERGENCE_TIME = 12.0
# Frames of the truth and its convergence check.
TRUTH_FRAME = 0.5
# Exponax's float32 ETDRK2 needs a CFL number well below 0.25; at 0.1 its
# undealiased configuration is stable at these resolutions. The speed bound is
# the forced flow's largest, about 7.
_TRUTH_CFL = 0.1
_SPEED_BOUND = 7.0


def truth_dt(n: int) -> float:
    """Timestep of the ground-truth runs on an ``n``-cell grid."""
    steps = int(np.ceil(TRUTH_FRAME / (_TRUTH_CFL * DOMAIN_EXTENT / n / _SPEED_BOUND)))
    return TRUTH_FRAME / steps


Advance = Callable[[np.ndarray, float, int], np.ndarray]


def _run(advance: Advance, v: np.ndarray, t_end: float) -> np.ndarray:
    """Advance ``v`` by ``t_end`` in frames of :data:`TRUTH_FRAME`, returning them."""
    n = v.shape[0]
    dt = truth_dt(n)
    steps = round(TRUTH_FRAME / dt)
    frames = []
    for _ in range(round(t_end / TRUTH_FRAME)):
        v = np.asarray(advance(v, dt, steps), dtype=np.float32)
        if not np.isfinite(v).all():
            raise FloatingPointError(f"ground-truth run diverged at N={n}")
        frames.append(v)
    return np.stack(frames)


def generate_ground_truth(
    advance: Advance, log: Callable[[str], None] = logger.info
) -> dict[str, np.ndarray]:
    """Burned-in initial states and the converged trajectories from the test ones.

    ``advance(v, dt, steps)`` runs the reference solver with the Kolmogorov
    force and drag on the grid of ``v`` and returns the state after ``steps``
    steps of ``dt``. Each seed starts from a random field, spins up for
    :data:`BURN_IN_TIME` at :data:`BURN_IN_N` and continues at
    :data:`TRUTH_N`. For the test seeds the run at :data:`CONVERGENCE_N` from
    the same state measures how long the truth is converged.

    Returns ``initial_<seed>`` at :data:`INITIAL_N` and, for the test seeds,
    ``truth_<seed>`` at the times :data:`HORIZON_TIMES` on the coarse grid,
    plus the convergence record.
    """
    test = set(GROUND_TRUTH_SEEDS["test"])
    out: dict[str, np.ndarray] = {}
    frame_of = {t: round(t / TRUTH_FRAME) - 1 for t in HORIZON_TIMES}
    for seed in sorted({s for seeds in GROUND_TRUTH_SEEDS.values() for s in seeds}):
        v = np.asarray(_multimode(BURN_IN_N, DOMAIN_EXTENT, seed=seed))
        v = _run(advance, v, BURN_IN_TIME)[-1]
        # The truth starts from exactly the field the solvers get.
        initial = np.asarray(resample(v, INITIAL_N))
        start = np.asarray(resample(initial, TRUTH_N))
        out[f"initial_{seed}"] = initial
        log(f"seed {seed}: burned in for {BURN_IN_TIME:g}")
        if seed not in test:
            continue
        horizon = CONVERGENCE_TIME
        truth = _run(advance, start, horizon)
        check = _run(advance, np.asarray(resample(start, CONVERGENCE_N)), horizon)
        truth_coarse = np.stack([np.asarray(resample(f, COARSE_N)) for f in truth])
        check_coarse = np.stack([np.asarray(resample(f, COARSE_N)) for f in check])
        out[f"truth_{seed}"] = truth_coarse[[frame_of[t] for t in HORIZON_TIMES]]
        out[f"convergence_correlation_{seed}"] = np.array(
            [
                float(vorticity_correlation(a, b))
                for a, b in zip(check_coarse, truth_coarse, strict=True)
            ]
        )
        out[f"convergence_error_{seed}"] = np.array(
            [
                float(relative_l2(a, b))
                for a, b in zip(check_coarse, truth_coarse, strict=True)
            ]
        )
        hct = high_correlation_time(out[f"convergence_correlation_{seed}"], TRUTH_FRAME)
        log(f"seed {seed}: N={CONVERGENCE_N} tracks N={TRUTH_N} for {hct:g}")
    out["horizon_times"] = np.asarray(HORIZON_TIMES)
    out["convergence_times"] = TRUTH_FRAME * np.arange(
        1, round(CONVERGENCE_TIME / TRUTH_FRAME) + 1
    )
    return out


def load_ground_truth(domain: str = "ns-grid") -> dict[str, np.ndarray]:
    """The checked-in ground truth of ``forward/kolmogorov``."""
    path = reference_path(domain, "forward/kolmogorov")
    if not path.exists():
        raise FileNotFoundError(
            f"no Kolmogorov ground truth at {path}; generate it with "
            "`mosaic reference -p ns-grid -e forward/kolmogorov`"
        )
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def initial_state(truth: dict, seed: int, n: int) -> np.ndarray:
    """Seed ``seed``'s initial state resampled to an ``n``-cell grid."""
    return np.asarray(resample(truth[f"initial_{seed}"], n), dtype=np.float32)


# ── forward/kolmogorov ───────────────────────────────────────────────────────


def _rollout_errors(
    run: Callable[[np.ndarray, int], np.ndarray | None],
    v0: np.ndarray,
    dt: float,
    times: tuple[float, ...],
    targets: list[np.ndarray],
    n_score: int,
) -> tuple[list[float], list[float], np.ndarray | None, str | None]:
    """Error and correlation against ``targets`` at ``times``, each from ``v0``.

    Each time is its own call from the initial state, so solvers that keep
    their own staggered or lattice state never round-trip through the
    canonical grid. After a non-finite state the remaining times are scored
    as diverged without running.
    """
    errors: list[float] = []
    correlations: list[float] = []
    final, failure = None, None
    for t_end, target in zip(times, targets, strict=True):
        state = None if failure else run(v0, round(t_end / dt))
        if state is None:
            failure = failure or (get_last_apply_error() or "non-finite state")
            errors.append(DIVERGED_ERROR)
            correlations.append(0.0)
            continue
        state = np.asarray(resample(state, n_score))
        errors.append(capped_error(state, target))
        correlations.append(float(vorticity_correlation(state, target)))
        final = state
    return errors, correlations, final, failure


def _check_ground_truth(truth: dict, seeds: tuple[int, ...]) -> None:
    missing = [s for s in seeds if f"truth_{s}" not in truth]
    if missing:
        raise KeyError(f"ground truth lacks trajectories for seeds {missing}")


@kernel(sweep_mode="none", selector_fn=active_solvers)
def forward(t: Any, ctx: KernelContext) -> dict:
    """Coarse and reference configurations of one solver against the ground truth.

    Both start from each test seed's burned-in state, restricted to their grid,
    and are scored on the coarse grid at :data:`HORIZON_TIMES`. ``error`` is
    the reference configuration's median error over the seeds at the last
    time, and ``ratio`` divides it by the coarse configuration's.
    """
    phys = dict(ctx.phys)
    n = int(phys.pop("N"))
    factor = int(phys.pop("reference_factor"))
    dt = float(phys.pop("dt"))
    seeds = tuple(int(s) for s in phys.pop("seeds", GROUND_TRUTH_SEEDS["test"]))
    truth = load_ground_truth(ctx.cfg.name)
    _check_ground_truth(truth, seeds)
    times = tuple(float(x) for x in truth["horizon_times"])

    configs = {"coarse": (n, dt), "reference": (n * factor, dt / factor)}
    metrics: dict[str, Any] = {"valid": True, "times": list(times), "seeds": seeds}
    finals: dict[str, list] = {}
    for name, (grid, step) in configs.items():

        def run(v0: np.ndarray, steps: int, _step: float = step) -> np.ndarray | None:
            inputs = ctx.make_inputs(ctx.name, v0, dt=_step, steps=steps, **phys)
            return safe_apply(t, inputs, ctx.output_key)

        per_seed = []
        for seed in seeds:
            targets = [np.asarray(resample(f, n)) for f in truth[f"truth_{seed}"]]
            v0 = initial_state(truth, seed, grid)
            errors, correlations, final, failure = _rollout_errors(
                run, v0, step, times, targets, n
            )
            per_seed.append(
                {
                    "seed": seed,
                    "errors": errors,
                    "correlations": correlations,
                    "failure": failure,
                }
            )
            finals.setdefault(name, []).append(final)
        horizon_errors = [s["errors"][-1] for s in per_seed]
        metrics[name] = {
            "grid": grid,
            "dt": step,
            "per_seed": per_seed,
            "median_error": float(np.median(horizon_errors)),
            "median_errors": np.median(
                [s["errors"] for s in per_seed], axis=0
            ).tolist(),
            "diverged": any(s["failure"] for s in per_seed),
            "spectrum_error": [
                spectrum_error([f], [resample(truth[f"truth_{s}"][-1], n)])
                if f is not None
                else float("inf")
                for f, s in zip(finals[name], seeds, strict=True)
            ],
        }
    metrics["error"] = metrics["reference"]["median_error"]
    metrics["coarse_error"] = metrics["coarse"]["median_error"]
    metrics["ratio"] = metrics["error"] / metrics["coarse_error"]
    return {"metrics": metrics}


def max_reference_ratio(threshold: float) -> Callable[[ForwardSummary], CheckOutcome]:
    """Anomaly if the reference configuration's error exceeds ``threshold`` × the coarse one's.

    Training against the solver's own reference can improve its accuracy by
    at most the coarse error E minus the reference error τ. With τ ≤ E/4,
    any improvement up to three quarters of E is a real gain rather than a
    better imitation of the reference's own error.
    """

    def _check(s: ForwardSummary) -> CheckOutcome:
        m = s.metrics_by_pval.get("")
        if m is None:
            return None
        if m["ratio"] <= threshold:
            return None
        reason = (
            f"reference error {m['error']:.3g} is {m['ratio']:.2f}× the coarse "
            f"error {m['coarse_error']:.3g}, above {threshold}"
        )
        if m["reference"]["diverged"]:
            reason += "; the reference configuration diverged"
        return ("anomaly", reason)

    return _check
