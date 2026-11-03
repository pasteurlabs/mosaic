# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kolmogorov flow on the periodic box: setup, statistics and ground truth.

The flow obeys ∂ₜu + (u·∇)u = −∇p + ν∇²u + f − αu with the body force
f = (A sin(κy), 0), κ = 2πk/L, and linear drag α. The drag removes the energy
that the 2D inverse cascade piles up at the largest scales, so the flow
settles into a statistically steady state.

Every field here uses the canonical layout of the ns-grid solvers, shape
(N, N, 1, 2) with cell (i, j) centered at ((i + ½)h, (j + ½)h).

This module also holds the ``forward/kolmogorov`` kernel and its status check.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.core.reference import reference_path
from mosaic.benchmarks.core.runner import get_last_apply_error, safe_apply_with_state
from mosaic.benchmarks.core.status_checks import CheckOutcome, ForwardSummary
from mosaic.benchmarks.core.utils import active_solvers, l2_error_rel

from .ics import _multimode

logger = logging.getLogger(__name__)

# ── Setup ────────────────────────────────────────────────────────────────────
# A chaotic, statistically steady flow. A 64² grid resolves it only roughly,
# missing a 4× finer run by about 2% every 0.1 time units, while the converged
# truth stays predictable for longer than the scored horizon.

DOMAIN_EXTENT = 2 * np.pi
WAVENUMBER = 4
AMPLITUDE = 1.0
DRAG = 0.1
VISCOSITY = 2e-3

# The coarse configuration, on whose grid every run is scored, and the grids of
# the forward check. Each grid runs at the coarse configuration's CFL number.
COARSE_N = 64
COARSE_DT = 0.01
FORWARD_GRIDS = (32, 64, 128, 256)

# The length of one correction interval of the solver-in-the-loop study, and the
# refinement of its reference configuration in space and time.
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


def cell_centers(n: int, domain_extent: float = DOMAIN_EXTENT) -> np.ndarray:
    """Cell-center coordinates (j + ½)h of an ``n``-cell periodic axis."""
    return (np.arange(n) + 0.5) * domain_extent / n


def forcing_field(
    n: int,
    domain_extent: float = DOMAIN_EXTENT,
    wavenumber: int = WAVENUMBER,
    amplitude: float = AMPLITUDE,
) -> np.ndarray:
    """The Kolmogorov force (A sin(κy), 0) at the cell centers, (n, n, 1, 2)."""
    kappa = 2 * np.pi * wavenumber / domain_extent
    field = np.zeros((n, n, 1, 2), dtype=np.float32)
    field[..., 0, 0] = amplitude * np.sin(kappa * cell_centers(n, domain_extent))
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
    at the rate λ = νκ² + α, so u = (A/λ)(1 − e^{−λt}) sin(κy) and v = 0.
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
    centers. The two grids' cell centers do not coincide, so resampling shifts
    the phases by their offset. Coarsening keeps the modes below the target
    Nyquist frequency and refining pads with zeros. The Nyquist modes of both
    grids are dropped because a real field cannot carry that phase shift on
    them. Differentiable.
    """
    field = jnp.asarray(field)
    size = field.shape[0]
    if size == n:
        return field
    k = _wavenumbers(size, domain_extent)
    # Cell centers of the two grids are offset by (h_n - h_N)/2.
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


def capped_error(state: Any, target: Any) -> float:
    """Relative L2 error capped at :data:`DIVERGED_ERROR`, which a non-finite state also gets."""
    state = np.asarray(state)
    if not np.isfinite(state).all():
        return DIVERGED_ERROR
    return min(l2_error_rel(state, target), DIVERGED_ERROR)


# ── Ground truth ─────────────────────────────────────────────────────────────

# Seeds of the random fields the initial states spin up from.
GROUND_TRUTH_SEEDS = (5, 6, 7)

# Exponax spins each state up at BURN_IN_N, then runs the truth at TRUTH_N and
# a CONVERGENCE_N run alongside it to show convergence. The initial states are
# band-limited to INITIAL_N, which only drops modes deep in the dissipation
# range. The flow regenerates those within a fraction of a time unit.
INITIAL_N = 128
TRUTH_N = 1024
CONVERGENCE_N = 512
BURN_IN_N = 512
BURN_IN_TIME = 100.0
# Convergence of the truth is followed for three horizons.
CONVERGENCE_TIME = 12.0
# Interval between the stored frames of the truth and its convergence run.
TRUTH_FRAME = 0.5
# Exponax's float32 ETDRK2 needs a CFL number well below 0.25 and is stable at
# 0.1 on these resolutions. The speed bound is about the forced flow's largest
# speed.
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
    """Burned-in initial states and the converged trajectories from them.

    ``advance(v, dt, steps)`` runs the reference solver with the Kolmogorov
    force and drag on the grid of ``v``. Returns ``initial_<seed>`` at
    :data:`INITIAL_N`, ``truth_<seed>`` at :data:`HORIZON_TIMES` on the coarse
    grid, and how closely the :data:`CONVERGENCE_N` run tracks the truth.
    """
    out: dict[str, np.ndarray] = {}
    frame_of = {t: round(t / TRUTH_FRAME) - 1 for t in HORIZON_TIMES}
    for seed in GROUND_TRUTH_SEEDS:
        v = np.asarray(_multimode(BURN_IN_N, DOMAIN_EXTENT, seed=seed))
        v = _run(advance, v, BURN_IN_TIME)[-1]
        # The truth starts from exactly the field the solvers get.
        initial = np.asarray(resample(v, INITIAL_N))
        start = np.asarray(resample(initial, TRUTH_N))
        out[f"initial_{seed}"] = initial
        log(f"seed {seed}: burned in for {BURN_IN_TIME:g}")
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
                l2_error_rel(a, b)
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


Run = Callable[[np.ndarray, int, int], np.ndarray | None]


def _rollout_errors(
    run: Run,
    v0: np.ndarray,
    dt: float,
    times: tuple[float, ...],
    targets: list[np.ndarray],
    n_score: int,
) -> dict[str, Any]:
    """Error against ``targets`` at ``times`` from one run of ``v0``, and its wall time.

    ``times`` are multiples of the first. The run returns the state at each
    of them as a snapshot, so solvers that keep their own staggered or
    lattice state never round-trip through the canonical grid. From the first
    non-finite snapshot on, the times are scored as diverged.
    """
    every = round(times[0] / dt)
    if not np.allclose(times, every * dt * np.arange(1, len(times) + 1)):
        raise ValueError(f"times {times} are not multiples of the first")
    started = time.perf_counter()
    snapshots = run(v0, every * len(times), every)
    seconds = time.perf_counter() - started
    failure = None
    if snapshots is None:
        failure = get_last_apply_error() or "the run returned no snapshots"
    errors: list[float] = []
    for i, (t_end, target) in enumerate(zip(times, targets, strict=True)):
        if not failure and not np.isfinite(snapshots[i]).all():
            failure = f"solver output is not finite at t = {t_end:g}"
        if failure:
            errors.append(DIVERGED_ERROR)
            continue
        errors.append(capped_error(resample(snapshots[i], n_score), target))
    return {"errors": errors, "seconds": seconds, "failure": failure}


def _check_ground_truth(truth: dict, seeds: tuple[int, ...]) -> None:
    missing = [s for s in seeds if f"truth_{s}" not in truth]
    if missing:
        raise KeyError(f"ground truth lacks trajectories for seeds {missing}")


def _seconds_per_time(per_seed: list[dict], horizon: float) -> float | None:
    """Wall time of one time unit, from each seed's run.

    The first seed's run includes compilation and warm-up, so it only counts
    when it is the only seed.
    """
    warm = per_seed[1:] or per_seed
    values = [s["seconds"] / horizon for s in warm if not s["failure"]]
    return float(np.median(values)) if values else None


@kernel(sweep_mode="none", selector_fn=active_solvers)
def forward(t: Any, ctx: KernelContext) -> dict:
    """Each grid of the ladder, scored on the coarse grid against the ground truth.

    ``error`` is the finest grid's median error over the seeds at the last
    time. ``order`` is the observed order of convergence from the coarse grid
    to the finest at the first time.
    """
    phys = dict(ctx.phys)
    n = int(phys.pop("N"))
    dt = float(phys.pop("dt"))
    grids = sorted(int(g) for g in phys.pop("grids"))
    seeds = GROUND_TRUTH_SEEDS
    if n not in grids:
        raise ValueError(f"the coarse grid {n} is not among the grids {grids}")
    truth = load_ground_truth(ctx.cfg.name)
    _check_ground_truth(truth, seeds)
    times = tuple(float(x) for x in truth["horizon_times"])

    runs = []
    for grid in grids:
        step = dt * n / grid

        def run(
            v0: np.ndarray, steps: int, every: int, _step: float = step
        ) -> np.ndarray | None:
            inputs = ctx.make_inputs(
                ctx.name, v0, dt=_step, steps=steps, snapshot_every=every, **phys
            )
            snapshots, _ = safe_apply_with_state(
                t, inputs, "snapshots", require_finite=False
            )
            return None if snapshots is None else np.asarray(snapshots)

        per_seed = []
        for seed in seeds:
            targets = [np.asarray(resample(f, n)) for f in truth[f"truth_{seed}"]]
            v0 = initial_state(truth, seed, grid)
            record = _rollout_errors(run, v0, step, times, targets, n)
            per_seed.append({"seed": seed, **record})
        runs.append(
            {
                "grid": grid,
                "dt": step,
                "per_seed": per_seed,
                "median_errors": np.median(
                    [s["errors"] for s in per_seed], axis=0
                ).tolist(),
                "diverged": any(s["failure"] for s in per_seed),
                "seconds_per_time": _seconds_per_time(per_seed, times[-1]),
            }
        )
    coarse = next(r for r in runs if r["grid"] == n)
    finest = runs[-1]
    ratio = np.float64(finest["median_errors"][0]) / coarse["median_errors"][0]
    return {
        "metrics": {
            "valid": True,
            "times": list(times),
            "seeds": seeds,
            "coarse_grid": n,
            "runs": runs,
            "error": finest["median_errors"][-1],
            "order": float(-np.log(ratio) / np.log(finest["grid"] / n)),
        }
    }


def min_refinement_order(order: float) -> Callable[[ForwardSummary], CheckOutcome]:
    """Anomaly if the coarse grid or a finer one diverged, or if the observed order is below ``order``.

    The order is measured at the first time, while the errors on both grids
    are still small enough that the flow's chaos amplifies them alike. Grids
    below the coarse one are reported but not checked, since they may be too
    coarse for the flow.
    """

    def _check(s: ForwardSummary) -> CheckOutcome:
        m = s.metrics_by_pval.get("")
        if m is None:
            return None
        diverged = [
            r["grid"]
            for r in m["runs"]
            if r["grid"] >= m["coarse_grid"] and r["diverged"]
        ]
        if diverged:
            grids = ", ".join(f"{g}²" for g in diverged)
            return ("anomaly", f"diverged on {grids}")
        if m["order"] >= order:
            return None
        reason = (
            f"observed order {m['order']:.2f} from {m['coarse_grid']}² to "
            f"{m['runs'][-1]['grid']}², below {order}"
        )
        return ("anomaly", reason)

    return _check
