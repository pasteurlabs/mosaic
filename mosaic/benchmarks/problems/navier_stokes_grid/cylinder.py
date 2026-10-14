# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow past a cylinder: the Schäfer–Turek DFG 2D-1 benchmark.

Steady channel flow at Re = Ū D / ν = 20 (Schäfer & Turek 1996, test 2D-1):

- channel [0, 2.2] × [0, 0.41], cylinder of diameter D = 0.1 centered at
  (0.2, 0.2), no-slip on the channel walls and the cylinder;
- inlet at x = 0 with the parabolic profile u_x(y) = 4 U_m y (H − y) / H²,
  U_m = 0.3, so the mean inflow speed is Ū = 2 U_m / 3 = 0.2;
- outlet at x = 2.2 with zero normal velocity gradient and p = 0;
- ν = 10⁻³ and ρ = 1.

Every run uses a uniform grid with ``resolution`` cells per diameter, which
fixes the grid at (22 × resolution) × (4.1 × resolution) cells. Solvers return
the force on the cylinder at every step; the drag and lift coefficients
Cd = 2 F_x / (Ū² D) and Cl = 2 F_y / (Ū² D) are its mean over the final
:data:`WINDOW` of the run. Δp is the pressure difference between the front
and back of the cylinder, (0.15, 0.2) and (0.25, 0.2).

This module is the single home of the geometry: :func:`channel_inputs` builds
the solver inputs for both ``forward/cylinder`` and ``optimization/drag_opt``.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mosaic_shared.problems.navier_stokes_grid import control_volume_force

from mosaic.benchmarks.core.experiment import (
    KernelContext,
    _build_result_envelope,
    _flatten_by_solver,
    kernel,
)
from mosaic.benchmarks.core.io import save_field_snapshots_npz
from mosaic.benchmarks.core.runner import get_last_apply_error, safe_apply_with_extras
from mosaic.benchmarks.core.status_checks import CheckOutcome, ForwardSummary
from mosaic.benchmarks.core.utils import active_solvers

LENGTH = 2.2
HEIGHT = 0.41
DIAMETER = 0.1
CENTER = (0.2, 0.2)
U_MAX = 0.3
U_MEAN = 2.0 * U_MAX / 3.0
VISCOSITY = 1e-3
FRONT = (0.15, 0.2)
BACK = (0.25, 0.2)

# Reference values for the steady state. Schäfer & Turek (1996) bound them by
# Cd ∈ [5.57, 5.59], Cl ∈ [0.0104, 0.0110] and Δp ∈ [0.1172, 0.1176].
CD_REF = 5.5795
CL_REF = 0.0106
DP_REF = 0.1175

# Simulated time. Started from the inflow profile copied across the channel,
# a staircase finite-volume solver settles to within 1e-4 of its final Cd by
# t ≈ 6 at 10, 20 and 40 cells per diameter. 8 leaves margin for slower solvers.
T_END = 8.0
# Time step: a convective CFL of 0.3 at the peak inflow speed, capped where
# forward-Euler diffusion is stable (ν dt / dx² ≤ 0.2 < 1/4).
CFL = 0.3
DIFFUSION_NUMBER = 0.2
# Fraction of the force history averaged for Cd and Cl. The window before it
# measures the drift.
WINDOW = 0.2

BOUNDARY_CONDITIONS = {
    "x_lo": {"type": "dirichlet"},
    "x_hi": {"type": "neumann"},
    "y_lo": {"type": "no_slip"},
    "y_hi": {"type": "no_slip"},
}
OBSTACLE = {"shape": "cylinder", "center": list(CENTER), "radius": DIAMETER / 2}


# ── Geometry ─────────────────────────────────────────────────────────────────


def grid(resolution: int) -> tuple[int, int, float]:
    """``(nx, ny, dx)`` for ``resolution`` cells per diameter."""
    dx = DIAMETER / resolution
    nx, ny = round(LENGTH / dx), round(HEIGHT / dx)
    if not (math.isclose(nx * dx, LENGTH) and math.isclose(ny * dx, HEIGHT)):
        raise ValueError(
            f"resolution {resolution} does not tile the {LENGTH} × {HEIGHT} "
            "channel; use a multiple of 10 cells per diameter"
        )
    return nx, ny, dx


def cell_centers(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """1-D cell-center coordinates ``(x, y)``."""
    nx, ny, dx = grid(resolution)
    return (np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dx


def solid_mask(resolution: int) -> np.ndarray:
    """``(nx, ny)`` bool, True for cells whose center lies inside the cylinder."""
    x, y = cell_centers(resolution)
    r2 = (x[:, None] - CENTER[0]) ** 2 + (y[None, :] - CENTER[1]) ** 2
    return r2 < (DIAMETER / 2) ** 2


def inflow_profile(resolution: int = 20, **_: Any) -> jax.Array:
    """DFG 2D-1 parabolic inlet profile u_x(y) at the cell-center heights, shape (ny,)."""
    _, y = cell_centers(resolution)
    return jnp.asarray(4.0 * U_MAX * y * (HEIGHT - y) / HEIGHT**2, dtype=jnp.float32)


def time_step(resolution: int, t_end: float = T_END) -> tuple[float, int]:
    """``(dt, steps)`` reaching ``t_end`` exactly."""
    _, _, dx = grid(resolution)
    dt = min(CFL * dx / U_MAX, DIFFUSION_NUMBER * dx**2 / VISCOSITY)
    steps = math.ceil(t_end / dt - 1e-9)
    return t_end / steps, steps


def channel_inputs(profile: Any, resolution: int, t_end: float = T_END) -> dict:
    """Solver inputs for the DFG channel with inlet profile ``profile`` (shape (ny,)).

    The initial velocity is the inlet profile copied along x, zero inside the
    cylinder. It carries no gradient: the steady state does not depend on it.
    """
    dt, steps = time_step(resolution, t_end)
    fluid = jnp.asarray(~solid_mask(resolution))
    ux = jnp.where(fluid, jax.lax.stop_gradient(jnp.asarray(profile))[None, :], 0.0)
    v0 = jnp.stack([ux, jnp.zeros_like(ux)], axis=-1)[:, :, None, :]
    return {
        "v0": v0.astype(jnp.float32),
        "inflow_profile": profile,
        "viscosity": jnp.array([VISCOSITY], dtype=jnp.float32),
        "dt": jnp.array([dt], dtype=jnp.float32),
        "steps": steps,
        "domain_extent": [LENGTH, HEIGHT],
        "boundary_conditions": BOUNDARY_CONDITIONS,
        "obstacle": OBSTACLE,
    }


# ── Quantities of interest ───────────────────────────────────────────────────


def window_size(n: int) -> int:
    """Number of trailing samples in the averaging window of an ``n``-sample history."""
    return max(1, round(WINDOW * n))


def drag_coefficient(body_force: Any) -> Any:
    """Cd from a ``(steps, 2)`` force history; differentiable under JAX."""
    w = window_size(body_force.shape[0])
    return 2.0 * body_force[-w:, 0].mean() / (U_MEAN**2 * DIAMETER)


def force_summary(body_force: np.ndarray) -> dict[str, float]:
    """Cd and Cl over the final window, and the relative drift of Cd into it."""
    f = np.asarray(body_force, dtype=np.float64)
    w = window_size(len(f))
    scale = 2.0 / (U_MEAN**2 * DIAMETER)
    last = f[-w:].mean(axis=0) * scale
    prev = f[-2 * w : -w].mean(axis=0) * scale if len(f) >= 2 * w else last
    return {
        "cd": float(last[0]),
        "cl": float(last[1]),
        "cd_drift": float(abs(last[0] - prev[0]) / abs(last[0])),
    }


def _surface_pressure(
    p: np.ndarray, fluid: np.ndarray, point: tuple[float, float], dx: float, side: int
) -> float:
    """Pressure at ``point`` on the cylinder surface from the adjacent fluid cells.

    Each of the two rows straddling the point is extrapolated linearly in x
    from its two fluid cells nearest the point on ``side`` (−1 upstream, +1
    downstream). The two rows are then interpolated in y.
    """
    jf = point[1] / dx - 0.5
    j0 = math.floor(jf)
    values = []
    for j in (j0, j0 + 1):
        cells = np.flatnonzero(fluid[:, j])
        x = (cells + 0.5) * dx
        cells = cells[x < point[0]][-2:] if side < 0 else cells[x > point[0]][:2]
        (xa, xb) = (cells + 0.5) * dx
        pa, pb = p[cells[0], j], p[cells[1], j]
        values.append(pa + (pb - pa) * (point[0] - xa) / (xb - xa))
    w = jf - j0
    return float((1 - w) * values[0] + w * values[1])


def pressure_difference(p: np.ndarray, resolution: int) -> float:
    """Δp = p(front) − p(back) from a ``(nx, ny)`` cell-centered pressure field."""
    _, _, dx = grid(resolution)
    fluid = ~solid_mask(resolution)
    p = np.asarray(p, dtype=np.float64)
    front = _surface_pressure(p, fluid, FRONT, dx, side=-1)
    back = _surface_pressure(p, fluid, BACK, dx, side=1)
    return front - back


def control_volume_cd(
    velocity: np.ndarray, p: np.ndarray, resolution: int
) -> tuple[float, float]:
    """Cd and Cl from a momentum balance over a box around the cylinder.

    The box spans one diameter upstream, two downstream and one to either
    side of the cylinder center, well inside the fluid at every resolution.
    """
    _, _, dx = grid(resolution)

    def index(coord: float) -> int:
        return round(coord / dx - 0.5)

    box = (
        index(CENTER[0] - DIAMETER),
        index(CENTER[0] + 2 * DIAMETER),
        index(CENTER[1] - DIAMETER),
        index(CENTER[1] + DIAMETER),
    )
    v = np.asarray(velocity, dtype=np.float64)
    fx, fy = control_volume_force(
        v[:, :, 0, 0],
        v[:, :, 0, 1],
        np.asarray(p, dtype=np.float64),
        viscosity=VISCOSITY,
        dx=dx,
        box=box,
    )
    scale = 2.0 / (U_MEAN**2 * DIAMETER)
    return float(fx) * scale, float(fy) * scale


def dfg_metrics(
    velocity: np.ndarray, pressure: np.ndarray, body_force: np.ndarray, resolution: int
) -> dict[str, float]:
    """Per-run metrics scored against the DFG 2D-1 reference values.

    ``error`` is the relative Cd error, the primary score. ``cd_cv`` is the
    control-volume drag from the returned fields, a check on the solver's own
    force that does not depend on how it represents the cylinder.
    """
    p = np.asarray(pressure)[:, :, 0]
    out = force_summary(body_force)
    out["dp"] = pressure_difference(p, resolution)
    out["cd_cv"], out["cl_cv"] = control_volume_cd(velocity, p, resolution)
    out["error"] = abs(out["cd"] - CD_REF) / CD_REF
    out["dp_error"] = abs(out["dp"] - DP_REF) / DP_REF
    return out


# ── Status checks ────────────────────────────────────────────────────────────


def max_cd_err(threshold: float, *, resolution: int) -> Any:
    """Anomaly if the relative Cd error at ``resolution`` exceeds ``threshold``.

    Only the canonical resolution is judged: other resolutions show
    convergence, and their errors are expected to differ.
    """
    key = str(resolution)

    def _check(s: ForwardSummary) -> CheckOutcome:
        err = s.errs_by_pval.get(key)
        if err is None:
            return ("anomaly", f"no valid Cd at {resolution} cells per diameter")
        if err <= threshold:
            return None
        cd = s.metrics_by_pval.get(key, {}).get("cd", float("nan"))
        return (
            "anomaly",
            (
                f"Cd {cd:.4g} at {resolution} cells per diameter is "
                f"{100 * err:.1f}% from {CD_REF} (max_cd_err={threshold})"
            ),
        )

    return _check


def max_cd_drift(threshold: float) -> Any:
    """Anomaly if Cd still drifts by more than ``threshold`` at the end of any run."""

    def _check(s: ForwardSummary) -> CheckOutcome:
        drifts = {
            pval: m["cd_drift"]
            for pval, m in s.metrics_by_pval.items()
            if isinstance(m.get("cd_drift"), int | float)
        }
        bad = {pval: d for pval, d in drifts.items() if d > threshold}
        if not bad:
            return None
        pval, drift = max(bad.items(), key=lambda kv: kv[1])
        return (
            "anomaly",
            (
                f"Cd changed by {100 * drift:.2g}% between the last two "
                f"averaging windows at resolution {pval}; not steady "
                f"(max_cd_drift={threshold})"
            ),
        )

    return _check


# ── Kernel ───────────────────────────────────────────────────────────────────


def _cylinder_aggregate(
    by_solver: Any,
    *,
    cfg: Any,
    run: Any,
    snapshots: Any,
    sweep_values: Any,
    sweep_key: Any,
    out_dir: Any,
    snapshot_filename: Any,
    snapshot_prefixes: Any,
    **_kw: Any,
) -> dict:
    """Write ``fields.npz`` and the result envelope.

    Each solver is scored against the literature on its own, so unlike
    ``agreement`` there is no cross-solver reference to form. Per solver and
    sweep index ``i``, ``fields.npz`` holds the velocity (``{solver}_{i}``),
    pressure (``{solver}_pressure_{i}``) and force history
    (``{solver}_force_{i}``).
    """
    # The framework keys extra snapshots "prefix:index"; store them as
    # "prefix_index" next to the velocity's bare "index".
    per_solver = {
        name: {k.replace(":", "_"): v for k, v in snaps.items()}
        for name, snaps in snapshots.items()
    }
    save_field_snapshots_npz(
        out_dir,
        solver_names=[s.name for s in cfg.solvers],
        per_solver_arrays=per_solver,
        shared_arrays={"sweep_values": np.array([float(v) for v in sweep_values])},
        filename=snapshot_filename,
        prefixes=snapshot_prefixes,
        flat_keys=True,
    )
    return _build_result_envelope(
        cfg=cfg,
        suite=_kw.get("suite", "forward"),
        exp_key=_kw.get("exp_key", "cylinder"),
        run=run,
        sweep_key=sweep_key,
        sweep_values=sweep_values,
        results=_flatten_by_solver(by_solver, sweep_key),
        extras={
            "reference": {"cd": CD_REF, "cl": CL_REF, "dp": DP_REF},
            "reference_label": "Schäfer & Turek (1996), DFG 2D-1",
        },
    )


@kernel(
    sweep_mode="default",
    ic_sweep=True,
    selector_fn=active_solvers,
    catch=True,
    catch_label="apply failed",
    snapshot_filename="fields.npz",
    snapshot_prefixes=("sweep_values",),
    aggregate_fn=_cylinder_aggregate,
)
def cylinder(t: Any, ctx: KernelContext) -> dict:
    """Run one solver on the DFG channel at one resolution and score it.

    Returns (by_param[resolution][solver] schema)::

        {"valid": True, "error": rel. Cd error, "cd", "cl", "cd_drift", "dp",
         "dp_error", "cd_cv", "cl_cv"}
        | {"valid": False, "apply_error": str | None}
    """
    resolution = int(ctx.sweep_value)
    inputs = ctx.make_inputs(ctx.name, ctx.ic, **ctx.phys)
    velocity, _, state = safe_apply_with_extras(
        t, inputs, ctx.output_key, [], state_keys=["pressure", "body_force"]
    )
    if velocity is None:
        return {"metrics": {"apply_error": get_last_apply_error(), "valid": False}}
    missing = [k for k in ("pressure", "body_force") if k not in state]
    if missing:
        return {
            "metrics": {
                "apply_error": f"solver returned no {' or '.join(missing)}",
                "valid": False,
            }
        }
    velocity = np.asarray(velocity)
    pressure = np.asarray(state["pressure"])
    body_force = np.asarray(state["body_force"])
    metrics = {"valid": True, **dfg_metrics(velocity, pressure, body_force, resolution)}
    return {
        "metrics": metrics,
        "snapshot": velocity,
        "snapshots": {"pressure": pressure, "force": body_force},
    }
