"""Cost suite: forward and VJP wall-clock timing sweeps.

Three kernels:
  * :func:`spatial_cost`  — forward pass timing vs N (fixed step count)
  * :func:`temporal_cost` — forward pass timing vs step count (fixed N)
  * :func:`vjp_cost`      — VJP (backward pass) timing, registered twice in
                            each problem config (``by_N`` / ``by_steps``
                            variants) since the framework supports one
                            sweep axis at a time and the on-disk schema
                            (one ``{by_N, by_steps}`` result) becomes two
                            single-axis sub-experiment outputs.

Timing semantics (apply to every experiment in this module):
  * ``mean`` / ``std`` are in seconds, aggregated across ``cost.n_trials``
    repeated calls. Default ``n_trials = 3``. Plot code multiplies by 1000
    for a millisecond presentation.
  * ``trials_s`` carries the raw per-trial elapsed times so distributional
    statistics can be recomputed without rerunning.
  * Each (solver, sweep-point) pair does one unreported warmup call before
    the trial loop — absorbs per-solver JIT compilation, first-touch CUDA
    kernel caching, and scan-unroll tracing.
  * Each entry also records ``vram_peak_mib`` and ``ram_peak_mib`` (peak GPU
    VRAM and container RAM during the trial batch).
  * A per-trial wall-clock limit of ``_SPATIAL_WALL_S`` seconds is enforced.
    If the first trial exceeds this limit, the kernel sets ``stop_sweep``
    on its return so the framework marks remaining values as None and
    moves on to the next solver.
  * Solvers that raise an exception at a given value behave the same way —
    the failure-shape record is stored at that value, remaining values
    become None.
  * ``vjp_cost`` additionally stores one gradient field snapshot per
    (solver, sweep-value) under ``gradient_fields.npz``.

Run from the terminal:
    mosaic run <problem> cost [--experiments EXPR] [--plots-only]
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.core.hardware import get_hardware_info
from mosaic.benchmarks.core.harness import run_timed_trials
from mosaic.benchmarks.core.io import save_field_snapshots_npz
from mosaic.benchmarks.core.runner import current_worker_context
from mosaic.benchmarks.core.utils import active_differentiable_solvers, active_solvers
from mosaic.benchmarks.problems.shared.gradient import _vjp_grad

_SPATIAL_WALL_S = 1000


def _timed_kernel(t, ctx: KernelContext, *, timed_call, capture_value=False) -> dict:
    """Shared body for every cost kernel — run ``timed_call`` under
    :func:`run_timed_trials`, return the standard ``mean/std/...`` record,
    optionally capture the last call's value as a snapshot, and signal
    ``stop_sweep`` on wall-limit-hit / failure so the framework marks
    remaining sweep values as None for this solver.
    """
    cost_cfg = ctx.run.get("cost", {})
    n_trials = cost_cfg.get("n_trials", 3)
    worker = current_worker_context()
    result = run_timed_trials(
        timed_call,
        n_trials=n_trials,
        wall_limit_s=_SPATIAL_WALL_S,
        gpu_id=worker.gpu_id,
        image_tag=worker.image_tag,
        capture_value=capture_value,
    )

    record = result.as_record()
    snapshot: np.ndarray | None = None
    if capture_value and result.last_value is not None:
        grad_norm = float(jnp.linalg.norm(result.last_value))
        record = result.as_record(grad_norm=grad_norm)
        snapshot = np.array(result.last_value)

    stop = result.failure is not None or result.wall_limit_hit
    return {"metrics": record, "snapshot": snapshot, "stop_sweep": stop}


# ── Aggregates ───────────────────────────────────────────────────────────────


def _axis_key(sweep_key: str) -> str:
    """``sweep_key`` to ``by_N`` / ``by_steps`` output dict key."""
    return "by_N" if sweep_key != "steps" else "by_steps"


_CI_ISOLATION_NOTE = (
    "Wall-clock times measured on dedicated per-suite VM in CI."
    " Relative rankings reliable; absolute times may vary"
    " ±10-15% across runs."
)


def _cost_aggregate(by_solver, *, cfg, run, sweep_values, sweep_key, **_) -> dict:
    """Build ``{by_N or by_steps: {solver: {val: metrics}}, hardware, params}``.

    Pre-allocates an empty dict for every ``cfg.solver`` so excluded ones
    still appear as ``{}`` in the result — matches the legacy layout.
    """
    out_key = _axis_key(sweep_key)
    by_axis: dict = {s.name: {} for s in cfg.solvers}
    for name, smetrics in by_solver.items():
        by_axis[name] = smetrics
    result = {
        out_key: by_axis,
        "hardware": get_hardware_info(),
        "params": run,
    }
    if os.environ.get("CI"):
        result["_isolation_note"] = _CI_ISOLATION_NOTE
    return result


def _vjp_cost_aggregate(
    by_solver,
    *,
    cfg,
    run,
    sweep_values,
    sweep_key,
    out_dir,
    snapshots,
    snapshot_filename,
    snapshot_prefixes,
    **_,
) -> dict:
    """Like :func:`_cost_aggregate` plus a gradient-field NPZ save."""
    out_key = _axis_key(sweep_key)
    by_axis: dict = dict(by_solver)

    # Save gradient snapshots from successful trials.
    if snapshots:
        shared_key = "N_values" if sweep_key != "steps" else "steps_values"
        save_field_snapshots_npz(
            out_dir,
            solver_names=list(snapshots.keys()),
            per_solver_arrays=snapshots,
            shared_arrays={shared_key: np.array(sweep_values)},
            filename=snapshot_filename,
            prefixes=snapshot_prefixes,
        )

    result = {
        out_key: by_axis,
        "hardware": get_hardware_info(),
        "params": run,
    }
    if os.environ.get("CI"):
        result["_isolation_note"] = _CI_ISOLATION_NOTE
    return result


# ── Kernels ──────────────────────────────────────────────────────────────────


@kernel(
    sweep_mode="default",
    selector_fn=active_solvers,
    ic_sweep=True,
    catch=False,  # cost owns its own failure handling via stop_sweep
    aggregate_fn=_cost_aggregate,
)
def spatial_cost(t, ctx: KernelContext) -> dict:
    """Forward-pass wall-clock timing at one (solver, N) point.

    Returns ``{"metrics": {mean, std, trials_s, vram_peak_mib, ram_peak_mib},
    "stop_sweep": bool}``; on failure / wall-limit-hit, the metrics dict
    carries the failure shape and ``stop_sweep=True``.
    """
    inputs = ctx.make_inputs(ctx.name, ctx.ic, **ctx.phys)
    return _timed_kernel(t, ctx, timed_call=lambda: t.apply(inputs))


@kernel(
    sweep_mode="default",
    selector_fn=active_solvers,
    ic_sweep=False,
    catch=False,
    aggregate_fn=_cost_aggregate,
)
def temporal_cost(t, ctx: KernelContext) -> dict:
    """Forward-pass wall-clock timing at one (solver, steps) point.

    ``ic_sweep=False``: IC shape is independent of ``steps``, so the
    framework builds the IC once at the run's base physics (with the
    fixed N) and reuses it across every step value.
    """
    inputs = ctx.make_inputs(ctx.name, ctx.ic, **ctx.phys)
    return _timed_kernel(t, ctx, timed_call=lambda: t.apply(inputs))


@kernel(
    sweep_mode="default",
    selector_fn=active_differentiable_solvers,
    ic_sweep=True,
    catch=False,
    aggregate_fn=_vjp_cost_aggregate,
    snapshot_filename="gradient_fields.npz",
    snapshot_prefixes=("grad",),
)
def vjp_cost(t, ctx: KernelContext) -> dict:
    """VJP wall-clock timing + gradient snapshot at one (solver, sweep-value) point.

    Registered twice in each problem config — once as a ``by_N`` variant
    sweeping ``N`` at fixed ``steps``, once as a ``by_steps`` variant
    sweeping ``steps`` at fixed ``N``. Each variant produces its own
    ``result.json`` under ``cost/vjp_cost/<variant_name>/`` and shares
    the same kernel + aggregate; the aggregate keys output by sweep_key.
    """
    inputs = ctx.make_inputs(ctx.name, ctx.ic, **ctx.phys)
    return _timed_kernel(
        t,
        ctx,
        timed_call=lambda: _vjp_grad(t, inputs, ctx.output_key, ctx.ic_key),
        capture_value=True,
    )
