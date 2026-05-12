"""Cost suite: forward and VJP wall-clock timing sweeps.

Experiments:
  spatial_cost   — forward pass timing vs problem size N (fixed step count)
  temporal_cost  — forward pass timing vs step count (fixed problem size)
  vjp_cost       — VJP (backward pass) timing vs N and steps (differentiable solvers only)

All three read from the cost runs payload.

Timing semantics (apply to every experiment in this module):
  * The reported ``mean`` / ``std`` are in **seconds**, aggregated across
    ``cost.n_trials`` repeated ``t.apply(inputs)`` calls (or ``_vjp_grad(...)``
    for vjp_cost). Default ``n_trials = 3``. Plot code multiplies by 1000 for
    a millisecond presentation.
  * Aggregation is **mean** across trials (no min/median variant). ``std``
    captures trial-to-trial variance.
  * Each (solver, sweep-point) pair does **one unreported warmup call** before
    the trial loop. This absorbs per-solver JIT compilation, first-touch CUDA
    kernel caching, and scan-unroll tracing so steady-state latencies are timed.
  * Each entry also records ``vram_peak_mib`` and ``ram_peak_mib`` (peak GPU
    VRAM and container RAM during the trial batch, sampled by a background
    poller thread).
  * A per-trial wall-clock limit of ``_SPATIAL_WALL_S`` seconds is enforced.
    If the first trial of a given N exceeds this limit, remaining N values for
    that solver are marked None and the sweep stops early.
  * Solvers that raise an exception at a given N are also stopped early (all
    remaining N values set to None).
  * ``vjp_cost`` additionally stores one gradient field snapshot per
    (solver, N) in ``gradient_fields.npz`` alongside the JSON result, using
    the same positional-index convention as the gradient suite.
  * What the ``perf_counter`` window encloses:
        — serialization of ``inputs`` to JSON on the host side;
        — HTTP round-trip to the Tesseract container (uvicorn worker);
        — solver compute (whatever ``apply`` / ``vector_jacobian_product``
          does inside the container);
        — deserialization of the response back to Python on the host.
    It does **NOT** include container create/teardown or Docker image build.
  * Host-side RPC and (de)serialization together add ~2-20 ms of fixed cost
    per call.

The per-trial timing loop, peak-memory sampling, and exception classification
live in ``core/harness.py`` (``run_timed_trials``) so every suite shares the
same semantics. This file only owns the cost-specific input construction,
sweep dispatch, and result wiring.

Run from the terminal:
    mosaic run <problem> cost [--experiments EXPR] [--plots-only]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.console import console
from mosaic.benchmarks.core.hardware import get_hardware_info
from mosaic.benchmarks.core.harness import run_timed_trials
from mosaic.benchmarks.core.io import (
    experiment_dir,
    results_dir,
    save_experiment,
    save_field_snapshots_npz,
)
from mosaic.benchmarks.core.runner import current_worker_context, per_solver_loop
from mosaic.benchmarks.core.utils import (
    active_differentiable_solvers,
    active_solvers,
    iter_runs,
)

_SUITE = "cost"

# Maximum wall-clock seconds for a single forward/VJP trial.  If the first
# trial at a given N exceeds this, the solver's sweep stops early (remaining
# N values are recorded as None).
_SPATIAL_WALL_S = 1000


def _mark_remaining_none(target: dict, values: list, current) -> None:
    """Mark every value after ``current`` in ``values`` as ``None`` in ``target``."""
    for remaining in values[values.index(current) + 1 :]:
        target[remaining] = None


# ── Internal context + helpers (module-level to keep complexity manageable) ──


@dataclass
class _CostCtx:
    """Bundle of state shared between the run-impl and its inner closures."""

    cfg: Problem
    make_ic: Any
    make_inputs: Callable
    domain_extent: float
    resolution_key: str
    measure: str
    sweep: str
    output_key: str
    ic_key: str

    # Per-run state populated after we resolve ``runs``.
    phys: dict = field(default_factory=dict)
    N_values: list = field(default_factory=list)
    steps_values: list = field(default_factory=list)
    n_trials: int = 3
    ref_N: Any = None
    ref_steps: Any = None
    ref_ic_name: str = ""

    # Output containers (mutated by _work).
    by_N: dict = field(default_factory=dict)
    by_steps: dict = field(default_factory=dict)
    grad_snaps_N: dict = field(default_factory=dict)
    csv_rows: list = field(default_factory=list)
    wall_times: dict = field(default_factory=dict)

    # Callable resolved once: forward apply or vjp_grad with bound keys.
    timed_call: Callable | None = None

    @property
    def is_vjp(self) -> bool:
        return self.measure == "vjp"

    @property
    def sweep_N(self) -> bool:
        return self.sweep in ("spatial", "both")

    @property
    def sweep_steps(self) -> bool:
        return self.sweep in ("temporal", "both")

    @property
    def vjp_tag(self) -> str:
        return "VJP " if self.is_vjp else ""


def _resolve_solvers(cx: _CostCtx, exp_key: str) -> list[str]:
    """Active solver names for this sweep — vjp filters down to differentiable."""
    if not cx.is_vjp:
        return active_solvers(cx.cfg, "cost", exp_key)
    diff = set(active_differentiable_solvers(cx.cfg, "cost", "vjp_cost")) & set(
        active_differentiable_solvers(cx.cfg, "gradient")
    )
    return [s.name for s in cx.cfg.solvers if s.name in diff]


def _build_inputs_N(cx: _CostCtx, name: str, res):
    _phys = {**cx.phys, cx.resolution_key: res}
    if cx.ref_steps is not None:
        _phys["steps"] = cx.ref_steps
    ic = cx.make_ic[cx.ref_ic_name](L=cx.domain_extent, **_phys)
    return cx.make_inputs(name, ic, domain_extent=cx.domain_extent, **_phys)


def _build_inputs_steps(cx: _CostCtx, name: str, ic_ref, steps):
    return cx.make_inputs(
        name,
        ic_ref,
        domain_extent=cx.domain_extent,
        **{**cx.phys, cx.resolution_key: cx.ref_N, "steps": steps},
    )


def _record_with_grad(cx: _CostCtx, result, target, name, val, *, capture_grads):
    """Write a successful trial's record into ``target`` (and snapshot grads)."""
    if cx.is_vjp:
        grad_norm = (
            float(jnp.linalg.norm(result.last_value))
            if result.last_value is not None
            else None
        )
        target[val] = result.as_record(grad_norm=grad_norm)
        if capture_grads and result.last_value is not None:
            cx.grad_snaps_N[name][val] = np.array(result.last_value)
    else:
        target[val] = result.as_record()


@dataclass
class _AxisSpec:
    """Per-axis sweep recipe (built once per call to ``_work_one_solver``)."""

    label: str  # "N" key (e.g. "nx") or "steps"
    kind_word: str  # "sizes" or "counts" (banner phrasing)
    values: list
    target: dict
    build_inputs: Callable  # value -> inputs
    csv_row: Callable  # (value, record) -> dict
    emit_first_trial_warn: bool
    capture_grads: bool


def _sweep_axis(
    cx: _CostCtx, name: str, color: str, t, ctx_worker, spec: _AxisSpec
) -> None:
    """Run the timed-trial loop for a single sweep axis (N or steps)."""
    console.print(
        f"  [{color}]{name}[/]  {cx.vjp_tag}{spec.label} sweep "
        f"({len(spec.values)} {spec.kind_word}, {cx.n_trials} trials each)"
    )
    for val in spec.values:
        inputs = spec.build_inputs(val)
        result = run_timed_trials(
            lambda: cx.timed_call(t, inputs),
            n_trials=cx.n_trials,
            wall_limit_s=_SPATIAL_WALL_S,
            gpu_id=ctx_worker.gpu_id,
            image_tag=ctx_worker.image_tag,
            capture_value=cx.is_vjp,
        )

        if result.failure is not None:
            console.print(
                f"  [yellow][WARN][/] {name} {cx.vjp_tag}{spec.label}={val} failed "
                f"({result.failure['failure_type']}): "
                f"{result.failure['exc_msg'][:80]}"
            )
            spec.target[val] = result.as_record()
            _mark_remaining_none(spec.target, spec.values, val)
            return

        if spec.emit_first_trial_warn and result.wall_limit_hit:
            console.print(
                f"  [yellow][WARN][/] {name} {cx.vjp_tag}{spec.label}={val}: "
                f"first trial {result.first_elapsed:.1f}s > {_SPATIAL_WALL_S}s limit"
            )

        _record_with_grad(
            cx, result, spec.target, name, val, capture_grads=spec.capture_grads
        )
        cx.csv_rows.append(spec.csv_row(val, spec.target[val]))

        if result.wall_limit_hit:
            _mark_remaining_none(spec.target, spec.values, val)
            return


def _csv_factory_N(cx: _CostCtx, name: str) -> Callable:
    def _row(res, record):
        row: dict = {"solver": name}
        if cx.sweep == "both":
            row["sweep"] = "N"
        row[cx.resolution_key] = res
        row["ref_steps"] = cx.ref_steps
        return {**row, **record}

    return _row


def _csv_factory_steps(cx: _CostCtx, name: str) -> Callable:
    def _row(steps, record):
        row: dict = {"solver": name}
        if cx.sweep == "both":
            row["sweep"] = "steps"
        row[cx.resolution_key] = cx.ref_N
        row["steps"] = steps
        return {**row, **record}

    return _row


def _work_one_solver(cx: _CostCtx, name: str, t) -> None:
    """Per-solver worker body — runs the configured sweeps for one solver.

    Wall-time bookkeeping is handled by :func:`per_solver_loop`; this body
    only owns the cost-specific N/steps sweep dispatch.
    """
    color = cx.cfg.solver(name).color
    ctx_worker = current_worker_context()

    if cx.sweep_N and cx.N_values:
        _sweep_axis(
            cx,
            name,
            color,
            t,
            ctx_worker,
            _AxisSpec(
                label=cx.resolution_key,
                kind_word="sizes",
                values=cx.N_values,
                target=cx.by_N[name],
                build_inputs=lambda res: _build_inputs_N(cx, name, res),
                csv_row=_csv_factory_N(cx, name),
                emit_first_trial_warn=True,
                capture_grads=cx.is_vjp,
            ),
        )

    if cx.sweep_steps and cx.steps_values:
        _phys_ref = {**cx.phys, cx.resolution_key: cx.ref_N}
        ic_ref = cx.make_ic[cx.ref_ic_name](L=cx.domain_extent, **_phys_ref)
        _sweep_axis(
            cx,
            name,
            color,
            t,
            ctx_worker,
            _AxisSpec(
                label="steps",
                kind_word="counts",
                values=cx.steps_values,
                target=cx.by_steps[name],
                build_inputs=lambda steps: _build_inputs_steps(cx, name, ic_ref, steps),
                csv_row=_csv_factory_steps(cx, name),
                emit_first_trial_warn=False,
                capture_grads=False,
            ),
        )


def _save_gradient_snapshots(cx: _CostCtx, out_dir, solver_names_list) -> None:
    if not (cx.is_vjp and cx.sweep_N and cx.grad_snaps_N):
        return
    if not any(snaps for snaps in cx.grad_snaps_N.values()):
        return
    if not cx.N_values:
        return
    per_solver = {
        name: {f"N{res}": arr for res, arr in snaps.items()}
        for name, snaps in cx.grad_snaps_N.items()
        if snaps
    }
    save_field_snapshots_npz(
        out_dir,
        solver_names=solver_names_list,
        per_solver_arrays=per_solver,
        shared_arrays={"N_values": np.array(cx.N_values)},
        prefixes=("grad",),
    )
    console.print(f"  Saved gradient fields → {out_dir / 'gradient_fields.npz'}")


def _label_for(cx: _CostCtx) -> str:
    if cx.is_vjp:
        return "run_vjp_cost"
    return "run_spatial_cost" if cx.sweep == "spatial" else "run_temporal_cost"


def _run_cost_impl(
    cfg: Problem,
    tags: dict[str, str],
    *,
    make_ic,
    make_inputs,
    domain_extent: float,
    resolution_key: str,
    measure: str,
    sweep: str,
    exp_key: str,
    output_key: str = "",
    ic_key: str = "",
    runs=None,
    **overrides,
) -> dict:
    """Parametric wall-clock timing harness shared by spatial / temporal / VJP cost.

    Parameters:
        measure  — "forward" times ``t.apply(inputs)``;
                   "vjp"     times ``_vjp_grad(t, inputs, output_key, ic_key)``
                             and captures gradient snapshots.
        sweep    — "spatial"  sweeps ``N_values`` at fixed ``ref_steps``;
                   "temporal" sweeps ``steps_values`` at fixed ``ref_N``;
                   "both"     sweeps both axes (used by vjp_cost).
        exp_key  — output sub-directory name under ``<results>/<problem>/cost/``.

    Returns:
        Forward sweeps return ``{"by_N": ...}`` or ``{"by_steps": ...}``.
        Both-sweeps (vjp_cost) returns ``{"by_N": ..., "by_steps": ...}``.
    """
    cx = _CostCtx(
        cfg=cfg,
        make_ic=make_ic,
        make_inputs=make_inputs,
        domain_extent=domain_extent,
        resolution_key=resolution_key,
        measure=measure,
        sweep=sweep,
        output_key=output_key,
        ic_key=ic_key,
    )

    if cx.is_vjp:
        from mosaic.benchmarks.problems.shared.gradient import _vjp_grad

        cx.timed_call = lambda t, inputs: _vjp_grad(t, inputs, output_key, ic_key)
    else:
        cx.timed_call = lambda t, inputs: t.apply(inputs)

    if not runs:
        raise NotImplementedError(
            f"{_label_for(cx)} requires runs= payload (not configured for '{cfg.name}')"
        )
    run = next(iter_runs(runs, overrides), None)
    if run is None:
        return {}

    solver_names_list = _resolve_solvers(cx, exp_key)
    if cx.is_vjp and not solver_names_list:
        console.print("  [yellow]No differentiable solvers — skipping vjp_cost[/]")
        return {}

    cost_cfg = run.get("cost", {})
    cx.N_values = cost_cfg.get("N_values", [])
    cx.steps_values = cost_cfg.get("steps_values", [])
    cx.n_trials = cost_cfg.get("n_trials", 3)
    cx.phys = run.get("physics", {})

    if cx.sweep == "spatial" and not cx.N_values:
        raise NotImplementedError(
            f"{_label_for(cx)} requires cost.N_values in runs payload"
            f" (not configured for '{cfg.name}')"
        )
    if cx.sweep == "temporal" and not cx.steps_values:
        raise NotImplementedError(
            f"{_label_for(cx)} requires cost.steps_values in runs payload"
            f" (not configured for '{cfg.name}')"
        )

    cx.ref_steps = (
        cx.steps_values[len(cx.steps_values) // 2]
        if cx.steps_values
        else cx.phys.get("steps")
    )
    cx.ref_N = (
        cx.N_values[len(cx.N_values) // 2] if cx.N_values else cx.phys.get("N", 64)
    )
    cx.ref_ic_name = next(iter(make_ic))
    hardware = get_hardware_info()

    # Forward sweeps pre-allocate entries for every cfg solver (excluded ones
    # remain empty) — matches the legacy result-dict shape. The VJP path
    # pre-allocates only differentiable solvers (excluded ones never appear).
    preallocate = [s.name for s in cfg.solvers] if not cx.is_vjp else solver_names_list
    if cx.sweep_N:
        cx.by_N = {name: {} for name in preallocate}
    if cx.sweep_steps:
        cx.by_steps = {name: {} for name in preallocate}
    if cx.is_vjp and cx.sweep_N:
        cx.grad_snaps_N = {name: {} for name in solver_names_list}

    gpu_ids = overrides.get("gpu_ids")
    cx.wall_times = per_solver_loop(
        cfg,
        tags,
        solver_names_list,
        lambda name, t: _work_one_solver(cx, name, t),
        gpu_ids=gpu_ids,
    )

    result: dict = {"params": run, "hardware": hardware}
    if cx.sweep_N:
        result["by_N"] = cx.by_N
    if cx.sweep_steps:
        result["by_steps"] = cx.by_steps

    out_dir = experiment_dir(
        results_dir(),
        cfg.name,
        _SUITE,
        exp_key,
        suffix="_debug" if overrides.get("debug") else "",
    )
    save_experiment(
        result,
        out_dir,
        csv_rows=cx.csv_rows,
        cfg=cfg,
        harness_fn=_run_cost_impl,
        wall_time_s=cx.wall_times,
    )

    _save_gradient_snapshots(cx, out_dir, solver_names_list)
    return result


# ── Public wrappers ──────────────────────────────────────────────────────────


def run_spatial_cost(
    cfg: Problem,
    tags: dict[str, str],
    *,
    make_ic,
    make_inputs,
    domain_extent: float,
    resolution_key: str,
    runs=None,
    **overrides,
) -> dict:
    """Wall-clock timing sweep over problem sizes (N) at a fixed step count.

    Problem-semantics state is passed explicitly. ``cfg`` is only used for the
    runtime registry (``cfg.solvers``, ``cfg.solver(name)``, ``cfg.name``,
    ``cfg.exclusions`` via :func:`active_solvers`).

    Returns:
        {"by_N": {solver: {N: {"mean", "std", "vram_peak_mib", "ram_peak_mib"}
                             or None on failure}},
         "hardware": {...}}
    """
    return _run_cost_impl(
        cfg,
        tags,
        make_ic=make_ic,
        make_inputs=make_inputs,
        domain_extent=domain_extent,
        resolution_key=resolution_key,
        measure="forward",
        sweep="spatial",
        exp_key="spatial_cost",
        runs=runs,
        **overrides,
    )


def run_temporal_cost(
    cfg: Problem,
    tags: dict[str, str],
    *,
    make_ic,
    make_inputs,
    domain_extent: float,
    resolution_key: str,
    runs=None,
    **overrides,
) -> dict:
    """Wall-clock timing sweep over step counts at a fixed problem size.

    Returns:
        {"by_steps": {solver: {steps: {"mean", "std", ...resource stats}}},
         "hardware": {...}}
    """
    return _run_cost_impl(
        cfg,
        tags,
        make_ic=make_ic,
        make_inputs=make_inputs,
        domain_extent=domain_extent,
        resolution_key=resolution_key,
        measure="forward",
        sweep="temporal",
        exp_key="temporal_cost",
        runs=runs,
        **overrides,
    )


def run_vjp_cost(
    cfg: Problem,
    tags: dict[str, str],
    *,
    make_ic,
    make_inputs,
    domain_extent: float,
    resolution_key: str,
    output_key: str,
    ic_key: str,
    runs=None,
    **overrides,
) -> dict:
    """Wall-clock timing of the VJP (backward pass) for differentiable solvers.

    Sweeps both N (spatial) and steps (temporal) in one function, mirroring
    run_spatial_cost and run_temporal_cost but timing jax.grad instead of
    t.apply.  Non-differentiable solvers are skipped.

    Gradient fields (one snapshot per solver per N value) are saved to
    ``gradient_fields.npz`` alongside the JSON result, keyed as
    ``grad_{si}_N{N}`` where si is the solver index in ``solver_names``.

    Reads the cost runs payload — same params dict as spatial/temporal cost.

    Returns:
        {"by_N":     {solver: {N:     {"mean", "std", "grad_norm",
                                       "vram_peak_mib", "ram_peak_mib"}
                               or None on failure}},
         "by_steps": {solver: {steps: {"mean", "std", "grad_norm",
                                       "vram_peak_mib", "ram_peak_mib"}
                               or None on failure}},
         "hardware": {...}}
    """
    return _run_cost_impl(
        cfg,
        tags,
        make_ic=make_ic,
        make_inputs=make_inputs,
        domain_extent=domain_extent,
        resolution_key=resolution_key,
        measure="vjp",
        sweep="both",
        exp_key="vjp_cost",
        output_key=output_key,
        ic_key=ic_key,
        runs=runs,
        **overrides,
    )
