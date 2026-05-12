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

import time

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
from mosaic.benchmarks.core.runner import current_worker_context, run_with_gpu_pool
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


# ── Spatial cost ─────────────────────────────────────────────────────────────


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
    if not runs:
        raise NotImplementedError(
            f"run_spatial_cost requires runs= payload (not configured for '{cfg.name}')"
        )
    run = next(iter_runs(runs, overrides), None)
    if run is None:
        return {}

    cost_cfg = run.get("cost", {})
    N_values = cost_cfg.get("N_values", [])
    steps_values = cost_cfg.get("steps_values", [])
    n_trials = cost_cfg.get("n_trials", 3)
    phys = run.get("physics", {})

    if not N_values:
        raise NotImplementedError(
            f"run_spatial_cost requires cost.N_values in runs payload"
            f" (not configured for '{cfg.name}')"
        )

    ref_steps = (
        steps_values[len(steps_values) // 2] if steps_values else phys.get("steps")
    )
    ref_ic_name = next(iter(make_ic))
    res_key = resolution_key
    hardware = get_hardware_info()

    by_N: dict = {s.name: {} for s in cfg.solvers}
    csv_rows: list[dict] = []
    _wall_times: dict[str, float] = {}
    gpu_ids = overrides.get("gpu_ids")

    def _spatial_work(name: str, t) -> None:
        color = cfg.solver(name).color
        t_solver = time.perf_counter()
        console.print(
            f"  [{color}]{name}[/]  {res_key} sweep ({len(N_values)} sizes, {n_trials} trials each)"
        )
        ctx = current_worker_context()

        for res in N_values:
            _phys = {**phys, res_key: res}
            if ref_steps is not None:
                _phys["steps"] = ref_steps
            ic = make_ic[ref_ic_name](L=domain_extent, **_phys)
            inputs = make_inputs(name, ic, domain_extent=domain_extent, **_phys)

            result = run_timed_trials(
                lambda: t.apply(inputs),
                n_trials=n_trials,
                wall_limit_s=_SPATIAL_WALL_S,
                gpu_id=ctx.gpu_id,
                image_tag=ctx.image_tag,
            )

            if result.failure is not None:
                console.print(
                    f"  [yellow][WARN][/] {name} {res_key}={res} failed "
                    f"({result.failure['failure_type']}): "
                    f"{result.failure['exc_msg'][:80]}"
                )
                by_N[name][res] = result.as_record()
                _mark_remaining_none(by_N[name], N_values, res)
                break

            if result.wall_limit_hit:
                console.print(
                    f"  [yellow][WARN][/] {name} {res_key}={res}: "
                    f"first trial {result.first_elapsed:.1f}s > {_SPATIAL_WALL_S}s limit"
                )

            by_N[name][res] = result.as_record()
            csv_rows.append(
                {
                    "solver": name,
                    res_key: res,
                    "ref_steps": ref_steps,
                    **by_N[name][res],
                }
            )
            if result.wall_limit_hit:
                _mark_remaining_none(by_N[name], N_values, res)
                break

        elapsed = time.perf_counter() - t_solver
        _wall_times[name] = elapsed
        console.print(f"  [{color}]{name}[/] done in {elapsed:.1f}s")

    run_with_gpu_pool(
        active_solvers(cfg, "cost", "spatial_cost"),
        tags,
        _spatial_work,
        gpu_ids=gpu_ids,
    )

    result = {"by_N": by_N, "params": run, "hardware": hardware}
    out_dir = experiment_dir(
        results_dir(),
        cfg.name,
        _SUITE,
        "spatial_cost",
        suffix="_debug" if overrides.get("debug") else "",
    )
    save_experiment(
        result,
        out_dir,
        csv_rows=csv_rows,
        cfg=cfg,
        harness_fn=run_spatial_cost,
        wall_time_s=_wall_times,
    )
    return result


# ── Temporal cost ─────────────────────────────────────────────────────────────


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
    if not runs:
        raise NotImplementedError(
            f"run_temporal_cost requires runs= payload (not configured for '{cfg.name}')"
        )
    run = next(iter_runs(runs, overrides), None)
    if run is None:
        return {}

    cost_cfg = run.get("cost", {})
    steps_values = cost_cfg.get("steps_values", [])
    N_values = cost_cfg.get("N_values", [])
    n_trials = cost_cfg.get("n_trials", 3)
    phys = run.get("physics", {})

    if not steps_values:
        raise NotImplementedError(
            f"run_temporal_cost requires cost.steps_values in runs payload"
            f" (not configured for '{cfg.name}')"
        )

    ref_N = N_values[len(N_values) // 2] if N_values else phys.get("N", 64)
    ref_ic_name = next(iter(make_ic))
    res_key = resolution_key
    hardware = get_hardware_info()

    by_steps: dict = {s.name: {} for s in cfg.solvers}
    csv_rows: list[dict] = []
    _wall_times: dict[str, float] = {}
    gpu_ids = overrides.get("gpu_ids")

    def _temporal_work(name: str, t) -> None:
        color = cfg.solver(name).color
        t_solver = time.perf_counter()
        _phys_ref = {**phys, res_key: ref_N}
        ic_ref = make_ic[ref_ic_name](L=domain_extent, **_phys_ref)
        console.print(
            f"  [{color}]{name}[/]  steps sweep ({len(steps_values)} counts, {n_trials} trials each)"
        )
        ctx = current_worker_context()

        for steps in steps_values:
            inputs = make_inputs(
                name,
                ic_ref,
                domain_extent=domain_extent,
                **{**_phys_ref, "steps": steps},
            )
            result = run_timed_trials(
                lambda: t.apply(inputs),
                n_trials=n_trials,
                wall_limit_s=_SPATIAL_WALL_S,
                gpu_id=ctx.gpu_id,
                image_tag=ctx.image_tag,
            )

            if result.failure is not None:
                console.print(
                    f"  [yellow][WARN][/] {name} steps={steps} failed "
                    f"({result.failure['failure_type']}): "
                    f"{result.failure['exc_msg'][:80]}"
                )
                by_steps[name][steps] = result.as_record()
                _mark_remaining_none(by_steps[name], steps_values, steps)
                break

            by_steps[name][steps] = result.as_record()
            csv_rows.append(
                {
                    "solver": name,
                    res_key: ref_N,
                    "steps": steps,
                    **by_steps[name][steps],
                }
            )
            if result.wall_limit_hit:
                _mark_remaining_none(by_steps[name], steps_values, steps)
                break

        elapsed = time.perf_counter() - t_solver
        _wall_times[name] = elapsed
        console.print(f"  [{color}]{name}[/] done in {elapsed:.1f}s")

    run_with_gpu_pool(
        active_solvers(cfg, "cost", "temporal_cost"),
        tags,
        _temporal_work,
        gpu_ids=gpu_ids,
    )

    result = {"by_steps": by_steps, "params": run, "hardware": hardware}
    out_dir = experiment_dir(
        results_dir(),
        cfg.name,
        _SUITE,
        "temporal_cost",
        suffix="_debug" if overrides.get("debug") else "",
    )
    save_experiment(
        result,
        out_dir,
        csv_rows=csv_rows,
        cfg=cfg,
        harness_fn=run_temporal_cost,
        wall_time_s=_wall_times,
    )
    return result


# ── VJP cost ──────────────────────────────────────────────────────────────────


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
    from mosaic.benchmarks.shared.gradient import _vjp_grad  # reuse — no duplication

    if not runs:
        raise NotImplementedError(
            f"run_vjp_cost requires runs= payload (not configured for '{cfg.name}')"
        )
    run = next(iter_runs(runs, overrides), None)
    if run is None:
        return {}

    # VJP cost: a solver that's excluded from the entire "gradient" suite (no
    # IC-level adjoint) or specifically from "cost/vjp_cost" should be filtered.
    diff_solver_names = set(
        active_differentiable_solvers(cfg, "cost", "vjp_cost")
    ) & set(active_differentiable_solvers(cfg, "gradient"))
    diff_solvers = [(s.name, s) for s in cfg.solvers if s.name in diff_solver_names]
    if not diff_solvers:
        console.print("  [yellow]No differentiable solvers — skipping vjp_cost[/]")
        return {}

    cost_cfg = run.get("cost", {})
    N_values = cost_cfg.get("N_values", [])
    steps_values = cost_cfg.get("steps_values", [])
    n_trials = cost_cfg.get("n_trials", 3)
    phys = run.get("physics", {})

    ref_steps = (
        steps_values[len(steps_values) // 2] if steps_values else phys.get("steps")
    )
    ref_N = N_values[len(N_values) // 2] if N_values else phys.get("N", 64)
    ref_ic_name = next(iter(make_ic))
    res_key = resolution_key
    hardware = get_hardware_info()

    by_N: dict = {name: {} for name, _ in diff_solvers}
    by_steps: dict = {name: {} for name, _ in diff_solvers}
    grad_snaps_N: dict[str, dict] = {name: {} for name, _ in diff_solvers}
    csv_rows: list[dict] = []
    _wall_times: dict[str, float] = {}
    gpu_ids = overrides.get("gpu_ids")
    diff_solver_names_list = [name for name, _ in diff_solvers]

    def _vjp_work(name: str, t) -> None:
        color = cfg.solver(name).color
        t_solver = time.perf_counter()
        ctx = current_worker_context()

        # ── N sweep (spatial) ─────────────────────────────────────────────
        if N_values:
            console.print(
                f"  [{color}]{name}[/]  VJP {res_key} sweep ({len(N_values)} sizes, {n_trials} trials each)"
            )
            for res in N_values:
                _phys = {**phys, res_key: res}
                if ref_steps is not None:
                    _phys["steps"] = ref_steps
                ic = make_ic[ref_ic_name](L=domain_extent, **_phys)
                inputs = make_inputs(name, ic, domain_extent=domain_extent, **_phys)

                result = run_timed_trials(
                    lambda: _vjp_grad(t, inputs, output_key, ic_key),
                    n_trials=n_trials,
                    wall_limit_s=_SPATIAL_WALL_S,
                    gpu_id=ctx.gpu_id,
                    image_tag=ctx.image_tag,
                    capture_value=True,
                )

                if result.failure is not None:
                    console.print(
                        f"  [yellow][WARN][/] {name} VJP {res_key}={res} failed "
                        f"({result.failure['failure_type']}): "
                        f"{result.failure['exc_msg'][:80]}"
                    )
                    by_N[name][res] = result.as_record()
                    _mark_remaining_none(by_N[name], N_values, res)
                    break

                if result.wall_limit_hit:
                    console.print(
                        f"  [yellow][WARN][/] {name} VJP {res_key}={res}: "
                        f"first trial {result.first_elapsed:.1f}s > {_SPATIAL_WALL_S}s limit"
                    )

                grad_norm = (
                    float(jnp.linalg.norm(result.last_value))
                    if result.last_value is not None
                    else None
                )
                by_N[name][res] = result.as_record(grad_norm=grad_norm)
                if result.last_value is not None:
                    grad_snaps_N[name][res] = np.array(result.last_value)
                csv_rows.append(
                    {
                        "solver": name,
                        "sweep": "N",
                        res_key: res,
                        "ref_steps": ref_steps,
                        **by_N[name][res],
                    }
                )
                if result.wall_limit_hit:
                    _mark_remaining_none(by_N[name], N_values, res)
                    break

        # ── steps sweep (temporal) ────────────────────────────────────────
        if steps_values:
            console.print(
                f"  [{color}]{name}[/]  VJP steps sweep ({len(steps_values)} counts, {n_trials} trials each)"
            )
            _phys_ref = {**phys, res_key: ref_N}
            ic_ref = make_ic[ref_ic_name](L=domain_extent, **_phys_ref)
            for steps in steps_values:
                inputs = make_inputs(
                    name,
                    ic_ref,
                    domain_extent=domain_extent,
                    **{**_phys_ref, "steps": steps},
                )

                result = run_timed_trials(
                    lambda: _vjp_grad(t, inputs, output_key, ic_key),
                    n_trials=n_trials,
                    wall_limit_s=_SPATIAL_WALL_S,
                    gpu_id=ctx.gpu_id,
                    image_tag=ctx.image_tag,
                    capture_value=True,
                )

                if result.failure is not None:
                    console.print(
                        f"  [yellow][WARN][/] {name} VJP steps={steps} failed "
                        f"({result.failure['failure_type']}): "
                        f"{result.failure['exc_msg'][:80]}"
                    )
                    by_steps[name][steps] = result.as_record()
                    _mark_remaining_none(by_steps[name], steps_values, steps)
                    break

                grad_norm = (
                    float(jnp.linalg.norm(result.last_value))
                    if result.last_value is not None
                    else None
                )
                by_steps[name][steps] = result.as_record(grad_norm=grad_norm)
                csv_rows.append(
                    {
                        "solver": name,
                        "sweep": "steps",
                        res_key: ref_N,
                        "steps": steps,
                        **by_steps[name][steps],
                    }
                )
                if result.wall_limit_hit:
                    _mark_remaining_none(by_steps[name], steps_values, steps)
                    break

        elapsed = time.perf_counter() - t_solver
        _wall_times[name] = elapsed
        console.print(f"  [{color}]{name}[/] done in {elapsed:.1f}s")

    run_with_gpu_pool(diff_solver_names_list, tags, _vjp_work, gpu_ids=gpu_ids)

    result = {"by_N": by_N, "by_steps": by_steps, "params": run, "hardware": hardware}
    out_dir = experiment_dir(
        results_dir(),
        cfg.name,
        _SUITE,
        "vjp_cost",
        suffix="_debug" if overrides.get("debug") else "",
    )
    save_experiment(
        result,
        out_dir,
        csv_rows=csv_rows,
        cfg=cfg,
        harness_fn=run_vjp_cost,
        wall_time_s=_wall_times,
    )

    # Save gradient field snapshots (one per solver per N, from last trial)
    any_grads = any(snaps for snaps in grad_snaps_N.values())
    if any_grads and N_values:
        per_solver = {
            name: {f"N{res}": arr for res, arr in snaps.items()}
            for name, snaps in grad_snaps_N.items()
            if snaps
        }
        save_field_snapshots_npz(
            out_dir,
            solver_names=[n for n, _ in diff_solvers],
            per_solver_arrays=per_solver,
            shared_arrays={"N_values": np.array(N_values)},
            prefixes=("grad",),
        )
        console.print(f"  Saved gradient fields → {out_dir / 'gradient_fields.npz'}")

    return result
