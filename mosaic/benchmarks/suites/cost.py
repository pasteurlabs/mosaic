"""Cost suite: forward and VJP wall-clock timing sweeps.

Experiments:
  spatial_cost   — forward pass timing vs problem size N (fixed step count)
  temporal_cost  — forward pass timing vs step count (fixed problem size)
  vjp_cost       — VJP (backward pass) timing vs N and steps (differentiable solvers only)

All three read from cfg.cost_defaults.

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

Run from the terminal:
    cd mosaic
    python -m benchmarks.suites.cost [--experiment EXPR] [--no-plots]
"""

from __future__ import annotations

import time
import traceback

import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.config import ProblemConfig
from mosaic.benchmarks.core.console import console
from mosaic.benchmarks.core.hardware import ResourceSampler, get_hardware_info
from mosaic.benchmarks.core.runner import run_with_gpu_pool
from mosaic.benchmarks.core.utils import (
    _diff_solvers,
    active_solvers,
    experiment_dir,
    iter_runs,
    results_dir,
    save_experiment,
    save_gradient_fields_npz,
)

_SUITE = "cost"

# Maximum wall-clock seconds for a single forward/VJP trial.  If the first
# trial at a given N exceeds this, the solver's sweep stops early (remaining
# N values are recorded as None).
_SPATIAL_WALL_S = 1000


def _classify_failure_import():
    from mosaic.benchmarks.suites.gradient import _classify_failure  # noqa: PLC0415

    return _classify_failure


def _sampler_to_mem(sampler: ResourceSampler) -> dict:
    """Map ResourceSampler summary to the cost-suite memory keys."""
    s = sampler.summary
    return {
        "vram_peak_mib": s.get("peak_gpu_mem_mb"),
        "ram_peak_mib": s.get("peak_ram_mb"),
    }


def _exc_info(exc: Exception) -> dict:
    """Capture full exception details for failure entries."""
    return {
        "exc_type": type(exc).__name__,
        "exc_msg": str(exc),
        "traceback": traceback.format_exc(),
    }


# ── Spatial cost ─────────────────────────────────────────────────────────────


def run_spatial_cost(cfg: ProblemConfig, tags: dict[str, str], **overrides) -> dict:
    """Wall-clock timing sweep over problem sizes (N) at a fixed step count.

    Reads cfg.cost_defaults.  Requires cost.N_values; uses the middle entry
    of cost.steps_values (or physics.steps) as the fixed reference step count.

    Returns:
        {"by_N": {solver: {N: {"mean", "std", "vram_peak_mib", "ram_peak_mib"}
                             or None on failure}},
         "hardware": {...}}
    """
    runs = cfg.cost_defaults
    if not runs:
        raise NotImplementedError(
            f"run_spatial_cost requires cost_defaults (not configured for '{cfg.name}')"
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
            f"run_spatial_cost requires cost.N_values in cost_defaults"
            f" (not configured for '{cfg.name}')"
        )

    ref_steps = (
        steps_values[len(steps_values) // 2] if steps_values else phys.get("steps")
    )
    ref_ic_name = next(iter(cfg.make_ic))
    res_key = cfg.resolution_key
    hardware = get_hardware_info()

    by_N: dict = {name: {} for name in cfg.solvers}
    csv_rows: list[dict] = []
    _wall_times: dict[str, float] = {}
    gpu_ids = overrides.get("gpu_ids")

    _classify_failure = _classify_failure_import()

    def _spatial_work(name: str, t) -> None:
        color = cfg.solvers[name].color
        t_solver = time.perf_counter()
        console.print(
            f"  [{color}]{name}[/]  {res_key} sweep ({len(N_values)} sizes, {n_trials} trials each)"
        )
        from mosaic.benchmarks.core.runner import _tl as _runner_tl

        _gpu_id = getattr(_runner_tl, "gpu_id", None)
        _image_tag = getattr(_runner_tl, "image_tag", None)

        for res in N_values:
            _phys = {**phys, res_key: res}
            if ref_steps is not None:
                _phys["steps"] = ref_steps
            ic = cfg.make_ic[ref_ic_name](L=cfg.domain_extent, **_phys)
            inputs = cfg.make_inputs(name, ic, domain_extent=cfg.domain_extent, **_phys)
            times: list[float] = []
            _hit_limit = False
            sampler = ResourceSampler(gpu_id=_gpu_id, image_tag=_image_tag)
            try:
                with sampler:
                    t.apply(
                        inputs
                    )  # warmup (unreported): absorbs JIT / CUDA / scan-trace cost
                    for i in range(n_trials):
                        t0 = time.perf_counter()
                        t.apply(inputs)
                        elapsed_trial = time.perf_counter() - t0
                        times.append(elapsed_trial)
                        if i == 0 and elapsed_trial > _SPATIAL_WALL_S:
                            console.print(
                                f"  [yellow][WARN][/] {name} {res_key}={res}: "
                                f"first trial {elapsed_trial:.1f}s > {_SPATIAL_WALL_S}s limit"
                            )
                            _hit_limit = True
                            break
                mem = _sampler_to_mem(sampler)
            except Exception as exc:
                mem = _sampler_to_mem(sampler)
                _ft = _classify_failure(type(exc).__name__, str(exc))
                console.print(
                    f"  [yellow][WARN][/] {name} {res_key}={res} failed ({_ft}): {str(exc)[:80]}"
                )
                by_N[name][res] = {
                    "status": "failed",
                    "failure_type": _ft,
                    **_exc_info(exc),
                    **mem,
                }
                for remaining in N_values[N_values.index(res) + 1 :]:
                    by_N[name][remaining] = None
                break

            by_N[name][res] = {
                "mean": float(jnp.mean(jnp.array(times))),
                "std": float(jnp.std(jnp.array(times))) if len(times) > 1 else 0.0,
                **mem,
            }
            csv_rows.append(
                {
                    "solver": name,
                    res_key: res,
                    "ref_steps": ref_steps,
                    **by_N[name][res],
                }
            )
            if _hit_limit:
                for remaining in N_values[N_values.index(res) + 1 :]:
                    by_N[name][remaining] = None
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


def run_temporal_cost(cfg: ProblemConfig, tags: dict[str, str], **overrides) -> dict:
    """Wall-clock timing sweep over step counts at a fixed problem size.

    Reads cfg.cost_defaults.  Requires cost.steps_values; uses the middle
    entry of cost.N_values (or physics.N) as the fixed reference size.

    Returns:
        {"by_steps": {solver: {steps: {"mean", "std", ...resource stats}}},
         "hardware": {...}}
    """
    runs = cfg.cost_defaults
    if not runs:
        raise NotImplementedError(
            f"run_temporal_cost requires cost_defaults (not configured for '{cfg.name}')"
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
            f"run_temporal_cost requires cost.steps_values in cost_defaults"
            f" (not configured for '{cfg.name}')"
        )

    ref_N = N_values[len(N_values) // 2] if N_values else phys.get("N", 64)
    ref_ic_name = next(iter(cfg.make_ic))
    res_key = cfg.resolution_key
    hardware = get_hardware_info()

    by_steps: dict = {name: {} for name in cfg.solvers}
    csv_rows: list[dict] = []
    _wall_times: dict[str, float] = {}
    gpu_ids = overrides.get("gpu_ids")

    _classify_failure = _classify_failure_import()

    def _temporal_work(name: str, t) -> None:
        color = cfg.solvers[name].color
        t_solver = time.perf_counter()
        _phys_ref = {**phys, res_key: ref_N}
        ic_ref = cfg.make_ic[ref_ic_name](L=cfg.domain_extent, **_phys_ref)
        console.print(
            f"  [{color}]{name}[/]  steps sweep ({len(steps_values)} counts, {n_trials} trials each)"
        )
        from mosaic.benchmarks.core.runner import _tl as _runner_tl

        _gpu_id = getattr(_runner_tl, "gpu_id", None)
        _image_tag = getattr(_runner_tl, "image_tag", None)

        for steps in steps_values:
            inputs = cfg.make_inputs(
                name,
                ic_ref,
                domain_extent=cfg.domain_extent,
                **{**_phys_ref, "steps": steps},
            )
            times: list[float] = []
            _hit_limit = False
            sampler = ResourceSampler(gpu_id=_gpu_id, image_tag=_image_tag)
            try:
                with sampler:
                    t.apply(inputs)  # warmup (unreported)
                    for i in range(n_trials):
                        t0 = time.perf_counter()
                        t.apply(inputs)
                        elapsed_trial = time.perf_counter() - t0
                        times.append(elapsed_trial)
                        if i == 0 and elapsed_trial > _SPATIAL_WALL_S:
                            _hit_limit = True
                            break
                mem = _sampler_to_mem(sampler)
            except Exception as exc:
                mem = _sampler_to_mem(sampler)
                _ft = _classify_failure(type(exc).__name__, str(exc))
                console.print(
                    f"  [yellow][WARN][/] {name} steps={steps} failed ({_ft}): {str(exc)[:80]}"
                )
                by_steps[name][steps] = {
                    "status": "failed",
                    "failure_type": _ft,
                    **_exc_info(exc),
                    **mem,
                }
                for remaining in steps_values[steps_values.index(steps) + 1 :]:
                    by_steps[name][remaining] = None
                break

            by_steps[name][steps] = {
                "mean": float(jnp.mean(jnp.array(times))),
                "std": float(jnp.std(jnp.array(times))) if len(times) > 1 else 0.0,
                **mem,
            }
            csv_rows.append(
                {
                    "solver": name,
                    res_key: ref_N,
                    "steps": steps,
                    **by_steps[name][steps],
                }
            )
            if _hit_limit:
                for remaining in steps_values[steps_values.index(steps) + 1 :]:
                    by_steps[name][remaining] = None
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


def run_vjp_cost(cfg: ProblemConfig, tags: dict[str, str], **overrides) -> dict:
    """Wall-clock timing of the VJP (backward pass) for differentiable solvers.

    Sweeps both N (spatial) and steps (temporal) in one function, mirroring
    run_spatial_cost and run_temporal_cost but timing jax.grad instead of
    t.apply.  Non-differentiable solvers are skipped.

    Gradient fields (one snapshot per solver per N value) are saved to
    ``gradient_fields.npz`` alongside the JSON result, keyed as
    ``grad_{si}_N{N}`` where si is the solver index in ``solver_names``.

    Reads cfg.cost_defaults — same params dict as spatial/temporal cost.

    Returns:
        {"by_N":     {solver: {N:     {"mean", "std", "grad_norm",
                                       "vram_peak_mib", "ram_peak_mib"}
                               or None on failure}},
         "by_steps": {solver: {steps: {"mean", "std", "grad_norm",
                                       "vram_peak_mib", "ram_peak_mib"}
                               or None on failure}},
         "hardware": {...}}
    """
    import jax.numpy as jnp

    from mosaic.benchmarks.suites.gradient import _vjp_grad  # reuse — no duplication

    runs = cfg.cost_defaults
    if not runs:
        raise NotImplementedError(
            f"run_vjp_cost requires cost_defaults (not configured for '{cfg.name}')"
        )
    run = next(iter_runs(runs, overrides), None)
    if run is None:
        return {}

    # VJP cost: a solver that's excluded from the entire "gradient" suite (no
    # IC-level adjoint) or specifically from "cost/vjp_cost" should be filtered.
    diff_solver_names = set(_diff_solvers(cfg, "cost", "vjp_cost")) & set(
        _diff_solvers(cfg, "gradient")
    )
    diff_solvers = [(n, s) for n, s in cfg.solvers.items() if n in diff_solver_names]
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
    ref_ic_name = next(iter(cfg.make_ic))
    res_key = cfg.resolution_key
    hardware = get_hardware_info()

    by_N: dict = {name: {} for name, _ in diff_solvers}
    by_steps: dict = {name: {} for name, _ in diff_solvers}
    grad_snaps_N: dict[str, dict] = {name: {} for name, _ in diff_solvers}
    csv_rows: list[dict] = []
    _wall_times: dict[str, float] = {}
    gpu_ids = overrides.get("gpu_ids")
    diff_solver_names_list = [name for name, _ in diff_solvers]

    _classify_failure = _classify_failure_import()

    def _vjp_work(name: str, t) -> None:
        color = cfg.solvers[name].color
        t_solver = time.perf_counter()
        from mosaic.benchmarks.core.runner import _tl as _runner_tl

        _gpu_id = getattr(_runner_tl, "gpu_id", None)
        _image_tag = getattr(_runner_tl, "image_tag", None)

        # ── N sweep (spatial) ─────────────────────────────────────────────
        if N_values:
            console.print(
                f"  [{color}]{name}[/]  VJP {res_key} sweep ({len(N_values)} sizes, {n_trials} trials each)"
            )
            for res in N_values:
                _phys = {**phys, res_key: res}
                if ref_steps is not None:
                    _phys["steps"] = ref_steps
                ic = cfg.make_ic[ref_ic_name](L=cfg.domain_extent, **_phys)
                inputs = cfg.make_inputs(
                    name, ic, domain_extent=cfg.domain_extent, **_phys
                )
                times: list[float] = []
                _hit_limit = False
                _last_grad = None
                sampler = ResourceSampler(gpu_id=_gpu_id, image_tag=_image_tag)
                try:
                    with sampler:
                        _vjp_grad(
                            t, inputs, cfg.output_key, cfg.ic_key
                        )  # warmup (unreported)
                        for i in range(n_trials):
                            t0 = time.perf_counter()
                            g = _vjp_grad(t, inputs, cfg.output_key, cfg.ic_key)
                            elapsed_trial = time.perf_counter() - t0
                            times.append(elapsed_trial)
                            _last_grad = g
                            if i == 0 and elapsed_trial > _SPATIAL_WALL_S:
                                console.print(
                                    f"  [yellow][WARN][/] {name} VJP {res_key}={res}: "
                                    f"first trial {elapsed_trial:.1f}s > {_SPATIAL_WALL_S}s limit"
                                )
                                _hit_limit = True
                                break
                    mem = _sampler_to_mem(sampler)
                except Exception as exc:
                    mem = _sampler_to_mem(sampler)
                    _ft = _classify_failure(type(exc).__name__, str(exc))
                    console.print(
                        f"  [yellow][WARN][/] {name} VJP {res_key}={res} failed ({_ft}): {str(exc)[:80]}"
                    )
                    by_N[name][res] = {
                        "status": "failed",
                        "failure_type": _ft,
                        **_exc_info(exc),
                        **mem,
                    }
                    for remaining in N_values[N_values.index(res) + 1 :]:
                        by_N[name][remaining] = None
                    break

                grad_norm = (
                    float(jnp.linalg.norm(_last_grad))
                    if _last_grad is not None
                    else None
                )
                by_N[name][res] = {
                    "mean": float(jnp.mean(jnp.array(times))),
                    "std": float(jnp.std(jnp.array(times))) if len(times) > 1 else 0.0,
                    "grad_norm": grad_norm,
                    **mem,
                }
                if _last_grad is not None:
                    grad_snaps_N[name][res] = np.array(_last_grad)
                csv_rows.append(
                    {
                        "solver": name,
                        "sweep": "N",
                        res_key: res,
                        "ref_steps": ref_steps,
                        **by_N[name][res],
                    }
                )
                if _hit_limit:
                    for remaining in N_values[N_values.index(res) + 1 :]:
                        by_N[name][remaining] = None
                    break

        # ── steps sweep (temporal) ────────────────────────────────────────
        if steps_values:
            console.print(
                f"  [{color}]{name}[/]  VJP steps sweep ({len(steps_values)} counts, {n_trials} trials each)"
            )
            _phys_ref = {**phys, res_key: ref_N}
            ic_ref = cfg.make_ic[ref_ic_name](L=cfg.domain_extent, **_phys_ref)
            for steps in steps_values:
                inputs = cfg.make_inputs(
                    name,
                    ic_ref,
                    domain_extent=cfg.domain_extent,
                    **{**_phys_ref, "steps": steps},
                )
                times = []
                _hit_limit = False
                sampler = ResourceSampler(gpu_id=_gpu_id, image_tag=_image_tag)
                try:
                    with sampler:
                        _vjp_grad(
                            t, inputs, cfg.output_key, cfg.ic_key
                        )  # warmup (unreported)
                        for i in range(n_trials):
                            t0 = time.perf_counter()
                            g = _vjp_grad(t, inputs, cfg.output_key, cfg.ic_key)
                            elapsed_trial = time.perf_counter() - t0
                            times.append(elapsed_trial)
                            if i == 0 and elapsed_trial > _SPATIAL_WALL_S:
                                _hit_limit = True
                                break
                    mem = _sampler_to_mem(sampler)
                except Exception as exc:
                    mem = _sampler_to_mem(sampler)
                    _ft = _classify_failure(type(exc).__name__, str(exc))
                    console.print(
                        f"  [yellow][WARN][/] {name} VJP steps={steps} failed ({_ft}): {str(exc)[:80]}"
                    )
                    by_steps[name][steps] = {
                        "status": "failed",
                        "failure_type": _ft,
                        **_exc_info(exc),
                        **mem,
                    }
                    for remaining in steps_values[steps_values.index(steps) + 1 :]:
                        by_steps[name][remaining] = None
                    break

                grad_norm = float(jnp.linalg.norm(g)) if times else None
                by_steps[name][steps] = {
                    "mean": float(jnp.mean(jnp.array(times))),
                    "std": float(jnp.std(jnp.array(times))) if len(times) > 1 else 0.0,
                    "grad_norm": grad_norm,
                    **mem,
                }
                csv_rows.append(
                    {
                        "solver": name,
                        "sweep": "steps",
                        res_key: ref_N,
                        "steps": steps,
                        **by_steps[name][steps],
                    }
                )
                if _hit_limit:
                    for remaining in steps_values[steps_values.index(steps) + 1 :]:
                        by_steps[name][remaining] = None
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
        save_gradient_fields_npz(
            out_dir,
            solver_names=[n for n, _ in diff_solvers],
            per_solver_arrays=per_solver,
            shared_arrays={"N_values": np.array(N_values)},
            prefixes=("grad",),
        )
        console.print(f"  Saved gradient fields → {out_dir / 'gradient_fields.npz'}")

    return result


# ── run_all + __main__ ────────────────────────────────────────────────────────

_EXPERIMENTS = {
    "spatial_cost": run_spatial_cost,
    "temporal_cost": run_temporal_cost,
    "vjp_cost": run_vjp_cost,
}


def _plot_fns() -> dict:
    from mosaic.benchmarks.plots.cost import plot_cost

    return {
        "spatial_cost": plot_cost,
        "temporal_cost": plot_cost,
        "vjp_cost": plot_cost,
    }


def run_all(
    cfg: ProblemConfig,
    tags: dict[str, str],
    experiments: list[str] | None = None,
    plots: bool = True,
) -> dict[str, dict]:
    """Run cost experiments and optionally generate plots."""
    from mosaic.benchmarks.core.runner import run_suite

    # Drop temporal_cost for problems that have no time steps (steady-state solvers).
    run = next(iter_runs(cfg.cost_defaults, {}), {})
    has_steps = bool(run.get("cost", {}).get("steps_values"))
    available = {
        k: v for k, v in _EXPERIMENTS.items() if k != "temporal_cost" or has_steps
    }

    return run_suite(
        cfg,
        tags,
        available,
        to_run=experiments,
        plots=plots,
        plot_fns=_plot_fns() if plots else None,
        suite_name=_SUITE,
    )
