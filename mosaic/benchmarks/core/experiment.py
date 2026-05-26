"""Generic harness scaffold for per-solver experiments with optional sweeps.

The core of every benchmark experiment is the same skeleton: walk the runs
payload, build the IC, iterate the selected solvers, for each solver iterate
the sweep, save snapshots and ``result.json``. This module owns that
skeleton so the per-experiment code shrinks to a tiny *kernel*.

A kernel is just a function:

    kernel(t, ctx) -> dict

where ``t`` is the tesseract handle and ``ctx`` is a :class:`KernelContext`
carrying the IC, physics, and per-call configuration. The return dict has
some or all of:

  * ``"metrics"`` — recorded under ``by_solver[name]`` (or
    ``by_solver[name][sweep_value]`` when sweeping).
  * ``"snapshot"`` — one array to save under the first
    ``snapshot_prefixes`` entry. Use this for single-array kernels.
  * ``"snapshots"`` — a ``{prefix: array}`` dict when the kernel emits
    multiple per-solver arrays (e.g. Jacobian + gradient).
  * ``"shared"`` — a ``{key: array}`` dict for arrays that belong at the
    NPZ top level (shared across solvers, e.g. the IC or the singular
    spectrum).

Kernels are decorated with :func:`kernel` so :meth:`Problem.add_experiment`
can hand them directly to the registry without an intermediate wrapper:

    @kernel(sweep_mode="default", horizons_shared=True)
    def param_sweep(t, ctx): ...

    problem.add_experiment("gradient/horizon_sweep", param_sweep, ...)

For experiments that need a post-pass across all solvers (Jacobian SVD,
cross-cosine, landscape pass), pass ``aggregate_fn=`` to the decorator —
it receives the populated state plus framework handles (cfg, tags,
gpu_ids, out_dir) and is responsible for both the final result dict and
any NPZ writing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.console import console
from mosaic.benchmarks.core.hardware import ResourceSampler
from mosaic.benchmarks.core.harness import classify_failure
from mosaic.benchmarks.core.io import (
    experiment_dir,
    results_dir,
    save_field_snapshots_npz,
    save_harness_result,
)
from mosaic.benchmarks.core.memory import MemoryPoller, container_id_from_tesseract
from mosaic.benchmarks.core.runner import current_worker_context, per_solver_loop
from mosaic.benchmarks.core.utils import (
    active_differentiable_solvers,
    iter_runs,
)


def random_direction(shape: tuple, key: jax.Array) -> jax.Array:
    """Unit-norm random direction in ``shape``-sized parameter space."""
    v = jax.random.normal(key, shape, dtype=jnp.float32)
    return v / (jnp.linalg.norm(v) + 1e-30)


@dataclass
class KernelContext:
    """Argument bundle handed to each kernel invocation.

    Kernels ignore any field that isn't relevant to their science.
    """

    name: str  # solver name (e.g., "jax_cfd")
    cfg: Problem  # the full Problem — for solver metadata (color, etc.)
    ic: jax.Array  # initial condition (regenerated per value if ic_sweep=True)
    phys: dict[str, Any]  # physics dict, including domain_extent + current sweep value
    sweep_key: str | None  # None for sweep_mode="none"
    sweep_value: Any | None  # None for sweep_mode="none"
    ic_key: str
    output_key: str
    make_inputs: Callable
    domain_extent: float
    run: dict = field(
        default_factory=dict
    )  # full per-run payload for kernel-specific config
    seed: int = 0


Kernel = Callable[[Any, KernelContext], dict]
AggregateFn = Callable[..., dict]


_KERNEL_CONFIG_ATTR = "_mosaic_kernel_config"


def kernel(
    *,
    sweep_mode: str = "none",
    warmup: bool = False,
    ic_sweep: bool = False,
    horizons_shared: bool = False,
    aggregate_fn: AggregateFn | None = None,
    catch: bool = True,
    catch_label: str = "kernel failed",
    selector_fn: Callable[..., list[str]] = active_differentiable_solvers,
    snapshot_filename: str = "gradient_fields.npz",
    snapshot_prefixes: tuple[str, ...] = ("grad",),
):
    """Decorator: tag a ``(t, ctx) -> dict`` function as an experiment kernel.

    Attaches a ``_mosaic_kernel_config`` attribute that
    :meth:`Problem.add_experiment` reads to route registration through
    :func:`run_experiment`.
    """

    def decorate(fn):
        setattr(
            fn,
            _KERNEL_CONFIG_ATTR,
            {
                "sweep_mode": sweep_mode,
                "warmup": warmup,
                "ic_sweep": ic_sweep,
                "horizons_shared": horizons_shared,
                "aggregate_fn": aggregate_fn,
                "catch": catch,
                "catch_label": catch_label,
                "selector_fn": selector_fn,
                "snapshot_filename": snapshot_filename,
                "snapshot_prefixes": snapshot_prefixes,
            },
        )
        return fn

    return decorate


def is_kernel(fn) -> bool:
    """True iff ``fn`` was decorated with :func:`kernel`."""
    return hasattr(fn, _KERNEL_CONFIG_ATTR)


def get_kernel_config(fn) -> dict:
    """Return the framework config attached by :func:`kernel` (or empty)."""
    return getattr(fn, _KERNEL_CONFIG_ATTR, {})


def run_experiment(  # noqa: C901, PLR0913 — generic harness, refactor tracked separately
    cfg: Problem,
    tags: dict[str, str],
    kernel_fn: Kernel,
    *,
    suite: str,
    exp_key: str,
    runs,
    make_ic,
    make_inputs,
    output_key: str,
    ic_key: str,
    domain_extent: float,
    sweep_mode: str = "none",
    warmup: bool = False,
    ic_sweep: bool = False,
    selector_fn: Callable[..., list[str]] = active_differentiable_solvers,
    catch: bool = True,
    catch_label: str = "kernel failed",
    aggregate_fn: AggregateFn | None = None,
    snapshot_filename: str = "gradient_fields.npz",
    snapshot_prefixes: tuple[str, ...] = ("grad",),
    horizons_shared: bool = False,
    **overrides,
) -> dict:
    """Generic per-solver experiment driver.

    Called from the closure :meth:`Problem.add_experiment` builds; not
    typically invoked directly. ``save_harness_result`` is called with
    ``harness_fn=kernel_fn`` so editing the kernel invalidates cached results.
    """
    if sweep_mode not in ("none", "default", "limits"):
        raise ValueError(
            f"sweep_mode must be 'none' | 'default' | 'limits', got {sweep_mode!r}"
        )
    if not runs:
        raise NotImplementedError(
            f"{exp_key!r}: requires runs= payload (not configured for {cfg.name!r})"
        )

    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(make_ic)))
        seed = ic_cfg.get("seed", 0)
        base_phys = run.get("physics", {})
        run_ic_key = run.get("ic_key", ic_key)
        run_output_key = run.get("output_key", output_key)
        gpu_ids = overrides.get("gpu_ids")

        sweep_key: str | None = None
        sweep_values: list = []
        ic_sweep_flag = False
        if sweep_mode != "none":
            sweep_cfg = run.get("sweep", {})
            sweep_key = sweep_cfg.get("key")
            sweep_values = sweep_cfg.get("values", [])
            # Kernel decorator's ``ic_sweep=True`` is the default — the run
            # dict's ``sweep.ic_sweep`` can still flip it on per registration
            # (and stays the only place to flip it on for kernels whose
            # default is False).
            ic_sweep_flag = sweep_cfg.get("ic_sweep", ic_sweep)
            if not sweep_key or not sweep_values:
                raise NotImplementedError(
                    f"{exp_key!r}: requires sweep.key and sweep.values in runs payload"
                )

        if ic_sweep_flag:
            ic_per_val: dict = {
                v: make_ic[ic_name](
                    L=domain_extent, seed=seed, **{**base_phys, sweep_key: v}
                )
                for v in sweep_values
            }
            base_ic = ic_per_val[sweep_values[0]]
        else:
            ic_per_val = None
            base_ic = make_ic[ic_name](L=domain_extent, seed=seed, **base_phys)

        if sweep_mode == "limits" and gpu_ids is None:
            console.print(
                f"  [yellow]WARN[/] {exp_key}: gpu_ids not set — solvers share all "
                "GPUs. Pass --gpu-ids 0 1 2 3 for isolated per-GPU OOM measurements."
            )

        selected = selector_fn(cfg, suite, exp_key)
        # ``mosaic run --only failed,stale,…`` installs a per-cell filter
        # before the experiment loop. When set, prune the active solver
        # list to cells matching the requested state(s); when unset this
        # is a no-op pass-through.
        from .cell_filter import filter_solvers

        selected = filter_solvers(f"{suite}/{exp_key}", selected)
        by_solver: dict = {}
        snapshots: dict[str, dict[str, np.ndarray]] = {}
        shared_extras: dict[str, np.ndarray] = {}
        # Solvers whose Tesseract container failed to start (or whose work
        # raised). Forwarded into the result so the status classifier marks
        # them FAILED instead of NOT_RUN — otherwise a broken container is
        # silently indistinguishable from a solver that wasn't selected.
        solver_failures: dict[str, str] = {}

        def _on_solver_error(name: str, exc: Exception) -> None:
            solver_failures[name] = f"{type(exc).__name__}: {exc}"[:300]

        def _ctx(name: str, val: Any) -> KernelContext:
            _ic = ic_per_val[val] if ic_per_val is not None else base_ic
            phys = {**base_phys, "domain_extent": domain_extent}
            if sweep_key is not None:
                phys[sweep_key] = val
            return KernelContext(
                name=name,
                cfg=cfg,
                ic=_ic,
                phys=phys,
                sweep_key=sweep_key,
                sweep_value=val,
                ic_key=run_ic_key,
                output_key=run_output_key,
                make_inputs=make_inputs,
                domain_extent=domain_extent,
                run=run,
                seed=seed,
            )

        def _absorb(
            out: dict, solver_snaps: dict[str, np.ndarray], idx_suffix: str
        ) -> None:
            """Fold one kernel return into ``solver_snaps`` and ``shared_extras``.

            ``idx_suffix`` is the sweep-iteration suffix ("" for sweep_mode="none",
            ``str(idx)`` otherwise). Multi-prefix snapshots are stored using
            the ``"prefix:" + suffix`` convention that
            :func:`save_field_snapshots_npz` parses.
            """
            single = out.get("snapshot")
            if single is not None:
                solver_snaps[idx_suffix] = np.asarray(single)
            multi = out.get("snapshots") or {}
            for prefix, arr in multi.items():
                solver_snaps[f"{prefix}:{idx_suffix}"] = np.asarray(arr)
            for k, v in (out.get("shared") or {}).items():
                shared_extras.setdefault(k, np.asarray(v))

        def _call_with_sampler(name: str, t, val):
            """Run one kernel invocation with a cheap before/after VRAM+RAM sampler.

            The sampler's ``vram_peak_mib`` / ``ram_peak_mib`` get merged into
            the kernel's metrics via :func:`dict.setdefault`, so kernels that
            already compute their own resource stats (cost suite — true peak
            via ``run_timed_trials``) take precedence. Sampler overhead is a
            handful of NVML/psutil calls (~3–5 ms), so we pay it
            unconditionally; this gives every kernel result a baseline
            VRAM/RAM read without per-kernel boilerplate.
            """
            with ResourceSampler(gpu_id=current_worker_context().gpu_id) as rs:
                out = kernel_fn(t, _ctx(name, val))
            metrics = out.setdefault("metrics", {})
            for k, v in rs.summary.items():
                if v is not None:
                    metrics.setdefault(k, v)
            return out

        def _work(name: str, t) -> None:
            if sweep_mode == "none":
                out = _call_with_sampler(name, t, None)
                by_solver[name] = out.get("metrics", {})
                solver_snaps: dict[str, np.ndarray] = {}
                _absorb(out, solver_snaps, "")
                if solver_snaps:
                    snapshots[name] = solver_snaps
                return

            if warmup and sweep_values:
                color = cfg.solver(name).color
                try:
                    kernel_fn(t, _ctx(name, sweep_values[0]))
                    console.print(f"  [{color}]{name}[/] warmup ok")
                except Exception as wex:
                    console.print(
                        f"  [{color}]{name}[/] warmup skipped ({type(wex).__name__})"
                    )

            solver_results: dict = {}
            solver_snaps: dict[str, np.ndarray] = {}

            if sweep_mode == "default":
                stopped_at: int | None = None
                for idx, val in enumerate(sweep_values):
                    out = _call_with_sampler(name, t, val)
                    solver_results[val] = out.get("metrics", {})
                    _absorb(out, solver_snaps, str(idx))
                    # Kernels may request that the framework abandon
                    # this solver's sweep after the current value (e.g.
                    # cost kernels do this on wall-limit-hit so a slow
                    # solver doesn't burn the remaining trials). The
                    # value at which we stopped is included; everything
                    # after gets marked ``None``.
                    if out.get("stop_sweep"):
                        stopped_at = idx
                        break
                if stopped_at is not None:
                    for remaining in sweep_values[stopped_at + 1 :]:
                        solver_results[remaining] = None
                by_solver[name] = solver_results
                if solver_snaps:
                    snapshots[name] = solver_snaps
                return

            # sweep_mode == "limits"
            color = cfg.solver(name).color
            _gpu_id = current_worker_context().gpu_id
            _cid = container_id_from_tesseract(t)
            failed = False
            fail_reason = ""

            for idx, val in enumerate(sweep_values):
                if failed:
                    solver_results[val] = {"status": "skipped", "reason": fail_reason}
                    continue
                t_step = time.perf_counter()
                with MemoryPoller(_gpu_id, _cid) as poller:
                    try:
                        out = kernel_fn(t, _ctx(name, val))
                        exc: Exception | None = None
                    except Exception as e:
                        out, exc = None, e
                mem = poller.summary
                step_wall = time.perf_counter() - t_step
                vram_str = (
                    f" vram={mem['vram_peak_mib']:.0f}MiB"
                    if mem.get("vram_peak_mib") is not None
                    else ""
                )
                if exc is None:
                    metrics = {
                        **(out.get("metrics") if out else {}),
                        "status": "ok",
                        "wall_time_s": step_wall,
                        **mem,
                    }
                    solver_results[val] = metrics
                    _absorb(out, solver_snaps, str(idx))
                    ram_str = (
                        f" ram={mem['ram_peak_mib']:.0f}MiB"
                        if mem.get("ram_peak_mib") is not None
                        else ""
                    )
                    gn = metrics.get("grad_norm")
                    gn_str = f" grad_norm={gn:.3g}" if isinstance(gn, float) else ""
                    console.print(
                        f"  [{color}]{name}[/] {sweep_key}={val} ok"
                        f"{gn_str}{vram_str}{ram_str} ({step_wall:.1f}s)"
                    )
                else:
                    failure_type = classify_failure(type(exc).__name__, str(exc))
                    err_short = str(exc)[:300]
                    solver_results[val] = {
                        "status": "failed",
                        "failure_type": failure_type,
                        "error": err_short,
                        "wall_time_s": step_wall,
                        **mem,
                    }
                    fail_reason = f"first failure at {sweep_key}={val} ({failure_type})"
                    failed = True
                    console.print(
                        f"  [{color}]{name}[/] [red]FAIL[/] {sweep_key}={val} "
                        f"({failure_type}){vram_str}: {err_short[:80]} ({step_wall:.1f}s)"
                    )

            by_solver[name] = solver_results
            if solver_snaps:
                snapshots[name] = solver_snaps

        wall_times = per_solver_loop(
            cfg,
            tags,
            selected,
            _work,
            gpu_ids=gpu_ids,
            # "limits" mode catches its own per-step failures inline; outer
            # catch would swallow programming errors silently.
            catch=catch and sweep_mode != "limits",
            catch_label=catch_label,
            on_error=_on_solver_error,
        )

        exp_subdir = exp_key
        out_dir = experiment_dir(
            results_dir(),
            cfg.name,
            suite,
            exp_subdir,
            suffix="_debug" if overrides.get("debug") else "",
        )

        # Aggregate is responsible for both the final result dict and any
        # custom NPZ saving (Jacobian SVD writes singular-value arrays the
        # framework has no way to produce). Default path: framework saves a
        # standard NPZ + builds a vanilla result dict.
        if aggregate_fn is None:
            if snapshots:
                shared = {"ic": np.asarray(base_ic), **shared_extras}
                if horizons_shared and sweep_key and sweep_values:
                    shared["horizons"] = np.array(
                        [
                            float(v) * float(base_phys.get("dt", 1.0))
                            for v in sweep_values
                        ]
                        if sweep_key == "steps"
                        else [float(v) for v in sweep_values]
                    )
                save_field_snapshots_npz(
                    out_dir,
                    list(snapshots.keys()),
                    snapshots,
                    shared_arrays=shared,
                    filename=snapshot_filename,
                    prefixes=snapshot_prefixes,
                )
            result = {"by_solver": by_solver, "params": run}
            if sweep_key is not None:
                result["sweep_key"] = sweep_key
        else:
            result = aggregate_fn(
                by_solver,
                run=run,
                cfg=cfg,
                tags=tags,
                out_dir=out_dir,
                selected=selected,
                gpu_ids=gpu_ids,
                snapshots=snapshots,
                shared_extras=shared_extras,
                ic=base_ic,
                sweep_values=sweep_values,
                sweep_key=sweep_key,
                snapshot_filename=snapshot_filename,
                snapshot_prefixes=snapshot_prefixes,
                horizons_shared=horizons_shared,
            )

        # Propagate solver-level failures (container start failed, work raised)
        # so the status classifier marks them FAILED rather than NOT_RUN.
        if solver_failures and isinstance(result, dict):
            existing = result.get("_solver_failures") or {}
            existing.update(solver_failures)
            result["_solver_failures"] = existing

        # Persist this experiment's sweep coordinates so aggregator plots
        # (and downstream readers) don't need to re-parse the experiment
        # name to discover its position in parameter space.
        if isinstance(result, dict):
            _exp = cfg.experiments.get(f"{suite}/{exp_key}")
            if _exp is not None and _exp.coords:
                result.setdefault("coords", dict(_exp.coords))

        save_harness_result(
            result,
            cfg=cfg,
            suite=suite,
            exp_subdir=exp_subdir,
            harness_fn=kernel_fn,
            wall_time_s=wall_times,
            debug=bool(overrides.get("debug")),
        )
        all_results = result

    return all_results
