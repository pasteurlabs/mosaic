"""Gradient evaluation suite: FD verification, parameter sweep, Jacobian SVD.

Only runs solvers where SolverSpec.differentiable is True.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.config import ProblemConfig
from mosaic.benchmarks.core.console import console
from mosaic.benchmarks.core.harness import classify_failure as _classify_failure
from mosaic.benchmarks.core.io import (
    experiment_dir,
    results_dir,
    save_experiment,
    save_field_snapshots_npz,
    try_load_npz,
)
from mosaic.benchmarks.core.memory import MemoryPoller, container_id_from_tesseract
from mosaic.benchmarks.core.runner import current_worker_context, run_with_gpu_pool

# JAX-traced closures capture this reference at trace time; using the
# tracer-aware wrapper ensures primitive binding sees the active trace.
from mosaic.benchmarks.core.tracer_apply import apply_tesseract
from mosaic.benchmarks.core.utils import (
    active_differentiable_solvers,
    extract_runs,
    iter_runs,
)

_SUITE = "gradient"


def _random_direction(shape: tuple, key: jax.Array) -> jax.Array:
    v = jax.random.normal(key, shape, dtype=jnp.float32)
    return v / (jnp.linalg.norm(v) + 1e-30)


def _vjp_grad(t, inputs: dict, output_key: str, ic_key: str) -> jax.Array:
    """Gradient of sum(output**2) w.r.t. inputs[ic_key] via jax.grad.

    Uses kinetic-energy-like loss rather than sum(output) so that the gradient
    is non-trivial for divergence-free / momentum-conserving solvers.
    """

    def f(ic):
        out = apply_tesseract(t, {**inputs, ic_key: ic})[output_key]
        return jnp.sum(out**2)

    return jax.grad(f)(inputs[ic_key])


def _fd_cosine(fd_arr: np.ndarray, vjp_arr: np.ndarray) -> float:
    """Subspace cosine: angle between fd and vjp directional-derivative vectors
    across all random directions.  1 = perfectly aligned, 0 = orthogonal."""
    return float(
        np.dot(fd_arr, vjp_arr)
        / (np.linalg.norm(fd_arr) * np.linalg.norm(vjp_arr) + 1e-30)
    )


def _safe_vjp_at(
    t,
    cfg: ProblemConfig,
    name: str,
    ic,
    phys: dict,
    sweep_key: str,
    val,
    output_key: str,
    ic_key: str,
) -> tuple[jax.Array | None, Exception | None]:
    """Compute the VJP gradient at one sweep point, returning ``(grad, exc)``.

    On success ``exc`` is ``None``. On any failure — including non-finite
    gradient values — ``grad`` is ``None`` and ``exc`` carries the cause.
    Used by horizon_sweep_limits so the call site stays flat (no
    try/except nested inside ``with MemoryPoller(...)``).
    """
    try:
        inputs = cfg.make_inputs(
            name, ic, **{**phys, sweep_key: val, "domain_extent": cfg.domain_extent}
        )
        g = _vjp_grad(t, inputs, output_key, ic_key)
        if not jnp.all(jnp.isfinite(g)):
            return None, ValueError("VJP returned non-finite gradient (NaN/Inf)")
        return g, None
    except Exception as exc:
        return None, exc


# ── Finite-difference verification ───────────────────────────────────────────


def run_fd_check(
    cfg: ProblemConfig, tags: dict[str, str], *, _exp_key: str = "fd_check", **overrides
) -> dict:
    """Verify VJP gradients against central finite differences over a range of ε values.

    For each solver × ε × random direction computes:
        - FD directional derivative of L = sum(output²): (L(f(x+εv)) - L(f(x-εv))) / (2ε)
        - VJP directional derivative: <grad_ic L, v>
    Reports relative error and subspace cosine similarity.

    Returns:
        {"by_solver": {solver: {eps: {"rel_error": [float], "cosine": float}}}}
        or {ic_name: <above>} when multiple runs are configured.
    """
    runs = cfg.gradient_defaults.get(_exp_key, [])
    if not runs:
        raise NotImplementedError(
            f"No '{_exp_key}' gradient_defaults configured for '{cfg.name}'"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
        seed = ic_cfg.get("seed", 0)
        fd_cfg = run.get("fd", {})
        eps_values = fd_cfg.get("eps_values", [5e0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4])
        n_dirs = fd_cfg.get("n_dirs", 20)
        phys = run.get("physics", {})
        ic_subdir = ic_name if n_runs > 1 else ""
        # Per-run ic_key and output_key allow source-identification experiments
        # to override the global cfg.ic_key="rho" / cfg.output_key defaults.
        run_ic_key = run.get("ic_key", cfg.ic_key)
        run_output_key = run.get("output_key", cfg.output_key)

        ic = cfg.make_ic[ic_name](seed=seed, L=cfg.domain_extent, **phys)
        keys = jax.random.split(jax.random.PRNGKey(seed), n_dirs)

        # Gate on the experiment-specific exclusion so source_fd_check /
        # source_width_sweep only run on source-differentiable solvers
        # (those without a {suite}/{_exp_key} or bare {_exp_key} exclusion).
        diff_solvers = active_differentiable_solvers(cfg, "gradient", _exp_key)
        results: dict = {}
        grad_snaps: dict = {}
        gpu_ids = overrides.get("gpu_ids")
        _wall_times: dict[str, float] = {}

        def _fd_work(
            name: str, t, _run_ic_key=run_ic_key, _run_output_key=run_output_key
        ) -> None:
            color = cfg.solvers[name].color
            t0 = time.perf_counter()
            try:
                base_inputs = cfg.make_inputs(
                    name, ic, domain_extent=cfg.domain_extent, **phys
                )
                # Use the actual IC from make_inputs as the perturbation base.  For some
                # problems (e.g. structural-mesh) make_inputs may override the passed ic
                # (e.g. via a rho_0 parameter), so we must perturb around base_inputs[ic_key]
                # rather than the raw ic array — otherwise the FD sees no perturbation.
                base_ic = jnp.array(base_inputs[_run_ic_key])
                # Scale ε relative to IC magnitude so eps_values are problem-agnostic.
                ic_scale = float(jnp.sqrt(jnp.mean(base_ic**2) + 1e-30))
                dirs_base = [_random_direction(base_ic.shape, k) for k in keys]
                solver_results: dict = {}
                g = _vjp_grad(t, base_inputs, _run_output_key, _run_ic_key)
                grad_snaps[name] = {"ic": np.array(base_ic), "grad": np.array(g)}
                vjp_arr = np.array(
                    [float(jnp.dot(g.ravel(), v.ravel())) for v in dirs_base]
                )
                for eps in eps_values:
                    abs_eps = eps * ic_scale
                    fd_arr = np.array(
                        [
                            float(
                                jnp.sum(
                                    apply_tesseract(
                                        t,
                                        {
                                            **base_inputs,
                                            _run_ic_key: base_ic + abs_eps * v,
                                        },
                                    )[_run_output_key]
                                    ** 2
                                    - apply_tesseract(
                                        t,
                                        {
                                            **base_inputs,
                                            _run_ic_key: base_ic - abs_eps * v,
                                        },
                                    )[_run_output_key]
                                    ** 2
                                )
                                / (2 * abs_eps)
                            )
                            for v in dirs_base
                        ]
                    )
                    denom = np.maximum(
                        np.maximum(np.abs(fd_arr), np.abs(vjp_arr)), 1e-30
                    )
                    solver_results[eps] = {
                        "rel_error": (np.abs(fd_arr - vjp_arr) / denom).tolist(),
                        "cosine": _fd_cosine(fd_arr, vjp_arr),
                        # Persist the raw per-direction FD derivative so the
                        # rel_error / cosine summaries can be recomputed (or
                        # replaced with alternative statistics) without
                        # rerunning the benchmark. ``vjp_arr`` is the same for
                        # every eps so it's stored once at the solver level.
                        "fd_arr": fd_arr.tolist(),
                    }
                results[name] = {
                    "ic_scale": ic_scale,
                    "vjp_arr": vjp_arr.tolist(),
                    "eps_sweep": solver_results,
                }
                elapsed = time.perf_counter() - t0
                _wall_times[name] = elapsed
                console.print(f"  [{color}]{name}[/] done in {elapsed:.1f}s")
            except Exception as exc:
                console.print(
                    f"  [{color}]{name}[/] [yellow]SKIP (VJP failed: {exc})[/]"
                )

        run_with_gpu_pool(diff_solvers, tags, _fd_work, gpu_ids=gpu_ids)

        exp_subdir = f"{_exp_key}/{ic_subdir}" if ic_subdir else _exp_key
        out_dir = experiment_dir(
            results_dir(),
            cfg.name,
            _SUITE,
            exp_subdir,
            suffix="_debug" if overrides.get("debug") else "",
        )
        solver_names = list(grad_snaps.keys())
        saved_ic = grad_snaps[solver_names[0]]["ic"] if grad_snaps else ic
        save_field_snapshots_npz(
            out_dir,
            solver_names,
            {name: {"": np.asarray(grad_snaps[name]["grad"])} for name in solver_names},
            shared_arrays={"ic": np.asarray(saved_ic)},
        )

        result = {"by_solver": results, "params": run}
        save_experiment(
            result, out_dir, cfg=cfg, harness_fn=run_fd_check, wall_time_s=_wall_times
        )
        if n_runs > 1:
            all_results[ic_name] = result
        else:
            all_results = result

    return all_results


# ── shared ε-sweep helper ─────────────────────────────────────────────────────

_DEFAULT_EPS = [1e0, 1e-1, 1e-2, 1e-3]


def _eps_sweep(
    t,
    cfg,
    name,
    ic,
    dirs,
    eps_values,
    make_inputs_kwargs,
    ic_key: str | None = None,
    output_key: str | None = None,
) -> tuple[dict, jax.Array]:
    """Run FD vs VJP over a list of ε values.

    ic_key and output_key override cfg.ic_key/cfg.output_key for per-experiment
    source-identification gradient checks (source_fd_check etc.).

    Returns:
        result : {"grad_norm", "eps_sweep": {eps: {rel_error_mean, rel_error_std,
                                                    cosine_mean}}}
        grad   : jax.Array  — VJP gradient (reuse by caller, avoids recomputation)
    """
    _ic_key = ic_key if ic_key is not None else cfg.ic_key
    _output_key = output_key if output_key is not None else cfg.output_key
    base_inputs = cfg.make_inputs(name, ic, **make_inputs_kwargs)
    # Use the actual IC from make_inputs as the perturbation base.  For some
    # problems (e.g. structural-mesh) make_inputs may override the passed ic
    # (e.g. via a rho_0 parameter), so we must perturb around base_inputs[ic_key]
    # rather than the raw ic array — otherwise the FD sees no perturbation.
    base_ic = jnp.array(base_inputs[_ic_key])
    # Scale ε relative to IC magnitude so eps_values are problem-agnostic.
    ic_scale = float(jnp.sqrt(jnp.mean(base_ic**2) + 1e-30))
    # Reuse the caller's random directions when they match base_ic's shape (the
    # common case).  Only regenerate if make_inputs changed the IC shape.
    dirs_base = (
        dirs
        if base_ic.shape == ic.shape
        else [
            _random_direction(base_ic.shape, k)
            for k in jax.random.split(jax.random.PRNGKey(0), len(dirs))
        ]
    )

    g = _vjp_grad(t, base_inputs, _output_key, _ic_key)
    vjp_arr = np.array([float(jnp.dot(g.ravel(), v.ravel())) for v in dirs_base])

    eps_sweep: dict = {}
    for eps in eps_values:
        abs_eps = eps * ic_scale
        fd_arr = np.array(
            [
                float(
                    jnp.sum(
                        apply_tesseract(
                            t, {**base_inputs, _ic_key: base_ic + abs_eps * v}
                        )[_output_key]
                        ** 2
                        - apply_tesseract(
                            t, {**base_inputs, _ic_key: base_ic - abs_eps * v}
                        )[_output_key]
                        ** 2
                    )
                    / (2 * abs_eps)
                )
                for v in dirs_base
            ]
        )
        denom = np.maximum(np.maximum(np.abs(fd_arr), np.abs(vjp_arr)), 1e-30)
        eps_sweep[eps] = {
            "rel_error_mean": float(np.mean(np.abs(fd_arr - vjp_arr) / denom)),
            "rel_error_std": float(np.std(np.abs(fd_arr - vjp_arr) / denom)),
            "cosine_mean": _fd_cosine(fd_arr, vjp_arr),
        }

    return {
        "grad_norm": float(jnp.linalg.norm(g)),
        "ic_scale": ic_scale,
        "eps_sweep": eps_sweep,
    }, g


# ── Generic parameter sweep ───────────────────────────────────────────────────


def _run_generic_param_sweep(
    cfg: ProblemConfig,
    tags: dict[str, str],
    exp_key: str,
    **overrides,
) -> dict:
    """Shared implementation for param_sweep and horizon_sweep experiments.

    Reads gradient_defaults[exp_key] and saves results to
    results/<problem>/gradient/<exp_key>/.
    """
    runs = cfg.gradient_defaults.get(exp_key, [])
    if not runs:
        raise NotImplementedError(
            f"No '{exp_key}' gradient_defaults configured for '{cfg.name}'"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
        seed = ic_cfg.get("seed", 0)
        fd_cfg = run.get("fd", {})
        eps_values = fd_cfg.get("eps_values", _DEFAULT_EPS)
        n_dirs = fd_cfg.get("n_dirs", 15)
        sweep_cfg = run.get("sweep", {})
        sweep_key = sweep_cfg.get("key")
        sweep_values = sweep_cfg.get("values", [])
        phys = run.get("physics", {})
        ic_subdir = ic_name if n_runs > 1 else ""
        run_ic_key = run.get("ic_key", cfg.ic_key)
        run_output_key = run.get("output_key", cfg.output_key)

        if not sweep_key or not sweep_values:
            raise NotImplementedError(
                f"'{exp_key}' requires sweep.key and sweep.values in gradient_defaults "
                f"(not configured for '{cfg.name}')"
            )

        # ic_sweep=True: regenerate IC per sweep value (e.g., sigma sweep where IC depends on sigma).
        # ic_sweep=False (default): generate IC once with base phys params.
        ic_sweep_flag = sweep_cfg.get("ic_sweep", False)

        if ic_sweep_flag:
            # Pre-compute IC and FD directions for each sweep value.
            ic_per_val: dict = {}
            dirs_per_val: dict = {}
            for _val in sweep_values:
                _ic_v = cfg.make_ic[ic_name](
                    L=cfg.domain_extent, seed=seed, **{**phys, sweep_key: _val}
                )
                _keys_v = jax.random.split(jax.random.PRNGKey(seed), n_dirs)
                ic_per_val[_val] = _ic_v
                dirs_per_val[_val] = [
                    _random_direction(_ic_v.shape, _k) for _k in _keys_v
                ]
            # Use a representative IC (first sweep value) for shape; dirs are per-val.
            ic = ic_per_val[sweep_values[0]]
            dirs = dirs_per_val[sweep_values[0]]
        else:
            ic = cfg.make_ic[ic_name](L=cfg.domain_extent, seed=seed, **phys)
            keys = jax.random.split(jax.random.PRNGKey(seed), n_dirs)
            dirs = [_random_direction(ic.shape, k) for k in keys]
            ic_per_val = None
            dirs_per_val = None

        # Gate on the experiment-specific exclusion so source-experiment
        # exclusions (e.g. "gradient/source_width_sweep") take precedence over
        # the suite-level "gradient" key.
        diff_solvers = active_differentiable_solvers(cfg, "gradient", exp_key)
        results: dict = {}
        grad_snaps: dict = {}  # name → {val: grad array}
        gpu_ids = overrides.get("gpu_ids")
        _wall_times: dict[str, float] = {}

        def _param_work(
            name: str,
            t,
            _run_ic_key=run_ic_key,
            _run_output_key=run_output_key,
            _ic_per_val=ic_per_val,
            _dirs_per_val=dirs_per_val,
        ) -> None:
            color = cfg.solvers[name].color
            t0 = time.perf_counter()
            try:
                solver_results: dict = {}
                solver_grads: dict = {}
                for val in sweep_values:
                    # Use per-val IC/dirs if ic_sweep=True, else shared IC/dirs.
                    _ic = _ic_per_val[val] if _ic_per_val is not None else ic
                    _dirs = _dirs_per_val[val] if _dirs_per_val is not None else dirs
                    entry, g = _eps_sweep(
                        t,
                        cfg,
                        name,
                        _ic,
                        _dirs,
                        eps_values,
                        {**phys, sweep_key: val, "domain_extent": cfg.domain_extent},
                        ic_key=_run_ic_key,
                        output_key=_run_output_key,
                    )
                    solver_results[val] = entry
                    solver_grads[val] = np.array(g)
                results[name] = solver_results
                grad_snaps[name] = solver_grads
                elapsed = time.perf_counter() - t0
                _wall_times[name] = elapsed
                console.print(f"  [{color}]{name}[/] done in {elapsed:.1f}s")
            except Exception as exc:
                console.print(
                    f"  [{color}]{name}[/] [yellow]SKIP (VJP failed: {exc})[/]"
                )

        run_with_gpu_pool(diff_solvers, tags, _param_work, gpu_ids=gpu_ids)

        out_dir = experiment_dir(
            results_dir(),
            cfg.name,
            _SUITE,
            f"{exp_key}/{ic_subdir}" if ic_subdir else exp_key,
            suffix="_debug" if overrides.get("debug") else "",
        )

        # Save gradient field snapshots per sweep value (needed by plot_horizon_sweep)
        if grad_snaps:
            solver_names = list(grad_snaps.keys())
            # Build per_solver_arrays: suffix = str index of sweep value
            per_solver: dict[str, dict[str, np.ndarray]] = {}
            for sname in solver_names:
                per_solver[sname] = {
                    str(k): grad_snaps[sname][v]
                    for k, v in enumerate(sweep_values)
                    if v in grad_snaps[sname]
                }
            save_field_snapshots_npz(
                out_dir,
                solver_names,
                per_solver,
                shared_arrays={
                    "ic": np.asarray(ic),
                    "horizons": np.array(
                        [float(v) * float(phys.get("dt", 1.0)) for v in sweep_values]
                        if sweep_key == "steps"
                        else [float(v) for v in sweep_values]
                    ),
                },
            )

        result = {"by_solver": results, "sweep_key": sweep_key, "params": run}
        save_experiment(
            result,
            out_dir,
            cfg=cfg,
            harness_fn=_run_generic_param_sweep,
            wall_time_s=_wall_times,
        )
        if n_runs > 1:
            all_results[ic_name] = result
        else:
            all_results = result

    return all_results


def run_param_sweep(cfg: ProblemConfig, tags: dict[str, str], **overrides) -> dict:
    """Gradient norm and per-solver FD ε-sweep vs one physics parameter.

    Uses sweep.key / sweep.values from gradient_defaults["param_sweep"] so any
    problem-specific parameter (nu, kT, sigma8, …) can be swept without changing
    this function.  Use sweep.key="N" or sweep.key="steps" to replace the former
    resolution_sweep and horizon_sweep respectively.

    Returns:
        {"by_solver": {solver: {val: {"grad_norm",
                                      "eps_sweep": {eps: {rel_error_mean,
                                                          rel_error_std,
                                                          cosine_mean}}}}},
         "sweep_key": sweep_key}
        or {ic_name: <above>} when multiple runs are configured.
    """
    return _run_generic_param_sweep(cfg, tags, "param_sweep", **overrides)


def run_horizon_sweep(cfg: ProblemConfig, tags: dict[str, str], **overrides) -> dict:
    """Gradient quality vs rollout horizon sweep.

    Uses sweep.key / sweep.values from gradient_defaults["horizon_sweep"].
    Saves results and gradient field snapshots to results/<problem>/gradient/horizon_sweep/.

    Returns:
        {"by_solver": {solver: {val: {"grad_norm",
                                      "eps_sweep": {eps: {rel_error_mean,
                                                          rel_error_std,
                                                          cosine_mean}}}}},
         "sweep_key": sweep_key}
        or {ic_name: <above>} when multiple runs are configured.
    """
    return _run_generic_param_sweep(cfg, tags, "horizon_sweep", **overrides)


def run_horizon_sweep_limits(
    cfg: ProblemConfig, tags: dict[str, str], **overrides
) -> dict:
    """Rollout-length limit sweep: VJP only, per-step failure recording, early stopping.

    For each solver, attempts VJP at increasing step counts.  Stops at the first
    failure (OOM, timeout, container_died, nan, error) and records all subsequent
    steps as 'skipped'.  No FD check is performed — gradient quality is out of scope
    for this experiment; OOM/failure boundary is the target metric.

    Result structure::

        {"by_solver": {
            solver: {
                steps_val: {"status": "ok",
                            "grad_norm": float,
                            "wall_time_s": float,
                            "vram_peak_mib": float | None,   # peak GPU VRAM during VJP
                            "ram_peak_mib": float | None}    # peak container RAM during VJP
                         | {"status": "failed",
                            "failure_type": "OOM"|"timeout"|"container_died"|"nan"|"error",
                            "error": str,
                            "wall_time_s": float,
                            "vram_peak_mib": float | None,   # last-known VRAM before failure
                            "ram_peak_mib": float | None}
                         | {"status": "skipped", "reason": str}
            }
        }, "sweep_key": "steps", "params": run}

    Intended to be run with one GPU per solver so OOM reflects a single GPU budget::

        mosaic gradient ns-3d-grid --experiments horizon_sweep_limits --gpu-ids 0 1 2 3
    """
    exp_key = "horizon_sweep_limits"
    runs = cfg.gradient_defaults.get(exp_key, [])
    if not runs:
        raise NotImplementedError(
            f"No '{exp_key}' gradient_defaults configured for '{cfg.name}'"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
        seed = ic_cfg.get("seed", 0)
        sweep_cfg = run.get("sweep", {})
        sweep_key = sweep_cfg.get("key")
        sweep_values = sweep_cfg.get("values", [])
        phys = run.get("physics", {})
        ic_subdir = ic_name if n_runs > 1 else ""
        run_ic_key = run.get("ic_key", cfg.ic_key)
        run_output_key = run.get("output_key", cfg.output_key)
        gpu_ids = overrides.get("gpu_ids")

        if gpu_ids is None:
            console.print(
                "  [yellow]WARN[/] horizon_sweep_limits: gpu_ids not set — solvers share all "
                "GPUs. Pass --gpu-ids 0 1 2 3 for isolated per-GPU OOM measurements."
            )

        if not sweep_key or not sweep_values:
            raise NotImplementedError(
                f"'{exp_key}' requires sweep.key and sweep.values in gradient_defaults"
            )

        ic = cfg.make_ic[ic_name](L=cfg.domain_extent, seed=seed, **phys)

        diff_solvers = active_differentiable_solvers(cfg, "gradient", exp_key)
        results: dict = {}
        grad_snaps: dict = {}
        _wall_times: dict[str, float] = {}

        def _limits_work(
            name: str,
            t,
            _run_ic_key=run_ic_key,
            _run_output_key=run_output_key,
        ) -> None:
            color = cfg.solvers[name].color
            t0 = time.perf_counter()

            # GPU ID is set per-thread by run_with_gpu_pool; container ID is
            # extracted from the Tesseract object. Both may be None (serial
            # mode or unknown SDK layout), in which case that metric is skipped.
            _gpu_id = current_worker_context().gpu_id
            _cid = container_id_from_tesseract(t)

            solver_results: dict = {}
            solver_grads: dict = {}
            failed = False
            fail_reason = ""

            # Warmup VJP at smallest step count to prime JIT/kernel compilation
            # (warp_ns Warp kernels, ins_jl Julia/Zygote JIT) before timing starts.
            try:
                _wu_inputs = cfg.make_inputs(
                    name,
                    ic,
                    **{
                        **phys,
                        sweep_key: sweep_values[0],
                        "domain_extent": cfg.domain_extent,
                    },
                )
                _vjp_grad(t, _wu_inputs, _run_output_key, _run_ic_key)
                console.print(f"  [{color}]{name}[/] warmup ok")
            except Exception as _wex:
                console.print(
                    f"  [{color}]{name}[/] warmup skipped ({type(_wex).__name__})"
                )

            for val in sweep_values:
                if failed:
                    solver_results[val] = {"status": "skipped", "reason": fail_reason}
                    continue
                t_step = time.perf_counter()
                with MemoryPoller(_gpu_id, _cid) as poller:
                    g, exc = _safe_vjp_at(
                        t,
                        cfg,
                        name,
                        ic,
                        phys,
                        sweep_key,
                        val,
                        _run_output_key,
                        _run_ic_key,
                    )
                mem = poller.summary
                step_wall = time.perf_counter() - t_step
                _vram_str = (
                    f" vram={mem['vram_peak_mib']:.0f}MiB"
                    if mem.get("vram_peak_mib") is not None
                    else ""
                )
                if exc is None:
                    g_np = np.array(g).ravel()
                    grad_norm = float(jnp.linalg.norm(g))
                    solver_results[val] = {
                        "status": "ok",
                        "grad_norm": grad_norm,
                        "grad_mean": float(g_np.mean()),
                        "grad_std": float(g_np.std()),
                        "grad_min": float(g_np.min()),
                        "grad_max": float(g_np.max()),
                        "wall_time_s": step_wall,
                        **mem,
                    }
                    solver_grads[val] = np.array(g)
                    _ram_str = (
                        f" ram={mem['ram_peak_mib']:.0f}MiB"
                        if mem.get("ram_peak_mib") is not None
                        else ""
                    )
                    console.print(
                        f"  [{color}]{name}[/] {sweep_key}={val} ok "
                        f"grad_norm={grad_norm:.3g}{_vram_str}{_ram_str} ({step_wall:.1f}s)"
                    )
                else:
                    exc_name = type(exc).__name__
                    failure_type = _classify_failure(exc_name, str(exc))
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
                        f"({failure_type}){_vram_str}: {err_short[:80]} ({step_wall:.1f}s)"
                    )

            results[name] = solver_results
            if solver_grads:
                grad_snaps[name] = solver_grads
            elapsed = time.perf_counter() - t0
            _wall_times[name] = elapsed
            console.print(f"  [{color}]{name}[/] done in {elapsed:.1f}s")

        run_with_gpu_pool(diff_solvers, tags, _limits_work, gpu_ids=gpu_ids)

        out_dir = experiment_dir(
            results_dir(),
            cfg.name,
            _SUITE,
            f"{exp_key}/{ic_subdir}" if ic_subdir else exp_key,
            suffix="_debug" if overrides.get("debug") else "",
        )

        if grad_snaps:
            solver_names = list(grad_snaps.keys())
            per_solver: dict[str, dict[str, np.ndarray]] = {}
            for sname in solver_names:
                per_solver[sname] = {
                    str(k): grad_snaps[sname][v]
                    for k, v in enumerate(sweep_values)
                    if v in grad_snaps[sname]
                }
            save_field_snapshots_npz(
                out_dir,
                solver_names,
                per_solver,
                shared_arrays={
                    "ic": np.asarray(ic),
                    "horizons": np.array(
                        [float(v) * float(phys.get("dt", 1.0)) for v in sweep_values]
                        if sweep_key == "steps"
                        else [float(v) for v in sweep_values]
                    ),
                },
            )

        result = {"by_solver": results, "sweep_key": sweep_key, "params": run}
        save_experiment(
            result,
            out_dir,
            cfg=cfg,
            harness_fn=run_horizon_sweep_limits,
            wall_time_s=_wall_times,
        )
        if n_runs > 1:
            all_results[ic_name] = result
        else:
            all_results = result

    return all_results


# ── Jacobian SVD ─────────────────────────────────────────────────────────────


def run_jacobian_svd(
    cfg: ProblemConfig,
    tags: dict[str, str],
    *,
    _exp_key: str = "jacobian_svd",
    **overrides,
) -> dict:
    """Singular-value spectrum of the stacked per-solver gradient matrix.

    Computes VJP gradients ∂L/∂IC for all differentiable solvers, stacks them
    into G (n_solvers × D), runs SVD, and records:

        - singular_values: σᵢ/σ₁ (normalised spectrum, length = n_solvers)
        - condition_number: σ₁/σₙ
        - effective_rank: (Σσ)²/(Σσ²)  — participation ratio
        - explained_variance: cumulative σᵢ² / Σσⱼ² per mode
        - cross_cosine: n×n pairwise cosine similarity between gradient vectors
          (inter-solver gradient agreement)
        - grad_norms: {solver: float}
        - landscape: 1-D loss slice along the top singular direction
          (set n_alphas=0 to skip)

    NPZ: solver_names, singular_values (raw), singular_vectors (top k × D), ic,
         grad_j for each solver j.

    Returns:
        {"solver_names": [...], "singular_values": [...], ...}
        or {ic_name: <above>} when multiple runs are configured.
    """
    runs = cfg.gradient_defaults.get(_exp_key, [])
    if not runs:
        raise NotImplementedError(
            f"No '{_exp_key}' gradient_defaults configured for '{cfg.name}'"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
        seed = ic_cfg.get("seed", 0)
        jacobian_cfg = run.get("jacobian", {})
        n_alphas = jacobian_cfg.get("n_alphas", 41)
        alpha_range = jacobian_cfg.get("alpha_range", 0.3)
        k_svd = jacobian_cfg.get("k_svd", None)
        phys = run.get("physics", {})
        ic_subdir = ic_name if n_runs > 1 else ""

        ic = cfg.make_ic[ic_name](seed=seed, L=cfg.domain_extent, **phys)

        diff_solvers = active_differentiable_solvers(cfg, "gradient", _exp_key)
        jacobians: dict = {}  # name → (D_out, D_in) ndarray
        grad_snaps: dict = {}  # name → (D_in,) ndarray  [for field plots]
        base_inputs_snap: dict = {}
        gpu_ids = overrides.get("gpu_ids")
        _wall_times: dict[str, float] = {}

        # ── Pass 1: full Jacobian via jacrev ──────────────────────────────────
        # jax.jacrev computes ∂output[i]/∂ic[j] for all (i,j), giving the full
        # (D_out, D_in) Jacobian with D_out VJP calls — tractable for small N.
        def _svd_work(name: str, t) -> None:
            color = cfg.solvers[name].color
            t0 = time.perf_counter()
            try:
                base_inputs = cfg.make_inputs(
                    name, ic, domain_extent=cfg.domain_extent, **phys
                )
                base_ic = jnp.array(base_inputs[cfg.ic_key])

                def fwd(ic_arr):
                    return apply_tesseract(t, {**base_inputs, cfg.ic_key: ic_arr})[
                        cfg.output_key
                    ]

                # Full Jacobian via sequential VJP (one call per output element).
                # jax.jacrev uses vmap which tesseract does not support, so we loop.
                out, vjp_fn = jax.vjp(fwd, base_ic)
                out_arr = np.array(out)
                D_out = int(out_arr.size)
                J_rows = []
                _log_every = max(1, D_out // 8)
                console.print(f"  [{color}]{name}[/] Jacobian {D_out} rows starting")
                for i in range(D_out):
                    e_i = jnp.zeros(D_out).at[i].set(1.0)
                    (row,) = vjp_fn(e_i.reshape(out_arr.shape))
                    J_rows.append(np.array(row).ravel())
                    if (i + 1) % _log_every == 0:
                        console.print(
                            f"  [{color}]{name}[/] Jacobian {i + 1}/{D_out} rows done"
                        )
                J_mat = np.stack(J_rows)  # (D_out, D_in)
                jacobians[name] = J_mat

                # Gradient of sum(output²) = J^T @ (2*output): for field plots
                grad_snaps[name] = (J_mat.T @ (2.0 * out_arr.ravel())).reshape(
                    base_ic.shape
                )

                base_inputs_snap[name] = (dict(base_inputs), np.array(base_ic))
                elapsed = time.perf_counter() - t0
                _wall_times[name] = elapsed
                console.print(
                    f"  [{color}]{name}[/] J {J_mat.shape} done in {elapsed:.1f}s"
                )
            except Exception as exc:
                console.print(
                    f"  [{color}]{name}[/] [yellow]SKIP (VJP failed: {exc})[/]"
                )

        run_with_gpu_pool(diff_solvers, tags, _svd_work, gpu_ids=gpu_ids)

        if not jacobians:
            raise RuntimeError("No differentiable solvers returned Jacobians")

        # ── Merge with existing Jacobians from NPZ (Option A) ─────────────────
        # If a prior partial run saved Jacobians to the NPZ, load them so that
        # aggregate statistics (combined SVD, cross_cosine) use the full solver
        # set rather than just the current subset.  The NPZ stores each solver's
        # full Jacobian matrix under the positional key ``jac_j`` (see save
        # below).  Per-solver entries already computed this run take precedence.
        out_dir_for_merge = experiment_dir(
            results_dir(),
            cfg.name,
            _SUITE,
            f"{_exp_key}/{ic_subdir}" if ic_subdir else _exp_key,
            suffix="_debug" if overrides.get("debug") else "",
        )
        _npz = try_load_npz(out_dir_for_merge / "jacobian_svd.npz")
        _old_names = [str(n) for n in _npz.get("solver_names", np.array([]))]
        for _j, _old_name in enumerate(_old_names):
            _jac_key = f"jac_{_j}"
            _grad_key = f"grad_{_j}"
            if _old_name not in jacobians and _jac_key in _npz:
                _J = _npz[_jac_key]
                jacobians[_old_name] = _J
                if _grad_key in _npz:
                    grad_snaps[_old_name] = _npz[_grad_key]
                elif _J.ndim == 2:
                    # Derive 1-D gradient from J: grad of sum(out²) = J^T @ (2*out).
                    # We cannot recover ``out`` here, so fall back to Frobenius
                    # row-norm as a proxy (correct sign not guaranteed; only
                    # used for grad_norms reporting, not for SVD).
                    grad_snaps[_old_name] = np.linalg.norm(_J, axis=0)
                console.print(
                    f"  [dim]jacobian_svd: merged existing Jacobian for {_old_name} from NPZ[/]"
                )

        solver_names = list(jacobians.keys())
        G_stack = np.vstack([jacobians[n] for n in solver_names])  # (n_s*D_out, D_in)

        # ── SVD ───────────────────────────────────────────────────────────────
        _U, S, Vt = np.linalg.svd(
            G_stack, full_matrices=False
        )  # S: (≤D_in,), Vt: (≤D_in, D_in)

        k_report = k_svd if k_svd is not None else len(S)
        S = S[:k_report]
        Vt = Vt[:k_report]

        S_norm = (S / (S[0] + 1e-30)).tolist()
        cond = float(S[0] / (S[-1] + 1e-30))
        eff_rank = float(S.sum() ** 2 / ((S**2).sum() + 1e-30))
        expl_var = (np.cumsum(S**2) / (float((S**2).sum()) + 1e-30)).tolist()

        # Per-solver singular value spectra (SVD of each solver's Jacobian separately).
        # This reveals spectral structure per solver family:
        # projection methods tend to have a steep singular-value drop (low effective rank)
        # while LBM Jacobians have a flatter spectrum (higher effective rank).
        per_solver_spectra: dict[str, list[float]] = {}
        per_solver_cond: dict[str, float] = {}
        per_solver_eff_rank: dict[str, float] = {}
        per_solver_grad_norm: dict[str, float] = {}
        for name in solver_names:
            _, Si, _ = np.linalg.svd(jacobians[name], full_matrices=False)
            k_i = k_svd if k_svd is not None else len(Si)
            Si = Si[:k_i]
            per_solver_spectra[name] = (Si / (Si[0] + 1e-30)).tolist()
            per_solver_cond[name] = float(Si[0] / (Si[-1] + 1e-30))
            per_solver_eff_rank[name] = float(Si.sum() ** 2 / ((Si**2).sum() + 1e-30))
            per_solver_grad_norm[name] = float(Si[0])

        # Cross-cosine: Frobenius inner product between per-solver Jacobians (normalised)
        J_flat = {n: jacobians[n].ravel() for n in solver_names}
        J_norms = {n: float(np.linalg.norm(v) + 1e-30) for n, v in J_flat.items()}
        cross_cos = [
            [
                float(np.dot(J_flat[a], J_flat[b]) / (J_norms[a] * J_norms[b]))
                for b in solver_names
            ]
            for a in solver_names
        ]
        grad_norms = {n: float(np.linalg.norm(grad_snaps[n])) for n in solver_names}

        # Top singular direction — shaped like the IC
        _, base_ic_arr = base_inputs_snap[solver_names[0]]
        ic_scale = float(np.sqrt(np.mean(base_ic_arr**2) + 1e-30))
        d_top = Vt[0].reshape(base_ic_arr.shape).astype(np.float32)

        # ── Pass 2: loss landscape along d_top (skip when n_alphas == 0) ──────
        alphas = (
            np.linspace(-alpha_range, alpha_range, n_alphas).tolist()
            if n_alphas > 0
            else []
        )
        landscape_by_solver: dict = {}

        if alphas:
            d_top_jax = jnp.array(d_top)

            def _landscape_work(name: str, t) -> None:
                color = cfg.solvers[name].color
                base_inputs, base_ic_solver = base_inputs_snap[name]
                base_ic_jax = jnp.array(base_ic_solver)
                losses = [
                    float(
                        jnp.sum(
                            apply_tesseract(
                                t,
                                {
                                    **base_inputs,
                                    cfg.ic_key: base_ic_jax
                                    + float(a) * ic_scale * d_top_jax,
                                },
                            )[cfg.output_key]
                            ** 2
                        )
                    )
                    for a in alphas
                ]
                landscape_by_solver[name] = losses
                console.print(f"  [{color}]{name}[/] landscape done")

            # Only run landscape for solvers that succeeded in pass 1 (have Jacobians).
            # Solvers that failed pass 1 (e.g. pict VJP error) are not in
            # base_inputs_snap and would cause a KeyError here.
            landscape_solvers = [n for n in diff_solvers if n in base_inputs_snap]
            run_with_gpu_pool(landscape_solvers, tags, _landscape_work, gpu_ids=gpu_ids)

        # ── Save NPZ ──────────────────────────────────────────────────────────
        out_dir = out_dir_for_merge  # already computed above for merge lookup
        # Build per-solver payload: grad_{j} (1-D gradient) and jac_{j} (full
        # Jacobian matrix).  The ``jac`` prefix enables future partial runs to
        # reload existing Jacobians and recompute aggregate statistics without
        # re-running all solvers (see merge block above).
        _per_solver_npz: dict[str, dict[str, np.ndarray]] = {}
        for _sname in solver_names:
            _per_solver_npz[_sname] = {
                "grad:": np.asarray(grad_snaps[_sname]),
                "jac:": np.asarray(jacobians[_sname]),
            }
        save_field_snapshots_npz(
            out_dir,
            solver_names,
            _per_solver_npz,
            shared_arrays={
                "singular_values": np.asarray(S),
                "singular_vectors": np.asarray(Vt),
                "ic": np.array(ic),
            },
            filename="jacobian_svd.npz",
            prefixes=("grad", "jac"),
        )

        result = {
            "solver_names": solver_names,
            "singular_values": S_norm,
            "singular_values_raw": S.tolist(),
            "condition_number": cond,
            "effective_rank": eff_rank,
            "explained_variance": expl_var,
            "per_solver_spectra": per_solver_spectra,
            "per_solver_cond": per_solver_cond,
            "per_solver_eff_rank": per_solver_eff_rank,
            "per_solver_grad_norm": per_solver_grad_norm,
            "cross_cosine": cross_cos,
            "grad_norms": grad_norms,
            "landscape": {"alphas": alphas, "by_solver": landscape_by_solver},
            "params": run,
        }
        save_experiment(
            result,
            out_dir,
            cfg=cfg,
            harness_fn=run_jacobian_svd,
            wall_time_s=_wall_times,
        )
        if n_runs > 1:
            all_results[ic_name] = result
        else:
            all_results = result

    return all_results


# ── run_all ───────────────────────────────────────────────────────────────────


def _jacobian_svd_variant(exp_key: str):
    def _run(cfg, tags, **kw):
        return run_jacobian_svd(cfg, tags, _exp_key=exp_key, **kw)

    _run.__name__ = f"run_{exp_key}"
    return _run


def _fd_check_variant(exp_key: str):
    def _run(cfg, tags, **kw):
        return run_fd_check(cfg, tags, _exp_key=exp_key, **kw)

    _run.__name__ = f"run_{exp_key}"
    return _run


_EXPERIMENTS = {
    "fd_check": run_fd_check,
    "source_fd_check": _fd_check_variant("source_fd_check"),
    "param_sweep": run_param_sweep,
    "source_width_sweep": lambda cfg, tags, **kw: _run_generic_param_sweep(
        cfg, tags, "source_width_sweep", **kw
    ),
    "horizon_sweep": run_horizon_sweep,
    "horizon_sweep_limits": run_horizon_sweep_limits,
    "jacobian_svd": run_jacobian_svd,
    "jacobian_svd_steps20": _jacobian_svd_variant("jacobian_svd_steps20"),
    "jacobian_svd_steps40": _jacobian_svd_variant("jacobian_svd_steps40"),
    "jacobian_svd_nu01": _jacobian_svd_variant("jacobian_svd_nu01"),
    # stokes-specific jacobian_svd variants (registered by stokes_grid problem config)
    "jacobian_svd_mu001": _jacobian_svd_variant("jacobian_svd_mu001"),
    "jacobian_svd_n16_mu001": _jacobian_svd_variant("jacobian_svd_n16_mu001"),
    "jacobian_svd_n32_mu001": _jacobian_svd_variant("jacobian_svd_n32_mu001"),
    "jacobian_svd_n32_mu01": _jacobian_svd_variant("jacobian_svd_n32_mu01"),
    "jacobian_svd_n32_mu03": _jacobian_svd_variant("jacobian_svd_n32_mu03"),
    "jacobian_svd_n32_mu05": _jacobian_svd_variant("jacobian_svd_n32_mu05"),
    "jacobian_svd_n32_mu07": _jacobian_svd_variant("jacobian_svd_n32_mu07"),
    "jacobian_svd_n32_mu09": _jacobian_svd_variant("jacobian_svd_n32_mu09"),
}


def _plot_fns() -> dict:
    from mosaic.benchmarks.plots.gradient import (
        plot_fd_check,
        plot_horizon_sweep,
        plot_jacobian_svd,
        plot_param_sweep,
    )

    def _jsvd_plot(exp_key):
        return lambda cfg, **kw: plot_jacobian_svd(cfg, exp_key=exp_key, **kw)

    return {
        "fd_check": plot_fd_check,
        "param_sweep": plot_param_sweep,
        "horizon_sweep": plot_horizon_sweep,
        "jacobian_svd": plot_jacobian_svd,
        "jacobian_svd_steps20": _jsvd_plot("jacobian_svd_steps20"),
        "jacobian_svd_steps40": _jsvd_plot("jacobian_svd_steps40"),
        "jacobian_svd_nu01": _jsvd_plot("jacobian_svd_nu01"),
        "source_fd_check": lambda cfg, **kw: plot_fd_check(
            cfg, exp_key="source_fd_check", **kw
        ),
        "source_width_sweep": lambda cfg, **kw: plot_param_sweep(
            cfg, exp_key="source_width_sweep", **kw
        ),
        "jacobian_svd_mu001": _jsvd_plot("jacobian_svd_mu001"),
        "jacobian_svd_n16_mu001": _jsvd_plot("jacobian_svd_n16_mu001"),
        "jacobian_svd_n32_mu001": _jsvd_plot("jacobian_svd_n32_mu001"),
        "jacobian_svd_n32_mu01": _jsvd_plot("jacobian_svd_n32_mu01"),
        "jacobian_svd_n32_mu03": _jsvd_plot("jacobian_svd_n32_mu03"),
        "jacobian_svd_n32_mu05": _jsvd_plot("jacobian_svd_n32_mu05"),
        "jacobian_svd_n32_mu07": _jsvd_plot("jacobian_svd_n32_mu07"),
        "jacobian_svd_n32_mu09": _jsvd_plot("jacobian_svd_n32_mu09"),
    }


def run_all(
    cfg: ProblemConfig,
    tags: dict[str, str],
    experiments: list[str] | None = None,
    plots: bool = True,
) -> dict[str, dict]:
    """Run gradient experiments and optionally generate plots."""
    from mosaic.benchmarks.core.runner import run_suite

    return run_suite(
        cfg,
        tags,
        _EXPERIMENTS,
        to_run=experiments,
        plots=plots,
        plot_fns=_plot_fns() if plots else None,
        suite_name=_SUITE,
    )
