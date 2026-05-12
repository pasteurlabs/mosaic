"""Forward suite: agreement, physical laws.

Each run_* function:
  - Opens each solver once (outer loop) and runs all conditions through it
  - Saves results as JSON + CSV under results/{problem_name}/forward/
  - Returns the results dict

Run from the terminal:
    mosaic run <problem> forward [--experiments EXPR] [--plots-only]
"""

from __future__ import annotations

import contextlib
import inspect
import threading

import jax.numpy as jnp
import numpy as np  # kept for string arrays (JAX doesn't support these)

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import (
    experiment_dir,
    results_dir,
    save_experiment,
    save_field_snapshots_npz,
    try_load_npz,
)
from mosaic.benchmarks.core.runner import (
    get_last_apply_error,
    safe_apply,
    safe_apply_with_extras,
    solver_sweep,
)
from mosaic.benchmarks.core.utils import (
    extract_runs,
    is_valid,
    iter_runs,
    trimmed_mean,
)

_SUITE = "forward"


# ── Agreement / baseline ──────────────────────────────────────────────────────


def run_agreement(
    cfg: Problem,
    tags: dict[str, str],
    *,
    make_ic,
    make_inputs,
    error_fn,
    output_key: str,
    domain_extent: float,
    agreement_transform=None,
    agreement_xaxis=None,
    analytic=None,
    runs=None,
    exp_key: str = "agreement",
    **overrides,
) -> dict:
    """Run all solvers across a sweep of one physics parameter; compute trimmed-mean
    consensus and per-solver error.

    Problem-semantics state is passed explicitly:
        make_ic              — dict[ic_name → IcSpec | Callable]
        make_inputs          — (solver_name, ic, **physics) → dict
        error_fn             — (pred, ref) → float
        output_key           — name of the solver-output array to compare
        domain_extent        — physical domain length (passed to make_ic / make_inputs)
        agreement_transform  — optional (arr, **physics) → arr applied before comparison
        agreement_xaxis      — optional (**physics) → 1-D x-axis array stored alongside the IC
        analytic             — optional (ic, t, L, **physics) → arr reference solution

    ``cfg`` retains its role as the runtime *registry*: solver list (already
    filtered by CLI) and problem name (for output paths). It is never read
    for problem-semantics fields.

    ``runs`` is the per-experiment run payload (list of run dicts, or a
    wrapper dict with a ``"runs"`` key). ``exp_key`` is the full experiment
    label used for output-dir naming and exclusion lookup (e.g.
    ``"agreement/tgv"``, ``"baseline"``).

    Each run dict must contain:
        ic=dict(name, seed)
        physics=dict(N, dt, steps, ...)
        sweep=dict(key, values)
        fine=dict(solvers, dt, steps)   [optional]

    Returns:
        {"by_param": {val: {solver: {"error": float, "valid": bool}}}, "spread": {val: float}}
        or {ic_name: <above>} when multiple IC runs are configured.
    """
    if not runs:
        raise NotImplementedError(
            f"run_agreement requires runs= payload for {exp_key!r} "
            f"(not configured for '{cfg.name}')"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_name, result = _run_single_agreement(
            cfg,
            tags,
            run,
            exp_key=exp_key,
            n_runs=n_runs,
            overrides=overrides,
            make_ic=make_ic,
            make_inputs=make_inputs,
            error_fn=error_fn,
            output_key=output_key,
            domain_extent=domain_extent,
            agreement_transform=agreement_transform,
            agreement_xaxis=agreement_xaxis,
            analytic=analytic,
        )
        if n_runs > 1:
            all_results[ic_name] = result
        else:
            all_results = result

    return all_results


def _run_single_agreement(
    cfg: Problem,
    tags: dict[str, str],
    run: dict,
    *,
    exp_key: str,
    n_runs: int,
    overrides: dict,
    make_ic,
    make_inputs,
    error_fn,
    output_key: str,
    domain_extent: float,
    agreement_transform,
    agreement_xaxis,
    analytic,
) -> tuple[str, dict]:
    """Body of one ``run`` iteration in :func:`run_agreement`.

    Returns ``(ic_name, result_dict)``. Receives problem-semantics state via
    explicit kwargs (see :func:`run_agreement` for descriptions).
    """
    ic_cfg = run.get("ic", {})
    ic_name = ic_cfg.get("name", next(iter(make_ic)))
    seed = ic_cfg.get("seed", 0)
    sweep = run.get("sweep", {})
    sweep_key = sweep.get("key")
    sweep_values = sweep.get("values", [])
    if not sweep_key or not sweep_values:
        raise NotImplementedError(
            f"run_agreement requires sweep.key and sweep.values "
            f"in runs payload (not configured for '{cfg.name}')"
        )
    fine_cfg = run.get("fine", {})
    fine_set = set(fine_cfg.get("solvers", set()))
    fine_dt = fine_cfg.get("dt")
    fine_steps = fine_cfg.get("steps")
    phys = run.get("physics", {})
    ic_subdir = ic_name if n_runs > 1 else ""

    _apply_errors: dict = {s.name: {} for s in cfg.solvers}
    _apply_errors_lock = threading.Lock()
    # Capture drag (force on obstacle) from solvers that produce it. Cylinder
    # experiments are the only ones that pass an obstacle; for the rest the
    # extras dict comes back empty and this stays as {}.
    _drags: dict = {s.name: {} for s in cfg.solvers}

    # Regenerate IC at each sweep value so N-sweeps use the correct IC shape.
    def _apply(name, t, val, _phys=phys, _ic_name=ic_name, _seed=seed):
        curr_phys = {**_phys, sweep_key: val}
        _ic = make_ic[_ic_name](L=domain_extent, seed=_seed, **curr_phys)
        _p = dict(curr_phys)
        if name in fine_set:
            if fine_dt is not None:
                _p["dt"] = fine_dt
            if fine_steps is not None:
                _p["steps"] = fine_steps
        inputs = make_inputs(name, _ic, domain_extent=domain_extent, **_p)
        result, extras, _ = safe_apply_with_extras(t, inputs, output_key, ["drag"])
        if result is None:
            err = get_last_apply_error()
            with _apply_errors_lock:
                _apply_errors[name][val] = err
        if "drag" in extras:
            _drags[name][val] = extras["drag"]
        norm = cfg.solver(name).normalize_output
        return norm(result) if (norm is not None and result is not None) else result

    raw, _wall_times = solver_sweep(
        cfg,
        tags,
        sweep_values,
        _apply,
        experiment=exp_key,
        label_fn=lambda v: f"{sweep_key}={v}",
        gpu_ids=overrides.get("gpu_ids"),
    )

    exp_subdir = f"{exp_key}/{ic_subdir}" if ic_subdir else exp_key
    out_dir = experiment_dir(
        results_dir(),
        cfg.name,
        _SUITE,
        exp_subdir,
        suffix="_debug" if overrides.get("debug") else "",
    )
    # Snapshot bookkeeping. Per-solver field arrays live in ``per_solver``
    # keyed by sweep index; consensus and IC/x_axis go into ``shared_arrays``.
    # save_field_snapshots_npz (flat_keys=True) handles the lock-and-merge:
    # solvers not rerun this invocation keep their entries in fields.npz.
    per_solver: dict[str, dict[str, np.ndarray]] = {}
    shared_arrays: dict[str, np.ndarray] = {
        "sweep_values": np.array([float(v) for v in sweep_values]),
    }
    if agreement_transform is None and sweep_values:
        # Store IC at the first sweep value so plots have a reference waveform.
        _ic0 = make_ic[ic_name](
            L=domain_extent, seed=seed, **{**phys, sweep_key: sweep_values[0]}
        )
        shared_arrays["ic"] = np.asarray(_ic0)
    if agreement_xaxis is not None:
        shared_arrays["x_axis"] = np.asarray(agreement_xaxis(**phys))

    # Pre-load cached arrays for solvers NOT being re-run this invocation
    # so consensus can still be formed from the union. cfg.solvers is
    # filtered by cli.py to only the requested solvers, so we parse solver
    # names from the snapshot keys rather than cfg.solvers.
    _cached_for_val = _load_cached_fields_for_vals(
        out_dir / "fields.npz", sweep_values, tags
    )

    # Pre-inspect analytic signature once (avoids repeated calls inside loop).
    _analytic_sig_params: set[str] = set()
    if analytic is not None:
        _analytic_sig_params = set(inspect.signature(analytic).parameters)

    by_param: dict = {}
    ctx = {
        "cfg": cfg,
        "run": run,
        "raw": raw,
        "phys": phys,
        "sweep_key": sweep_key,
        "ic_name": ic_name,
        "seed": seed,
        "transform": agreement_transform,
        "cached_for_val": _cached_for_val,
        "analytic_sig_params": _analytic_sig_params,
        "apply_errors": _apply_errors,
        "drags": _drags,
        "per_solver": per_solver,
        "shared_arrays": shared_arrays,
        "by_param": by_param,
        "make_ic": make_ic,
        "make_inputs": make_inputs,
        "error_fn": error_fn,
        "domain_extent": domain_extent,
        "analytic": analytic,
        "solvers_for_inputs": cfg.solvers,
    }
    reference_label = "consensus"  # updated per-iteration when analytic is used
    for i, val in enumerate(sweep_values):
        reference_label = _process_sweep_value(ctx, i, val, reference_label)
    # Atomic merge-save: partial reruns preserve un-rerun solvers' arrays
    # via save_field_snapshots_npz's lock-and-merge logic. ``prefixes``
    # lists the shared (non-per-solver) key prefixes so they don't get
    # parsed as ``{solver_name}_{suffix}`` on read.
    save_field_snapshots_npz(
        out_dir,
        solver_names=[s.name for s in cfg.solvers],
        per_solver_arrays=per_solver,
        shared_arrays=shared_arrays,
        filename="fields.npz",
        prefixes=("sweep_values", "ic", "consensus_", "x_axis"),
        flat_keys=True,
    )

    spread = {
        val: float(
            jnp.std(
                jnp.array(
                    [
                        r["error"]
                        for r in s.values()
                        if r["valid"] and r["error"] is not None
                    ]
                )
            )
        )
        for val, s in by_param.items()
    }

    result = {
        "by_param": by_param,
        "spread": spread,
        "sweep_key": sweep_key,
        "reference_label": reference_label,
        "params": run,
    }
    save_experiment(
        result,
        out_dir,
        csv_rows=[
            {
                "solver": n,
                sweep_key: val,
                "error": by_param[val][n]["error"],
                "valid": by_param[val][n]["valid"],
            }
            for val in sweep_values
            for n in (s.name for s in cfg.solvers)
        ],
        cfg=cfg,
        harness_fn=run_agreement,
        wall_time_s=_wall_times,
    )
    return ic_name, result


def _load_cached_fields_for_vals(snap_path, sweep_values, tags: dict[str, str]) -> dict:
    """Read fields.npz (if present) and return per-sweep-value cached arrays.

    Solver names appearing in ``tags`` are skipped — those are being re-run
    this invocation and will overwrite cached entries anyway. Meta keys
    (``sweep_values``, ``solver_names``, ``ic``, ``consensus``, ``x_axis``)
    are filtered out.
    """
    snap = try_load_npz(snap_path)
    if not snap:
        return {}
    meta_prefixes = ("sweep_values", "solver_names", "ic", "consensus", "x_axis")
    cached: dict = {}
    for ci, cv in enumerate(sweep_values):
        cached[cv] = {}
        sfx = f"_{ci}"
        for sfile, arr in snap.items():
            if not sfile.endswith(sfx):
                continue
            if any(sfile.startswith(p) for p in meta_prefixes):
                continue
            cn = sfile[: -len(sfx)]
            if cn not in tags:
                cached[cv][cn] = arr
    return cached


def _analytic_reference(
    *,
    ic_name: str,
    seed: int,
    phys: dict,
    sweep_key: str,
    val,
    analytic_sig_params: set[str],
    make_ic,
    make_inputs,
    domain_extent: float,
    analytic,
    solver_name_for_inputs: str,
) -> np.ndarray:
    """Compute the analytic reference field for one sweep value.

    ``solver_name_for_inputs`` is the solver whose ``make_inputs`` shape is
    queried for dt/steps/domain_extent — usually the first registered solver
    in cfg.solvers; only metadata is read, not its output.
    """
    curr_phys = {**phys, sweep_key: val}
    ic_ref = make_ic[ic_name](L=domain_extent, seed=seed, **curr_phys)
    inputs_ref = make_inputs(
        solver_name_for_inputs, ic_ref, domain_extent=domain_extent, **curr_phys
    )
    t_end = float(np.asarray(inputs_ref["dt"])[0]) * int(inputs_ref["steps"])
    L = float(inputs_ref.get("domain_extent", 2 * np.pi))
    extra = {
        k: curr_phys[k]
        for k in analytic_sig_params
        if k in curr_phys and k not in ("ic", "t", "L")
    }
    return np.asarray(analytic(ic_ref, t=t_end, L=L, **extra))


def _process_sweep_value(ctx: dict, i: int, val, reference_label: str) -> str:
    """Process one sweep value: build ``comparable``, choose a reference, and
    populate ``by_param[val]`` / ``per_solver`` / ``shared_arrays`` in place.

    ``ctx`` carries the loop-invariant state assembled in :func:`_run_single_agreement`
    (cfg, run, raw, phys, sweep_key, ic_name, seed, transform, cached_for_val,
    analytic_sig_params, apply_errors, drags, per_solver, shared_arrays, by_param).
    Returns the (possibly updated) ``reference_label``.
    """
    cfg = ctx["cfg"]
    raw = ctx["raw"]
    phys = ctx["phys"]
    transform = ctx["transform"]
    apply_errors = ctx["apply_errors"]
    drags = ctx["drags"]
    per_solver = ctx["per_solver"]
    shared_arrays = ctx["shared_arrays"]
    by_param = ctx["by_param"]
    error_fn = ctx["error_fn"]
    analytic = ctx["analytic"]

    valid_outputs = {
        n: raw[n][val]
        for n in (s.name for s in cfg.solvers)
        if val in raw.get(n, {}) and raw[n][val] is not None
    }
    comparable = (
        {n: np.asarray(transform(arr, **phys)) for n, arr in valid_outputs.items()}
        if transform is not None
        else valid_outputs
    )
    # Augment with cached arrays for solvers not re-run in this invocation.
    # Snapshot stores already-transformed arrays, matching comparable's format.
    for cn, arr in ctx["cached_for_val"].get(val, {}).items():
        if cn not in comparable:
            comparable[cn] = arr
    if len(comparable) < 2:
        by_param[val] = {
            n: {"error": apply_errors.get(n, {}).get(val), "valid": False}
            for n in (s.name for s in cfg.solvers)
        }
        # Still store whatever output we have so scientists can inspect
        # the lone solver's field even when consensus cannot be formed.
        for n, arr in comparable.items():
            per_solver.setdefault(n, {})[str(i)] = np.asarray(arr)
        return reference_label
    if analytic is not None and "obstacle" not in ctx["run"].get("physics", {}):
        reference = _analytic_reference(
            ic_name=ctx["ic_name"],
            seed=ctx["seed"],
            phys=phys,
            sweep_key=ctx["sweep_key"],
            val=val,
            analytic_sig_params=ctx["analytic_sig_params"],
            make_ic=ctx["make_ic"],
            make_inputs=ctx["make_inputs"],
            domain_extent=ctx["domain_extent"],
            analytic=analytic,
            solver_name_for_inputs=ctx["solvers_for_inputs"][0].name,
        )
        reference_label = "analytic"
    else:
        reference = trimmed_mean(list(comparable.values()))
        reference_label = "consensus"
    shared_arrays[f"consensus_{i}"] = np.asarray(reference)
    for n, arr in comparable.items():
        per_solver.setdefault(n, {})[str(i)] = np.asarray(arr)
    by_param[val] = {
        n: (
            {
                "error": error_fn(comparable[n], reference),
                "valid": True,
                **({"drag": drags[n][val]} if val in drags.get(n, {}) else {}),
            }
            if n in comparable
            else {
                "error": apply_errors.get(n, {}).get(val),
                "valid": False,
            }
        )
        for n in (s.name for s in cfg.solvers)
    }
    return reference_label


# ── Physical laws ─────────────────────────────────────────────────────────────


def run_physical_laws(
    cfg: Problem,
    tags: dict[str, str],
    *,
    make_ic,
    make_inputs,
    error_fn,
    output_key: str,
    domain_extent: float,
    diagnostics: dict,
    analytic=None,
    runs=None,
    **overrides,
) -> dict:
    """Sweep complexity parameter, compute physical diagnostics at each value.

    Problem-semantics state is passed explicitly (see :func:`run_agreement` for
    the field semantics). ``cfg`` is only used for the runtime solver registry
    (``cfg.solvers``, ``cfg.name``, ``cfg.solver(name)`` for output
    normalization).

    ``runs`` is a list of run dicts (or wrapper dict with ``"runs"``), each with:
        ic=dict(name, seed)
        physics=dict(N, dt, steps, ...)
        sweep=dict(key, values)

    At each sweep value the IC is regenerated, all solvers are run, and
    ``diagnostics`` are computed on each output. If ``analytic`` is available,
    ``analytic_error`` is also reported.

    Returns:
        {"by_param": {val: {solver: {diag_name: value, "analytic_error": float|None}}},
         "sweep_key": str}
        or {ic_name: <above>} when multiple IC runs are configured.
    """
    if not runs:
        raise NotImplementedError(
            f"run_physical_laws requires runs= payload "
            f"(not configured for '{cfg.name}')"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(make_ic)))
        seed = ic_cfg.get("seed", 0)
        sweep = run.get("sweep", {})
        sweep_key = sweep.get("key")
        sweep_values = sweep.get("values", [])
        if not sweep_key or not sweep_values:
            raise NotImplementedError(
                f"run_physical_laws requires sweep.key and sweep.values "
                f"in runs payload (not configured for '{cfg.name}')"
            )
        phys = run.get("physics", {})
        run_name = run.get("name", ic_name)
        ic_subdir = run_name if n_runs > 1 else ""

        # Regenerate IC at each sweep value (handles N sweeps correctly).
        def _apply(name, t, val, _phys=phys, _ic_name=ic_name, _seed=seed):
            curr_phys = {**_phys, sweep_key: val}
            _ic = make_ic[_ic_name](L=domain_extent, seed=_seed, **curr_phys)
            inputs = make_inputs(name, _ic, domain_extent=domain_extent, **curr_phys)
            out = safe_apply(t, inputs, output_key)
            norm = cfg.solver(name).normalize_output
            return norm(out) if (norm is not None and out is not None) else out

        raw, _wall_times = solver_sweep(
            cfg,
            tags,
            sweep_values,
            _apply,
            experiment="physical_laws",
            label_fn=lambda v: f"{sweep_key}={v}",
            gpu_ids=overrides.get("gpu_ids"),
        )

        # Analytic signature inspection done once.
        analytic_params: set[str] = set()
        if analytic is not None:
            analytic_params = set(inspect.signature(analytic).parameters)

        by_param: dict = {}
        csv_rows: list[dict] = []
        diag_ctx = {"domain_extent": domain_extent}

        for val in sweep_values:
            curr_phys = {**phys, sweep_key: val}
            by_param[val] = {}

            # Analytic reference (regenerate IC at this val for correct shape).
            analytic_ref = None
            if analytic is not None:
                ic_ref = make_ic[ic_name](L=domain_extent, seed=seed, **curr_phys)
                _dt = curr_phys.get("dt", 1.0)
                _steps = curr_phys.get("steps", 1)
                t_end = float(_dt) * int(_steps)
                analytic_kw = {
                    k: v for k, v in curr_phys.items() if k in analytic_params
                }
                try:
                    analytic_ref = analytic(
                        ic_ref, t=t_end, L=domain_extent, **analytic_kw
                    )
                except Exception:
                    analytic_ref = None

            for name in raw:
                out = raw[name][val]
                if not is_valid(out):
                    by_param[val][name] = None
                    continue
                diag: dict = {}
                for dname, fn in diagnostics.items():
                    with contextlib.suppress(Exception):
                        r = fn(out, **diag_ctx, **curr_phys)
                        if isinstance(r, (int, float)):
                            diag[dname] = float(r)
                if analytic_ref is not None:
                    with contextlib.suppress(Exception):
                        diag["analytic_error"] = float(error_fn(out, analytic_ref))
                by_param[val][name] = diag
                for dname, dval in diag.items():
                    csv_rows.append(
                        {
                            "solver": name,
                            sweep_key: val,
                            "diagnostic": dname,
                            "value": dval,
                        }
                    )

        out_dir = experiment_dir(
            results_dir(),
            cfg.name,
            _SUITE,
            f"physical_laws/{ic_subdir}" if ic_subdir else "physical_laws",
            suffix="_debug" if overrides.get("debug") else "",
        )
        result = {"by_param": by_param, "sweep_key": sweep_key, "params": run}
        save_experiment(
            result,
            out_dir,
            csv_rows=csv_rows,
            cfg=cfg,
            harness_fn=run_physical_laws,
            wall_time_s=_wall_times,
        )
        if n_runs > 1:
            all_results[run_name] = result
        else:
            all_results = result

    return all_results
