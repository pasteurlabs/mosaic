"""Forward suite: agreement, physical laws.

Each run_* function:
  - Opens each solver once (outer loop) and runs all conditions through it
  - Saves results as JSON + CSV under results/{problem_name}/forward/
  - Returns the results dict

Run from the terminal:
    cd mosaic
    python -m benchmarks.suites.forward [--experiment EXPR] [--no-plots]
"""

from __future__ import annotations

import inspect
import threading

import jax.numpy as jnp
import numpy as np  # kept for np.savez and string arrays (JAX doesn't support these)

from mosaic.benchmarks.core.config import ProblemConfig
from mosaic.benchmarks.core.runner import (
    get_last_apply_error,
    safe_apply,
    safe_apply_with_extras,
    solver_sweep,
)
from mosaic.benchmarks.core.utils import (
    experiment_dir,
    extract_runs,
    is_valid,
    iter_runs,
    results_dir,
    save_experiment,
    trimmed_mean,
)

_SUITE = "forward"


# ── Agreement / baseline ──────────────────────────────────────────────────────


def run_agreement(
    cfg: ProblemConfig,
    tags: dict[str, str],
    *,
    _exp_key: str = "agreement",
    **overrides,
) -> dict:
    """Run all solvers across a sweep of one physics parameter; compute trimmed-mean
    consensus and per-solver error.

    Reads cfg.forward_defaults[_exp_key] (default "agreement").
    Use _exp_key="baseline" for the single-step baseline experiment.

    Expects each run dict to contain:
        ic=dict(name, seed)
        physics=dict(N, dt, steps, ...)
        sweep=dict(key, values)
        fine=dict(solvers, dt, steps)   [optional]

    Returns:
        {"by_param": {val: {solver: {"error": float, "valid": bool}}}, "spread": {val: float}}
        or {ic_name: <above>} when multiple IC runs are configured.
    """
    runs = cfg.forward_defaults.get(_exp_key, [])
    if not runs:
        raise NotImplementedError(
            f"run_agreement requires '{_exp_key}' list in forward_defaults "
            f"(not configured for '{cfg.name}')"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
        seed = ic_cfg.get("seed", 0)
        sweep = run.get("sweep", {})
        sweep_key = sweep.get("key")
        sweep_values = sweep.get("values", [])
        if not sweep_key or not sweep_values:
            raise NotImplementedError(
                f"run_agreement requires sweep.key and sweep.values "
                f"in forward_defaults['{_exp_key}'] (not configured for '{cfg.name}')"
            )
        fine_cfg = run.get("fine", {})
        fine_set = set(fine_cfg.get("solvers", set()))
        fine_dt = fine_cfg.get("dt")
        fine_steps = fine_cfg.get("steps")
        phys = run.get("physics", {})
        ic_subdir = ic_name if n_runs > 1 else ""

        _apply_errors: dict = {n: {} for n in cfg.solvers}
        _apply_errors_lock = threading.Lock()
        # Capture drag (force on obstacle) from solvers that produce it. Cylinder
        # experiments are the only ones that pass an obstacle; for the rest the
        # extras dict comes back empty and this stays as {}.
        _drags: dict = {n: {} for n in cfg.solvers}

        # Regenerate IC at each sweep value so N-sweeps use the correct IC shape.
        def _apply(name, t, val, _phys=phys, _ic_name=ic_name, _seed=seed):
            curr_phys = {**_phys, sweep_key: val}
            _ic = cfg.make_ic[_ic_name](L=cfg.domain_extent, seed=_seed, **curr_phys)
            _p = dict(curr_phys)
            if name in fine_set:
                if fine_dt is not None:
                    _p["dt"] = fine_dt
                if fine_steps is not None:
                    _p["steps"] = fine_steps
            inputs = cfg.make_inputs(name, _ic, domain_extent=cfg.domain_extent, **_p)
            result, extras, _ = safe_apply_with_extras(
                t, inputs, cfg.output_key, ["drag"]
            )
            if result is None:
                err = get_last_apply_error()
                with _apply_errors_lock:
                    _apply_errors[name][val] = err
            if "drag" in extras:
                _drags[name][val] = extras["drag"]
            norm = cfg.solvers[name].normalize_output
            return norm(result) if (norm is not None and result is not None) else result

        raw, _wall_times = solver_sweep(
            cfg,
            tags,
            sweep_values,
            _apply,
            experiment=_exp_key,
            label_fn=lambda v: f"{sweep_key}={v}",
            gpu_ids=overrides.get("gpu_ids"),
        )

        transform = cfg.agreement_transform
        xaxis_fn = cfg.agreement_xaxis
        exp_subdir = f"{_exp_key}/{ic_subdir}" if ic_subdir else _exp_key
        out_dir = experiment_dir(
            results_dir(),
            cfg.name,
            _SUITE,
            exp_subdir,
            suffix="_debug" if overrides.get("debug") else "",
        )
        fsnap: dict = {
            "sweep_values": np.array([float(v) for v in sweep_values]),
            "solver_names": np.array(list(cfg.solvers)),
        }
        if transform is None and sweep_values:
            # Store IC at the first sweep value so plots have a reference waveform.
            _ic0 = cfg.make_ic[ic_name](
                L=cfg.domain_extent, seed=seed, **{**phys, sweep_key: sweep_values[0]}
            )
            fsnap["ic"] = np.asarray(_ic0)
        if xaxis_fn is not None:
            fsnap["x_axis"] = np.asarray(xaxis_fn(**phys))

        # When only a subset of solvers is re-run (e.g. --solvers warp_ns), load
        # cached field arrays from the previous snapshot so consensus can still be
        # formed. cfg.solvers is filtered by cli.py to only the requested solvers,
        # so solver names are parsed from the snapshot keys rather than cfg.solvers.
        _snap_path = out_dir / "fields.npz"
        _snap_meta_prefixes = (
            "sweep_values",
            "solver_names",
            "ic",
            "consensus",
            "x_axis",
        )
        _cached_for_val: dict = {}
        _old_snap_extras: dict = {}  # non-rerun solver arrays from old snapshot → merged into fsnap on save
        if _snap_path.exists():
            try:
                with np.load(str(_snap_path), allow_pickle=False) as _snap:
                    for _ci, _cv in enumerate(sweep_values):
                        _cached_for_val[_cv] = {}
                        sfx = f"_{_ci}"
                        for _sfile in _snap.files:
                            if _sfile.endswith(sfx) and not any(
                                _sfile.startswith(p) for p in _snap_meta_prefixes
                            ):
                                _cn = _sfile[: -len(sfx)]
                                _arr_copy = np.asarray(_snap[_sfile])
                                if _cn not in tags:
                                    _cached_for_val[_cv][_cn] = _arr_copy
                                    _old_snap_extras[_sfile] = _arr_copy
            except Exception:
                pass

        # Pre-inspect analytic signature once (avoids repeated calls inside loop).
        _analytic_sig_params: set[str] = set()
        if cfg.analytic is not None:
            _analytic_sig_params = set(inspect.signature(cfg.analytic).parameters)

        by_param: dict = {}
        reference_label = "consensus"  # updated per-iteration when analytic is used
        for i, val in enumerate(sweep_values):
            valid_outputs = {
                n: raw[n][val]
                for n in cfg.solvers
                if val in raw.get(n, {}) and raw[n][val] is not None
            }
            comparable = (
                {
                    n: np.asarray(transform(arr, **phys))
                    for n, arr in valid_outputs.items()
                }
                if transform is not None
                else valid_outputs
            )
            # Augment with cached arrays for solvers not re-run in this invocation.
            # Snapshot stores already-transformed arrays, matching comparable's format.
            for _cn, _arr in _cached_for_val.get(val, {}).items():
                if _cn not in comparable:
                    comparable[_cn] = _arr
            if len(comparable) < 2:
                by_param[val] = {
                    n: {
                        "error": _apply_errors.get(n, {}).get(val),
                        "valid": False,
                    }
                    for n in cfg.solvers
                }
                # Still store whatever output we have so scientists can inspect
                # the lone solver's field even when consensus cannot be formed.
                for n, arr in comparable.items():
                    fsnap[f"{n}_{i}"] = arr
                continue
            if cfg.analytic is not None and "obstacle" not in run.get("physics", {}):
                curr_phys = {**phys, sweep_key: val}
                _ic_ref = cfg.make_ic[ic_name](
                    L=cfg.domain_extent, seed=seed, **curr_phys
                )
                # Build a representative inputs dict for dt/steps/domain_extent lookup.
                _inputs_ref = cfg.make_inputs(
                    next(iter(cfg.solvers)),
                    _ic_ref,
                    domain_extent=cfg.domain_extent,
                    **curr_phys,
                )
                _t_end = float(np.asarray(_inputs_ref["dt"])[0]) * int(
                    _inputs_ref["steps"]
                )
                _L = float(_inputs_ref.get("domain_extent", 2 * np.pi))
                _extra = {
                    k: curr_phys[k]
                    for k in _analytic_sig_params
                    if k in curr_phys and k not in ("ic", "t", "L")
                }
                reference = np.asarray(cfg.analytic(_ic_ref, t=_t_end, L=_L, **_extra))
                reference_label = "analytic"
            else:
                reference = trimmed_mean(list(comparable.values()))
                reference_label = "consensus"
            fsnap[f"consensus_{i}"] = reference
            for n, arr in comparable.items():
                fsnap[f"{n}_{i}"] = arr
            by_param[val] = {
                n: (
                    {
                        "error": cfg.error_fn(comparable[n], reference),
                        "valid": True,
                        **(
                            {"drag": _drags[n][val]} if val in _drags.get(n, {}) else {}
                        ),
                    }
                    if n in comparable
                    else {
                        "error": _apply_errors.get(n, {}).get(val),
                        "valid": False,
                    }
                )
                for n in cfg.solvers
            }
        # Merge old snapshot arrays for solvers not re-run in this invocation so
        # that a partial run (--solvers X) never discards other solvers' data.
        for _key, _arr in _old_snap_extras.items():
            if _key not in fsnap:
                fsnap[_key] = _arr
        # Rebuild solver_names from every {name}_{index} key present after the
        # merge — a partial rerun (--solvers X) would otherwise write a truncated
        # solver_names while old solver field arrays are still in the npz.
        _all_present_solvers = sorted(
            {
                k[: k.rfind("_")]
                for k in fsnap
                if "_" in k
                and not any(k.startswith(p) for p in _snap_meta_prefixes)
                and k[k.rfind("_") + 1 :].isdigit()
            }
        )
        fsnap["solver_names"] = np.array(_all_present_solvers)
        np.savez(out_dir / "fields.npz", **fsnap)

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
                for n in cfg.solvers
            ],
            cfg=cfg,
            harness_fn=run_agreement,
            wall_time_s=_wall_times,
        )
        if n_runs > 1:
            all_results[ic_name] = result
        else:
            all_results = result

    return all_results


# ── Physical laws ─────────────────────────────────────────────────────────────


def run_physical_laws(cfg: ProblemConfig, tags: dict[str, str], **overrides) -> dict:
    """Sweep complexity parameter, compute physical diagnostics at each value.

    Expects cfg.forward_defaults["physical_laws"] to be a list of run dicts, each with:
        ic=dict(name, seed)
        physics=dict(N, dt, steps, ...)
        sweep=dict(key, values)

    At each sweep value the IC is regenerated (so N sweeps work correctly), all
    solvers are run, and cfg.diagnostics are computed on each output.  If
    cfg.analytic is available, analytic_error is also reported.

    Returns:
        {"by_param": {val: {solver: {diag_name: value, "analytic_error": float|None}}},
         "sweep_key": str}
        or {ic_name: <above>} when multiple IC runs are configured.
    """
    runs = cfg.forward_defaults.get("physical_laws", [])
    if not runs:
        raise NotImplementedError(
            f"run_physical_laws requires 'physical_laws' list in forward_defaults "
            f"(not configured for '{cfg.name}')"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
        seed = ic_cfg.get("seed", 0)
        sweep = run.get("sweep", {})
        sweep_key = sweep.get("key")
        sweep_values = sweep.get("values", [])
        if not sweep_key or not sweep_values:
            raise NotImplementedError(
                f"run_physical_laws requires sweep.key and sweep.values "
                f"in forward_defaults['physical_laws'] (not configured for '{cfg.name}')"
            )
        phys = run.get("physics", {})
        run_name = run.get("name", ic_name)
        ic_subdir = run_name if n_runs > 1 else ""

        # Regenerate IC at each sweep value (handles N sweeps correctly).
        def _apply(name, t, val, _phys=phys, _ic_name=ic_name, _seed=seed):
            curr_phys = {**_phys, sweep_key: val}
            _ic = cfg.make_ic[_ic_name](L=cfg.domain_extent, seed=_seed, **curr_phys)
            inputs = cfg.make_inputs(
                name, _ic, domain_extent=cfg.domain_extent, **curr_phys
            )
            out = safe_apply(t, inputs, cfg.output_key)
            norm = cfg.solvers[name].normalize_output
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
        if cfg.analytic is not None:
            analytic_params = set(inspect.signature(cfg.analytic).parameters)

        by_param: dict = {}
        csv_rows: list[dict] = []
        diag_ctx = {"domain_extent": cfg.domain_extent}

        for val in sweep_values:
            curr_phys = {**phys, sweep_key: val}
            by_param[val] = {}

            # Analytic reference (regenerate IC at this val for correct shape).
            analytic_ref = None
            if cfg.analytic is not None:
                ic_ref = cfg.make_ic[ic_name](
                    L=cfg.domain_extent, seed=seed, **curr_phys
                )
                _dt = curr_phys.get("dt", 1.0)
                _steps = curr_phys.get("steps", 1)
                t_end = float(_dt) * int(_steps)
                analytic_kw = {
                    k: v for k, v in curr_phys.items() if k in analytic_params
                }
                try:
                    analytic_ref = cfg.analytic(
                        ic_ref, t=t_end, L=cfg.domain_extent, **analytic_kw
                    )
                except Exception:
                    analytic_ref = None

            for name in raw:
                out = raw[name][val]
                if not is_valid(out):
                    by_param[val][name] = None
                    continue
                diag: dict = {}
                for dname, fn in cfg.diagnostics.items():
                    try:
                        r = fn(out, **diag_ctx, **curr_phys)
                        if isinstance(r, (int, float)):
                            diag[dname] = float(r)
                    except Exception:
                        pass
                if analytic_ref is not None:
                    try:
                        diag["analytic_error"] = float(cfg.error_fn(out, analytic_ref))
                    except Exception:
                        pass
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


# ── run_all + registry ────────────────────────────────────────────────────────


def _agreement_variant(exp_key: str):
    def _run(cfg, tags, **kw):
        return run_agreement(cfg, tags, _exp_key=exp_key, **kw)

    _run.__name__ = f"run_{exp_key}"
    return _run


_EXPERIMENTS = {
    "baseline": _agreement_variant("baseline"),
    "agreement": run_agreement,
    "agreement/tgv": run_agreement,
    "agreement/multimode": run_agreement,
    "tgv_nu_sweep": _agreement_variant("tgv_nu_sweep"),
    "physical_laws": run_physical_laws,
    "cylinder": _agreement_variant("cylinder"),
    "source_baseline": _agreement_variant("source_baseline"),
    "source_linearity": _agreement_variant("source_linearity"),
}


def _plot_fns() -> dict:
    from mosaic.benchmarks.plots.forward import plot_agreement, plot_physical_laws

    return {
        "baseline": lambda cfg, **kw: plot_agreement(cfg, exp_key="baseline", **kw),
        "agreement": plot_agreement,
        "agreement/tgv": plot_agreement,
        "agreement/multimode": plot_agreement,
        "tgv_nu_sweep": lambda cfg, **kw: plot_agreement(
            cfg, exp_key="tgv_nu_sweep", **kw
        ),
        "physical_laws": plot_physical_laws,
        "cylinder": lambda cfg, **kw: plot_agreement(cfg, exp_key="cylinder", **kw),
        "source_baseline": lambda cfg, **kw: plot_agreement(
            cfg, exp_key="source_baseline", **kw
        ),
        "source_linearity": lambda cfg, **kw: plot_agreement(
            cfg, exp_key="source_linearity", **kw
        ),
    }


def run_all(
    cfg: ProblemConfig,
    tags: dict[str, str],
    experiments: list[str] | None = None,
    plots: bool = True,
) -> dict[str, dict]:
    """Run forward experiments and optionally generate plots."""
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
