# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward suite: agreement, physical laws.

Two kernels, each paired with an aggregate function that builds the
``by_param`` / ``spread`` schema. The framework
(:func:`mosaic.benchmarks.core.experiment.run_experiment`) drives the per-
(solver, sweep-value) inner loop; the kernel just runs one solver and
hands back its output array, leaving cross-solver reference selection,
error computation, and NPZ writing to the aggregate.

Run from the terminal:
    mosaic run <problem> forward [--experiments EXPR] [--plots-only]
"""

from __future__ import annotations

import contextlib
import inspect
from typing import Any

import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.experiment import (
    KernelContext,
    _build_result_envelope,
    kernel,
)
from mosaic.benchmarks.core.io import (
    load_cached_field_snapshots,
    save_csv,
    save_field_snapshots_npz,
)
from mosaic.benchmarks.core.reference import load_reference
from mosaic.benchmarks.core.runner import (
    get_last_apply_error,
    safe_apply,
    safe_apply_with_extras,
)
from mosaic.benchmarks.core.utils import active_solvers, trimmed_mean

_SUITE = "forward"


# ── Shared helpers ───────────────────────────────────────────────────────────


def _agreement_phys(ctx: KernelContext) -> dict:
    """Resolve per-solver physics: apply the fine-grid override when appropriate.

    When the solver is named in the run's
    ``reference={"solvers": {...}, "dt", "steps"}`` spec, override dt/steps;
    otherwise pass through.
    """
    ref = ctx.run.get("reference")
    if not isinstance(ref, dict):
        return ctx.phys
    if ctx.name not in set(ref.get("solvers", set())):
        return ctx.phys
    phys = dict(ctx.phys)
    if ref.get("dt") is not None:
        phys["dt"] = ref["dt"]
    if ref.get("steps") is not None:
        phys["steps"] = ref["steps"]
    return phys


def _analytic_reference(
    *,
    ic_name: str,
    seed: int,
    phys: dict,
    val: Any,
    sweep_key: str,
    analytic_params: set[str],
    make_ic: Any,
    make_inputs: Any,
    domain_extent: float,
    analytic: Any,
    solver_name_for_inputs: str,
) -> np.ndarray:
    """Compute the analytic reference field at one sweep value.

    ``solver_name_for_inputs`` is the solver whose ``make_inputs`` shape
    is queried for ``dt`` / ``steps`` / ``domain_extent`` — only metadata
    is read, not its output.
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
        for k in analytic_params
        if k in curr_phys and k not in ("ic", "t", "L")
    }
    return np.asarray(analytic(ic_ref, t=t_end, L=L, **extra))


# ── agreement ────────────────────────────────────────────────────────────────


def _agreement_aggregate(
    by_solver: Any,
    *,
    cfg: Any,
    tags: Any,
    run: Any,
    snapshots: Any,
    ic: Any,
    sweep_values: Any,
    sweep_key: Any,
    out_dir: Any,
    snapshot_filename: Any,
    snapshot_prefixes: Any,
    **_kw: Any,
) -> dict:
    """Cross-solver reference + per-solver error pass for agreement.

    Pulls each solver's per-sweep-value output array from ``snapshots``,
    augments with cached arrays from a prior partial run, picks
    analytic-vs-consensus reference per sweep value, computes errors via
    ``cfg.error_fn``, and writes the flat-layout ``fields.npz``.
    """
    error_fn = cfg.error_fn
    domain_extent = cfg.domain_extent
    ic_cfg = run.get("ic", {})
    ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
    seed = ic_cfg.get("seed", 0)
    phys = run.get("physics", {})

    run_reference = run.get("reference")
    analytic_fn = run_reference if callable(run_reference) else cfg.reference
    analytic_params = (
        set(inspect.signature(analytic_fn).parameters) if analytic_fn else set()
    )

    # ── Unpack per-solver outputs from snapshots ──────────────────────────
    outputs_per_val: dict = {v: {} for v in sweep_values}
    for name, suf_map in snapshots.items():
        for idx_str, arr in suf_map.items():
            idx = int(idx_str)
            outputs_per_val[sweep_values[idx]][name] = np.asarray(arr)

    # Augment with cached arrays for solvers not re-run this invocation.
    cached_for_val = load_cached_field_snapshots(
        out_dir / snapshot_filename,
        sweep_values,
        skip_solvers=tags,
        shared_prefixes=("sweep_values", "solver_names", "ic", "consensus", "x_axis"),
    )
    for v, cache in cached_for_val.items():
        for n, arr in cache.items():
            outputs_per_val[v].setdefault(n, arr)

    # ── Build results list + collect arrays for the flat-layout NPZ ──
    per_solver_for_npz: dict[str, dict[str, np.ndarray]] = {}
    shared_arrays: dict[str, np.ndarray] = {
        "sweep_values": np.array([float(v) for v in sweep_values]),
        "ic": np.asarray(ic),
    }
    apply_errors: dict[str, dict] = {}
    drags: dict[str, dict] = {}
    for name, smetrics in by_solver.items():
        for val, m in smetrics.items():
            if not m.get("valid", False):
                apply_errors.setdefault(name, {})[val] = m.get("apply_error")
            if "drag" in m:
                drags.setdefault(name, {})[val] = m["drag"]

    reference_label = "consensus"
    solver_names = [s.name for s in cfg.solvers]
    reference_solver = run.get("reference_solver")

    # Collect per-(solver, sweep_value) results
    flat_results: list[dict] = []

    exp_key = _kw.get("exp_key", "agreement")

    for i, val in enumerate(sweep_values):
        comparable = outputs_per_val.get(val, {})
        for n, arr in comparable.items():
            per_solver_for_npz.setdefault(n, {})[str(i)] = np.asarray(arr)

        has_analytic = analytic_fn is not None and "obstacle" not in phys
        has_ref_solver = reference_solver is not None and reference_solver in comparable
        precomputed = load_reference(cfg.name, exp_key, i)
        has_precomputed = precomputed is not None

        if len(comparable) == 0 or (
            len(comparable) < 2
            and not has_analytic
            and not has_ref_solver
            and not has_precomputed
        ):
            for n in solver_names:
                flat_results.append(
                    {
                        "solver": n,
                        "sweep_value": str(val),
                        "metrics": {
                            "error": apply_errors.get(n, {}).get(val),
                            "valid": False,
                        },
                    }
                )
            continue

        if has_analytic:
            reference = _analytic_reference(
                ic_name=ic_name,
                seed=seed,
                phys=phys,
                val=val,
                sweep_key=sweep_key,
                analytic_params=analytic_params,
                make_ic=cfg.make_ic,
                make_inputs=cfg.make_inputs,
                domain_extent=domain_extent,
                analytic=analytic_fn,
                solver_name_for_inputs=cfg.solvers[0].name,
            )
            reference_label = "analytic"
        elif has_ref_solver:
            reference = np.asarray(comparable[reference_solver])
            reference_label = f"solver:{reference_solver}"
        elif has_precomputed:
            reference = precomputed
            reference_label = "precomputed"
        else:
            reference = trimmed_mean(list(comparable.values()))
            reference_label = "consensus"
        shared_arrays[f"consensus_{i}"] = np.asarray(reference)

        for n in solver_names:
            if n in comparable:
                metrics: dict = {
                    "error": error_fn(comparable[n], reference),
                    "valid": True,
                }
                if val in drags.get(n, {}):
                    metrics["drag"] = drags[n][val]
            else:
                metrics = {
                    "error": apply_errors.get(n, {}).get(val),
                    "valid": False,
                }
            flat_results.append(
                {
                    "solver": n,
                    "sweep_value": str(val),
                    "metrics": metrics,
                }
            )

    save_field_snapshots_npz(
        out_dir,
        solver_names=solver_names,
        per_solver_arrays=per_solver_for_npz,
        shared_arrays=shared_arrays,
        filename=snapshot_filename,
        prefixes=snapshot_prefixes,
        flat_keys=True,
    )

    csv_rows = [
        {
            "solver": e["solver"],
            sweep_key: e["sweep_value"],
            "error": e["metrics"].get("error"),
            "valid": e["metrics"].get("valid"),
        }
        for e in flat_results
    ]
    save_csv(csv_rows, out_dir / "result.csv")

    # Compute spread per sweep value
    spread: dict = {}
    for val in sweep_values:
        val_str = str(val)
        errors = [
            e["metrics"]["error"]
            for e in flat_results
            if e["sweep_value"] == val_str
            and e["metrics"].get("valid")
            and e["metrics"].get("error") is not None
        ]
        spread[val] = float(jnp.std(jnp.array(errors))) if errors else 0.0

    return _build_result_envelope(
        cfg=cfg,
        suite=_SUITE,
        exp_key=exp_key,
        run=run,
        sweep_key=sweep_key,
        sweep_values=sweep_values,
        results=flat_results,
        extras={
            "spread": spread,
            "reference_label": reference_label,
        },
    )


@kernel(
    sweep_mode="default",
    ic_sweep=True,
    selector_fn=active_solvers,
    catch=True,
    catch_label="apply failed",
    snapshot_filename="fields.npz",
    snapshot_prefixes=("sweep_values", "ic", "consensus_", "x_axis"),
    aggregate_fn=_agreement_aggregate,
)
def agreement(t: Any, ctx: KernelContext) -> dict:
    """Run one solver at one sweep value. Aggregate forms the reference and computes errors.

    The kernel returns the normalised output array as a snapshot plus a
    metrics dict (``valid``, ``apply_error``, optional ``drag``). The
    aggregate then either picks the cfg-level analytic reference or builds
    a trimmed-mean consensus across solvers, computes per-solver
    ``error`` vs that reference, and writes ``fields.npz`` (flat layout).

    Per-run ``reference`` may be:
      * a callable — analytic ``(ic, t, L, **physics) → arr``;
      * a dict ``{"solvers": {...}, "dt": d, "steps": n}`` — bumps the
        named solvers to a finer dt/steps to form a fine-grid reference.

    Returns (by_param[val][solver] schema)::

        {"error": float, "valid": True[, "drag": ...]}
        | {"error": str | None, "valid": False}
    """
    phys = _agreement_phys(ctx)
    inputs = ctx.make_inputs(ctx.name, ctx.ic, **phys)
    result, extras, _ = safe_apply_with_extras(t, inputs, ctx.output_key, ["drag"])

    if result is None:
        return {"metrics": {"apply_error": get_last_apply_error(), "valid": False}}

    norm = ctx.cfg.solver(ctx.name).normalize_output
    out = norm(result) if norm is not None else result

    metrics: dict = {"valid": True}
    if "drag" in extras:
        metrics["drag"] = extras["drag"]
    return {"metrics": metrics, "snapshot": np.asarray(out)}


# ── physical laws ────────────────────────────────────────────────────────────


def _physical_laws_aggregate(
    by_solver: Any,
    *,
    cfg: Any,
    run: Any,
    snapshots: Any,
    sweep_values: Any,
    sweep_key: Any,
    out_dir: Any,
    **_kw: Any,
) -> dict:
    """Compute physical diagnostics + analytic-error per (solver, sweep-value)."""
    diagnostics = run.get("diagnostics") or {}
    domain_extent = cfg.domain_extent
    error_fn = cfg.error_fn
    ic_cfg = run.get("ic", {})
    ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
    seed = ic_cfg.get("seed", 0)
    phys = run.get("physics", {})

    analytic_fn = cfg.reference if callable(cfg.reference) else None
    analytic_params = (
        set(inspect.signature(analytic_fn).parameters) if analytic_fn else set()
    )

    # Per-solver outputs indexed by sweep value via snapshots dict.
    outputs_per_val: dict = {v: {} for v in sweep_values}
    for name, suf_map in snapshots.items():
        for idx_str, arr in suf_map.items():
            outputs_per_val[sweep_values[int(idx_str)]][name] = np.asarray(arr)

    flat_results: list[dict] = []
    csv_rows: list[dict] = []
    diag_ctx = {"domain_extent": domain_extent}

    for val in sweep_values:
        curr_phys = {**phys, sweep_key: val}

        analytic_ref = None
        if analytic_fn is not None:
            ic_ref = cfg.make_ic[ic_name](L=domain_extent, seed=seed, **curr_phys)
            t_end = float(curr_phys.get("dt", 1.0)) * int(curr_phys.get("steps", 1))
            analytic_kw = {k: v for k, v in curr_phys.items() if k in analytic_params}
            with contextlib.suppress(Exception):
                analytic_ref = analytic_fn(
                    ic_ref, t=t_end, L=domain_extent, **analytic_kw
                )

        for name, out in outputs_per_val.get(val, {}).items():
            if not by_solver.get(name, {}).get(val, {}).get("valid", False):
                flat_results.append(
                    {
                        "solver": name,
                        "sweep_value": str(val),
                        "metrics": None,
                    }
                )
                continue
            diag: dict = {}
            for dname, fn in diagnostics.items():
                with contextlib.suppress(Exception):
                    r = fn(out, **diag_ctx, **curr_phys)
                    if isinstance(r, int | float):
                        diag[dname] = float(r)
            if analytic_ref is not None:
                with contextlib.suppress(Exception):
                    diag["analytic_error"] = float(error_fn(out, analytic_ref))
            flat_results.append(
                {
                    "solver": name,
                    "sweep_value": str(val),
                    "metrics": diag,
                }
            )
            for dname, dval in diag.items():
                csv_rows.append(
                    {
                        "solver": name,
                        sweep_key: val,
                        "diagnostic": dname,
                        "value": dval,
                    }
                )

        # Mark solvers that didn't produce output.
        produced = {e["solver"] for e in flat_results if e["sweep_value"] == str(val)}
        for s in cfg.solvers:
            if s.name not in produced:
                flat_results.append(
                    {
                        "solver": s.name,
                        "sweep_value": str(val),
                        "metrics": None,
                    }
                )

    if csv_rows:
        save_csv(csv_rows, out_dir / "result.csv")

    return _build_result_envelope(
        cfg=cfg,
        suite=_SUITE,
        exp_key=_kw.get("exp_key", "physical_laws"),
        run=run,
        sweep_key=sweep_key,
        sweep_values=sweep_values,
        results=flat_results,
    )


@kernel(
    sweep_mode="default",
    ic_sweep=True,
    selector_fn=active_solvers,
    catch=True,
    catch_label="apply failed",
    snapshot_filename="fields.npz",
    snapshot_prefixes=("sweep_values", "ic"),
    aggregate_fn=_physical_laws_aggregate,
)
def physical_laws(t: Any, ctx: KernelContext) -> dict:
    """Run one solver at one sweep value; aggregate computes diagnostics.

    The kernel just returns the normalised output array; the aggregate
    pass evaluates each registered ``diagnostics`` function on it and
    (optionally) the analytic-reference error.
    """
    inputs = ctx.make_inputs(ctx.name, ctx.ic, **ctx.phys)
    out = safe_apply(t, inputs, ctx.output_key)
    if out is None:
        return {"metrics": {"valid": False}}
    norm = ctx.cfg.solver(ctx.name).normalize_output
    if norm is not None:
        out = norm(out)
    return {
        "metrics": {"valid": True},
        "snapshot": np.asarray(out),
    }
