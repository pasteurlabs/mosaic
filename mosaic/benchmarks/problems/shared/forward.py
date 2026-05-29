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

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.core.io import (
    load_cached_field_snapshots,
    save_csv,
    save_field_snapshots_npz,
)
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
    **_: Any,
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
    # ``shared_prefixes`` mirrors the save side so meta keys (``sweep_values``,
    # ``consensus_``, ``ic``, …) aren't mistaken for solver arrays.
    cached_for_val = load_cached_field_snapshots(
        out_dir / snapshot_filename,
        sweep_values,
        skip_solvers=tags,
        shared_prefixes=("sweep_values", "solver_names", "ic", "consensus", "x_axis"),
    )
    for v, cache in cached_for_val.items():
        for n, arr in cache.items():
            outputs_per_val[v].setdefault(n, arr)

    # ── Build by_param / spread + collect arrays for the flat-layout NPZ ──
    by_param: dict = {}
    per_solver_for_npz: dict[str, dict[str, np.ndarray]] = {}
    shared_arrays: dict[str, np.ndarray] = {
        "sweep_values": np.array([float(v) for v in sweep_values]),
        "ic": np.asarray(ic),
    }
    apply_errors: dict[str, dict] = {}
    drags: dict[str, dict] = {}
    for name, smetrics in by_solver.items():
        # smetrics is {sweep_val: {valid, [apply_error], [drag]}}.
        for val, m in smetrics.items():
            if not m.get("valid", False):
                apply_errors.setdefault(name, {})[val] = m.get("apply_error")
            if "drag" in m:
                drags.setdefault(name, {})[val] = m["drag"]

    reference_label = "consensus"
    solver_names = [s.name for s in cfg.solvers]
    reference_solver = run.get("reference_solver")

    for i, val in enumerate(sweep_values):
        comparable = outputs_per_val.get(val, {})
        for n, arr in comparable.items():
            per_solver_for_npz.setdefault(n, {})[str(i)] = np.asarray(arr)

        has_analytic = analytic_fn is not None and "obstacle" not in phys
        has_ref_solver = reference_solver is not None and reference_solver in comparable

        if len(comparable) < 2 and not has_analytic and not has_ref_solver:
            by_param[val] = {
                n: {"error": apply_errors.get(n, {}).get(val), "valid": False}
                for n in solver_names
            }
            continue

        if len(comparable) == 0:
            by_param[val] = {
                n: {"error": apply_errors.get(n, {}).get(val), "valid": False}
                for n in solver_names
            }
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
        else:
            reference = trimmed_mean(list(comparable.values()))
            reference_label = "consensus"
        shared_arrays[f"consensus_{i}"] = np.asarray(reference)

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
            for n in solver_names
        }

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
            "solver": n,
            sweep_key: val,
            "error": by_param[val][n]["error"],
            "valid": by_param[val][n]["valid"],
        }
        for val in sweep_values
        for n in solver_names
    ]
    save_csv(csv_rows, out_dir / "result.csv")

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

    return {
        "by_param": by_param,
        "spread": spread,
        "sweep_key": sweep_key,
        "reference_label": reference_label,
        "params": run,
    }


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
    **_: Any,
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

    by_param: dict = {}
    csv_rows: list[dict] = []
    diag_ctx = {"domain_extent": domain_extent}

    for val in sweep_values:
        curr_phys = {**phys, sweep_key: val}
        by_param[val] = {}

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
                by_param[val][name] = None
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

        # Mark solvers that didn't produce output as None for this val.
        for s in cfg.solvers:
            by_param[val].setdefault(s.name, None)

    if csv_rows:
        save_csv(csv_rows, out_dir / "result.csv")

    return {
        "by_param": by_param,
        "sweep_key": sweep_key,
        "params": run,
    }


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
