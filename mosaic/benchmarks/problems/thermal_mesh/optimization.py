# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conductivity-recovery kernel + helpers for thermal-mesh.

Recovers the SIMP density field ``rho`` from temperature observations by
minimising ``identification_error = ||T(rho) - T_target||^2``. Supports
two inner optimisers (Adam and L-BFGS) selected via the ``optimizer`` kwarg
on :meth:`Problem.add_experiment` (folded into ``ctx.run`` by the framework).

This module lives under the ``thermal_mesh`` problem package because the
recovery loop is specific to thermal-mesh's SIMP conductivity setup; the
generic gradient-descent / L-BFGS primitives it depends on (``_run_optim``,
``_run_lbfgs``) remain in :mod:`mosaic.benchmarks.problems.shared.optimization`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.core.io import save_npz_merged

# JAX-traced loss_fn closures capture this reference at trace time;
# using the tracer-aware wrapper ensures primitive binding sees the
# active trace.
from mosaic.benchmarks.core.tracer_apply import apply_tesseract
from mosaic.benchmarks.core.utils import active_differentiable_solvers
from mosaic.benchmarks.problems.shared.optimization import _run_lbfgs, _run_optim


def _merge_rho_fields_npz(
    out_dir: Any,
    rho_init: Any,
    rho_truth: Any,
    solver_names: list[str],
    rho_snaps: dict,
    rho_histories: dict,
) -> None:
    """Save rho_fields.npz with merge logic preserving peer-solver entries."""
    payload: dict[str, np.ndarray] = {"rho_init": np.array(rho_init)}
    if rho_truth is not None:
        payload["rho_truth"] = rho_truth
    for sname in solver_names:
        payload[f"rho_final_{sname}"] = rho_snaps[sname]
        hist = rho_histories.get(sname)
        # ``rho_histories[sname]`` is a list of per-snapshot arrays under
        # Adam (or ``None`` / empty when ``snap_interval=0``). After the
        # has_aux collapse it may also arrive as an already-stacked ndarray;
        # ``len(...) > 0`` works for both without tripping numpy's
        # ambiguous-truth-value check on multi-element arrays.
        if hist is not None and len(hist) > 0:
            payload[f"rho_history_{sname}"] = np.asarray(hist)
    # Keep peer-solver entries from prior runs plus prior rho_truth (only when
    # the caller didn't compute a fresh one).
    new_keys = set(payload.keys())
    save_npz_merged(
        out_dir / "rho_fields.npz",
        payload,
        keep_old=lambda k: (
            k.startswith(("rho_final_", "rho_history_"))
            or (k == "rho_truth" and k not in new_keys)
        ),
    )


def _conductivity_recovery_aggregate(
    by_solver: Any,
    *,
    run: Any,
    out_dir: Any,
    snapshots: Any,
    shared_extras: Any,
    **_kw: Any,
) -> dict:
    """Aggregate per-solver conductivity-recovery output → result dict + NPZ.

    ``snapshots`` carries ``{name: {"rho_final:": arr, "rho_history:": arr}}``
    from each kernel invocation. ``shared_extras`` holds ``rho_init`` and
    optionally ``rho_truth`` (deposited by the first solver to run). Preserves
    the on-disk schema produced by the legacy harness: ``rho_fields.npz`` with
    flat per-solver keys (``rho_final_<name>`` / ``rho_history_<name>``) plus
    shared ``rho_init`` / ``rho_truth``.
    """
    rho_init = shared_extras.get("rho_init")
    rho_truth = shared_extras.get("rho_truth")

    rho_snaps: dict[str, np.ndarray] = {}
    rho_histories: dict[str, np.ndarray] = {}
    for name, suf_map in snapshots.items():
        # Multi-prefix keys land as "{prefix}:{idx_suffix}" where idx_suffix=""
        # for sweep_mode="none". A single-prefix kernel would use just "".
        for k, arr in suf_map.items():
            prefix, _, _ = k.partition(":")
            if prefix == "rho_final":
                rho_snaps[name] = np.asarray(arr)
            elif prefix == "rho_history":
                rho_histories[name] = np.asarray(arr)

    solver_names = list(rho_snaps.keys())
    _merge_rho_fields_npz(
        out_dir,
        rho_init,
        rho_truth,
        solver_names,
        rho_snaps,
        rho_histories,
    )

    from mosaic.benchmarks.core.experiment import (
        _build_result_envelope,
        _flatten_by_solver,
    )

    cfg = _kw.get("cfg")
    return _build_result_envelope(
        cfg=cfg,
        suite=_kw.get("suite", "optimization"),
        exp_key=_kw.get("exp_key", "conductivity_recovery"),
        run=run,
        sweep_key=None,
        sweep_values=None,
        results=_flatten_by_solver(by_solver, None),
    )


@kernel(
    sweep_mode="none",
    selector_fn=active_differentiable_solvers,
    aggregate_fn=_conductivity_recovery_aggregate,
    snapshot_filename="rho_fields.npz",
    snapshot_prefixes=("rho_final", "rho_history"),
)
def conductivity_recovery(t: Any, ctx: KernelContext) -> dict:
    """One solver's full conductivity-recovery optimisation.

    Recovers the SIMP density field ``rho`` (clipped to ``[x_min, 1]``) from
    temperature observations by minimising
    ``identification_error = ||T(rho) - T_target||^2``. The target
    temperature is produced by a forward solve with a two-Gaussian
    ground-truth conductivity (built by the ``two_gaussians_rho`` /
    ``two_gaussians`` IC).

    The inner optimiser is selected via ``ctx.run["optimizer"]`` (folded in
    by ``Problem.add_experiment``):

      * ``"adam"`` — vanilla Adam (default).
      * ``"bfgs"`` — L-BFGS with zoom line-search.

    Each run dict must contain:
        ic:     {name, seed}         — IC generator for initial rho (e.g. "uniform")
        physics: {nx, ny, nz, Lx, Ly, Lz, rho_0, Q_total, compliance_key,
                  penalty_weight, x_min, snap_interval, target_rho_from_two_gaussians}
        optim:  {lr, max_iters, patience}

    Returns ``metrics`` keyed by solver name (``errors``, ``initial_error``,
    ``final_error``, ``error_reduction_pct``, ``n_iters``, ``converged``,
    ``grad_norms``) plus the per-solver ``rho_final`` / ``rho_history``
    snapshots and the shared ``rho_init`` / ``rho_truth`` arrays consumed
    by :func:`_conductivity_recovery_aggregate`.
    """
    run = ctx.run
    phys = run.get("physics", {})
    optim_cfg = run.get("optim", {})
    optimizer = run.get("optimizer", "adam")

    compliance_key = phys.get("compliance_key", "identification_error")
    penalty_weight = float(phys.get("penalty_weight", 0.0))
    x_min = float(phys.get("x_min", 1e-3))
    snap_interval = int(phys.get("snap_interval", 0))

    ic_cfg = run.get("ic", {})
    make_ic = ctx.cfg.make_ic
    ic_name = ic_cfg.get("name", next(iter(make_ic)))

    rho_init = jnp.array(make_ic[ic_name](seed=ctx.seed, **phys))

    # Ground-truth conductivity field: prefer "two_gaussians_rho", then "two_gaussians".
    truth_ic_name: str | None = None
    for candidate in ("two_gaussians_rho", "two_gaussians"):
        if candidate in make_ic:
            truth_ic_name = candidate
            break
    if truth_ic_name is not None:
        rho_truth = np.asarray(
            make_ic[truth_ic_name](seed=ctx.seed, **phys), dtype=np.float32
        )
    else:
        rho_truth = None

    _loss_phys = {k: v for k, v in phys.items() if k != "rho_0"}

    def loss_components(rho: Any, _t: Any = t) -> Any:
        inp = ctx.make_inputs(ctx.name, rho, **_loss_phys)
        out = apply_tesseract(_t, inp)
        err = out.get(compliance_key)
        if err is None:
            raise RuntimeError(f"Solver '{ctx.name}' did not return '{compliance_key}'")
        err = jnp.squeeze(err)
        penalty = penalty_weight * jnp.mean(rho**2) if penalty_weight > 0 else 0.0
        return err + penalty, {
            "compliance": err,
            "rho_mean": jnp.mean(rho),
        }

    aux_history: dict = {}
    rho_history: list = []
    clip_fn = lambda r: jnp.clip(r, x_min, 1.0)

    # Adam defaults: lr=1e-2, max_iters=500, patience=50. BFGS uses
    # max_iters=500 (patience unused). Each branch reads its own knobs from
    # optim_cfg with those defaults so callers don't need to know which
    # optimiser is in use.
    if optimizer == "bfgs":
        max_iters = int(optim_cfg.get("max_iters", 500))
        rho_opt, losses, diag = _run_lbfgs(
            loss_components,
            rho_init,
            max_iters=max_iters,
            has_aux=True,
            aux_history=aux_history,
            clip_fn=clip_fn,
            snap_interval=snap_interval,
            history=rho_history if snap_interval > 0 else None,
            log_interval=10,
        )
    else:
        lr = float(optim_cfg.get("lr", 1e-2))
        max_iters = int(optim_cfg.get("max_iters", 500))
        patience = int(optim_cfg.get("patience", 50))
        rho_opt, losses, diag = _run_optim(
            loss_components,
            rho_init,
            lr=lr,
            max_iters=max_iters,
            patience=patience,
            has_aux=True,
            aux_history=aux_history,
            clip_fn=clip_fn,
            snap_interval=snap_interval,
            history=rho_history if snap_interval > 0 else None,
        )

    # ``errors`` is the per-iter identification-error trace (compliance aux).
    # Both branches converge when the loop terminates before max_iters
    # (Adam: patience-based early stop fired; L-BFGS: internal stopping
    # criterion fired).
    errors = aux_history.get("compliance", [])
    converged = len(losses) < max_iters
    grad_norms = (diag or {}).get("grad_norms")

    metrics = {
        "errors": errors,
        "initial_error": errors[0] if errors else None,
        "final_error": errors[-1] if errors else None,
        "error_reduction_pct": (
            100.0 * (errors[0] - errors[-1]) / (abs(errors[0]) + 1e-30)
            if len(errors) >= 2
            else 0.0
        ),
        "n_iters": len(errors),
        "converged": converged,
        "grad_norms": grad_norms,
    }

    snapshots: dict[str, np.ndarray] = {"rho_final": np.asarray(rho_opt)}
    if rho_history:
        snapshots["rho_history"] = np.asarray(rho_history)

    shared: dict[str, np.ndarray] = {"rho_init": np.asarray(rho_init)}
    if rho_truth is not None:
        shared["rho_truth"] = rho_truth

    return {
        "metrics": metrics,
        "snapshots": snapshots,
        "shared": shared,
    }
