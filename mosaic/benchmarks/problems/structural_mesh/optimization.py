# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SIMP topology-optimization kernel for the structural-mesh problem.

This module hosts the topology-optimization kernel (:func:`topopt`). The
kernel minimises FEA compliance via SIMP-style density updates, dispatching
to :func:`_run_optim` (Adam) or :func:`_run_lbfgs` (L-BFGS) from
``mosaic.benchmarks.problems.shared.optimization`` under a soft
volume-fraction penalty, or to :func:`_run_mma` (MMA) under a hard
volume-fraction constraint.

Only the structural-mesh problem exercises this code path; the generic
optimization helpers are imported from the shared module.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_jax import apply_tesseract

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.core.io import save_field_snapshots_npz
from mosaic.benchmarks.problems.shared.optimization import (
    _check_finite_loss,
    _iters_to_plateau,
    _run_lbfgs,
    _run_optim,
)


def _run_mma(
    compliance_fn: Any,
    init_x: Any,
    *,
    v_frac: float,
    x_min: float,
    max_evals: int,
    patience: int,
    snap_interval: int = 0,
    history: list | None = None,
) -> tuple:
    """NLopt's MMA (``LD_MMA``, Svanberg 2002) on a density field ``init_x``.

    Minimises ``compliance_fn(x)`` subject to ``mean(x) <= v_frac`` and
    ``x_min <= x <= 1``, all imposed by MMA itself rather than by a penalty.
    NLopt counts objective evaluations, not MMA's outer iterations, so the
    budget ``max_evals``, the ``patience`` (evaluations without a lower
    compliance) and the returned traces are all per evaluation. Evaluations
    that violate the volume constraint are counted like any other.

    Returns ``(final_x, compliances, vol_fracs, diag)``. ``final_x`` is
    NLopt's result, or the last evaluated point when ``patience`` stopped the
    run. ``diag`` holds ``grad_norms``, ``n_evals``,
    ``n_iters`` (equal to ``n_evals``), ``iters_to_plateau`` of the
    compliances and ``converged``, which is true unless the evaluation budget
    ran out. Raises ``FloatingPointError`` once the compliance is not finite.
    """
    import nlopt

    shape, dtype = init_x.shape, init_x.dtype
    value_and_grad = jax.jit(jax.value_and_grad(compliance_fn))
    compliances: list[float] = []
    vol_fracs: list[float] = []
    grad_norms: list[float] = []
    last_x = [np.asarray(init_x, dtype=np.float64).ravel()]
    best, no_improve = np.inf, 0

    def objective(z: np.ndarray, grad: np.ndarray) -> float:
        nonlocal best, no_improve
        last_x[0] = z.copy()
        x = jnp.asarray(z.reshape(shape).astype(dtype))
        value, g = value_and_grad(x)
        value = float(value)
        _check_finite_loss(value, len(compliances))
        g = np.asarray(g, dtype=np.float64).ravel()
        if grad.size > 0:
            grad[:] = g
        compliances.append(value)
        vol_fracs.append(float(np.mean(z)))
        grad_norms.append(float(np.linalg.norm(g)))
        if (
            history is not None
            and snap_interval > 0
            and len(compliances) % snap_interval == 0
        ):
            history.append(np.asarray(x))
        if value < best:
            best, no_improve = value, 0
        else:
            no_improve += 1
        if no_improve >= patience:
            raise nlopt.ForcedStop()
        return value

    n = last_x[0].size

    def volume(z: np.ndarray, grad: np.ndarray) -> float:
        if grad.size > 0:
            grad[:] = 1.0 / n
        return float(np.mean(z)) - v_frac

    opt = nlopt.opt(nlopt.LD_MMA, n)
    opt.set_lower_bounds(np.full(n, x_min))
    opt.set_upper_bounds(np.ones(n))
    opt.set_min_objective(objective)
    opt.add_inequality_constraint(volume, 1e-8)
    opt.set_maxeval(max_evals)
    opt.set_ftol_rel(1e-9)
    opt.set_xtol_rel(1e-9)
    try:
        final_z = opt.optimize(last_x[0])
        converged = opt.last_optimize_result() != nlopt.MAXEVAL_REACHED
    except nlopt.ForcedStop:
        final_z, converged = last_x[0], True
    diag = {
        "grad_norms": grad_norms,
        "converged": converged,
        "n_evals": len(compliances),
        "n_iters": len(compliances),
        "iters_to_plateau": _iters_to_plateau(compliances),
    }
    return (
        jnp.asarray(final_z.reshape(shape).astype(dtype)),
        compliances,
        vol_fracs,
        diag,
    )


def _topopt_aggregate(
    by_solver: Any,
    *,
    run: Any,
    cfg: Any,
    out_dir: Any,
    snapshots: Any,
    shared_extras: Any,
    ic: Any,
    snapshot_filename: Any,
    snapshot_prefixes: Any,
    **_kw: Any,
) -> dict:
    """Aggregate per-solver topopt output → result dict + NPZ.

    ``snapshots`` carries ``{name: {"rho_final:": rho_final_arr,
    "rho_history:": rho_history_arr}}`` from each kernel invocation. The
    ``"rho_init"`` shared array comes via ``shared_extras``. The aggregate
    writes the NPZ with prefixes ``("rho_final", "rho_history")`` and
    returns the unified result envelope.
    """
    from mosaic.benchmarks.core.experiment import (
        _build_result_envelope,
        _flatten_by_solver,
    )

    del ic  # IC is already present in ``shared_extras["rho_init"]``
    solver_names = list(snapshots.keys())
    save_field_snapshots_npz(
        out_dir,
        solver_names,
        snapshots,
        shared_arrays={k: np.asarray(v) for k, v in shared_extras.items()},
        filename=snapshot_filename,
        prefixes=snapshot_prefixes,
    )
    return _build_result_envelope(
        cfg=cfg,
        suite=_kw.get("suite", "optimization"),
        exp_key=_kw.get("exp_key", "topopt"),
        run=run,
        sweep_key=None,
        sweep_values=None,
        results=_flatten_by_solver(by_solver, None),
    )


@kernel(
    sweep_mode="none",
    aggregate_fn=_topopt_aggregate,
    snapshot_filename="topopt_fields.npz",
    snapshot_prefixes=("rho_final", "rho_history"),
)
def topopt(t: Any, ctx: KernelContext) -> dict:
    """One solver's full topology-optimisation run.

    Builds the initial density from the IC factory, then minimises
    compliance with the optimizer named by ``run["optimizer"]``: ``"adam"``
    (default) or ``"bfgs"`` under a soft volume-fraction penalty, or
    ``"mma"`` under a hard volume-fraction constraint. The kernel returns the
    final density (``rho_final``), optional history snapshots
    (``rho_history``), the shared ``rho_init`` array, and a metrics dict
    matching the on-disk schema (``compliances``, ``vol_fracs``,
    ``final_compliance``, ``final_vol_frac``, ``n_iters``, ``n_evals``,
    ``iters_to_plateau``, ``converged``, ``grad_norms``). For Adam and
    L-BFGS, ``iters_to_plateau`` follows the loss, which includes the volume
    penalty, so it can differ from where the compliance alone levels off.
    For MMA the traces, ``n_iters`` and ``iters_to_plateau`` count objective
    evaluations (see :func:`_run_mma`).
    """
    run = ctx.run
    phys = run.get("physics", {})
    optim_cfg = run.get("optim", {})
    optimizer = run.get("optimizer", "adam")

    v_frac = phys.get("v_frac")
    compliance_key = phys.get("compliance_key", "compliance")
    penalty_weight = phys.get("penalty_weight", 100.0)
    x_min = phys.get("x_min", 1e-3)
    snap_interval = int(phys.get("snap_interval", 0))

    if v_frac is None:
        raise NotImplementedError(
            "topopt requires physics.v_frac in runs payload "
            f"(not configured for {ctx.cfg.name!r})"
        )

    ic_cfg = run.get("ic", {})
    ic_name = ic_cfg.get("name", next(iter(ctx.cfg.make_ic)))
    rho_init = ctx.cfg.make_ic[ic_name](rho_0=v_frac, seed=ctx.seed, **phys)

    # Adam defaults to max_iters=200/patience=30/lr=1e-2; BFGS uses
    # max_iters=100. Each branch reads its own knobs from optim_cfg with
    # those defaults so callers don't need to know which optimiser is in
    # use.
    def loss_components(rho: Any, _t: Any) -> Any:
        inp = ctx.make_inputs(ctx.name, rho, **phys)
        compliance = apply_tesseract(_t, inp)[compliance_key]
        vol_penalty = penalty_weight * (jnp.mean(rho) - v_frac) ** 2
        return compliance + vol_penalty, {
            "compliance": compliance,
            "vol_frac": jnp.mean(rho),
        }

    def loss_with_aux(rho: Any, _t: Any = t) -> Any:
        return loss_components(rho, _t)

    aux_history: dict = {}
    history: list = []
    clip_fn = lambda r: jnp.clip(r, x_min, 1.0)

    if optimizer == "bfgs":
        max_iters = int(optim_cfg.get("max_iters", 100))

        def _log_iter(
            i: Any, loss_val: Any, _n: Any = ctx.name, _m: Any = max_iters
        ) -> None:
            from mosaic.benchmarks.core.console import console

            console.print(
                f"  [green]{_n}[/] topopt_bfgs iter {i}/{_m} loss={loss_val:.4g}"
            )

        rho_opt, _, diag = _run_lbfgs(
            loss_with_aux,
            rho_init,
            max_iters=max_iters,
            has_aux=True,
            aux_history=aux_history,
            bounds=(x_min, 1.0),
            snap_interval=snap_interval,
            history=history if snap_interval > 0 else None,
            log_fn=_log_iter,
            log_interval=10,
        )
        # losses include the volume penalty; do one extra forward to
        # recover the pure compliance value at the final iterate.
        _, aux_final = loss_components(rho_opt, t)
        final_compliance = float(aux_final["compliance"])
    elif optimizer == "mma":
        # Hard volume constraint: the objective is the compliance alone.
        rho_opt, compliances, vol_fracs, diag = _run_mma(
            lambda rho: loss_components(rho, t)[1]["compliance"],
            rho_init,
            v_frac=v_frac,
            x_min=x_min,
            max_evals=int(optim_cfg.get("max_iters", 200)),
            patience=int(optim_cfg.get("patience", 30)),
            snap_interval=snap_interval,
            history=history if snap_interval > 0 else None,
        )
        aux_history = {"compliance": compliances, "vol_frac": vol_fracs}
        # NLopt's result need not be the last evaluated point.
        _, aux_final = loss_components(rho_opt, t)
        final_compliance = float(aux_final["compliance"])
    elif optimizer == "adam":
        lr = float(optim_cfg.get("lr", 1e-2))
        max_iters = int(optim_cfg.get("max_iters", 200))
        patience = int(optim_cfg.get("patience", 30))
        rho_opt, _, diag = _run_optim(
            loss_with_aux,
            rho_init,
            lr=lr,
            max_iters=max_iters,
            patience=patience,
            has_aux=True,
            aux_history=aux_history,
            clip_fn=clip_fn,
            snap_interval=snap_interval,
            history=history if snap_interval > 0 else None,
        )
        compliance_trace = aux_history.get("compliance", [])
        final_compliance = compliance_trace[-1] if compliance_trace else float("nan")
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

    compliances = aux_history.get("compliance", [])
    vol_fracs = aux_history.get("vol_frac", [])
    final_vol_frac = (
        float(jnp.mean(rho_opt))
        if optimizer == "mma" or not vol_fracs
        else vol_fracs[-1]
    )
    converged = diag["converged"]
    grad_norms = diag.get("grad_norms")

    snapshots: dict[str, np.ndarray] = {"rho_final": np.asarray(rho_opt)}
    if history:
        snapshots["rho_history"] = np.asarray(history)

    return {
        "metrics": {
            "compliances": compliances,
            "vol_fracs": vol_fracs,
            "final_compliance": final_compliance,
            "final_vol_frac": final_vol_frac,
            "n_iters": diag["n_iters"],
            "n_evals": diag["n_evals"],
            "iters_to_plateau": diag["iters_to_plateau"],
            "converged": converged,
            "grad_norms": grad_norms,
        },
        "snapshots": snapshots,
        "shared": {"rho_init": np.asarray(rho_init)},
    }
