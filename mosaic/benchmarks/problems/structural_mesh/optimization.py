"""SIMP topology-optimization kernel for the structural-mesh problem.

This module hosts the topology-optimization kernel (:func:`topopt`). The
kernel minimises FEA compliance subject to a soft volume-fraction
constraint via SIMP-style density updates, dispatching to either
:func:`_run_optim` (Adam) or :func:`_run_lbfgs` (L-BFGS) from
``mosaic.benchmarks.problems.shared.optimization``.

Only the structural-mesh problem exercises this code path; the generic
optimization helpers are imported from the shared module.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.core.io import save_field_snapshots_npz

# JAX-traced loss_fn closures capture this reference at trace time;
# using the tracer-aware wrapper ensures primitive binding sees the
# active trace.
from mosaic.benchmarks.core.tracer_apply import apply_tesseract
from mosaic.benchmarks.problems.shared.optimization import (
    _run_lbfgs,
    _run_optim,
)


def _topopt_aggregate(
    by_solver,
    *,
    run,
    cfg,
    out_dir,
    snapshots,
    shared_extras,
    ic,
    snapshot_filename,
    snapshot_prefixes,
    **_,
) -> dict:
    """Aggregate per-solver topopt output → result dict + NPZ.

    ``snapshots`` carries ``{name: {"rho_final:": rho_final_arr,
    "rho_history:": rho_history_arr}}`` from each kernel invocation. The
    ``"rho_init"`` shared array comes via ``shared_extras``. The aggregate
    writes the NPZ with prefixes ``("rho_final", "rho_history")`` and
    returns the canonical ``{by_solver, params}`` result dict.
    """
    del cfg, ic  # IC is already present in ``shared_extras["rho_init"]``
    solver_names = list(snapshots.keys())
    save_field_snapshots_npz(
        out_dir,
        solver_names,
        snapshots,
        shared_arrays={k: np.asarray(v) for k, v in shared_extras.items()},
        filename=snapshot_filename,
        prefixes=snapshot_prefixes,
    )
    return {"by_solver": by_solver, "params": run}


@kernel(
    sweep_mode="none",
    aggregate_fn=_topopt_aggregate,
    snapshot_filename="topopt_fields.npz",
    snapshot_prefixes=("rho_final", "rho_history"),
)
def topopt(t, ctx: KernelContext) -> dict:
    """One solver's full topology-optimisation run.

    Builds the initial density from the IC factory, then runs either Adam
    or L-BFGS (selected via ``run["optimizer"]``) to minimise compliance
    subject to a soft volume-fraction penalty. The kernel returns the
    final density (``rho_final``), optional history snapshots
    (``rho_history``), the shared ``rho_init`` array, and a metrics dict
    matching the on-disk schema (``compliances``, ``vol_fracs``,
    ``final_compliance``, ``final_vol_frac``, ``n_iters``, ``converged``,
    ``grad_norms``).
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
    def loss_components(rho, _t):
        inp = ctx.make_inputs(ctx.name, rho, **phys)
        compliance = apply_tesseract(_t, inp)[compliance_key]
        vol_penalty = penalty_weight * (jnp.mean(rho) - v_frac) ** 2
        return compliance + vol_penalty, {
            "compliance": compliance,
            "vol_frac": jnp.mean(rho),
        }

    def loss_with_aux(rho, _t=t):
        return loss_components(rho, _t)

    aux_history: dict = {}
    history: list = []
    clip_fn = lambda r: jnp.clip(r, x_min, 1.0)

    if optimizer == "bfgs":
        max_iters = int(optim_cfg.get("max_iters", 100))

        def _log_iter(i, loss_val, _n=ctx.name, _m=max_iters):
            from mosaic.benchmarks.core.console import console

            console.print(
                f"  [green]{_n}[/] topopt_bfgs iter {i}/{_m} loss={loss_val:.4g}"
            )

        rho_opt, losses, diag = _run_lbfgs(
            loss_with_aux,
            rho_init,
            max_iters=max_iters,
            has_aux=True,
            aux_history=aux_history,
            clip_fn=clip_fn,
            snap_interval=snap_interval,
            history=history if snap_interval > 0 else None,
            log_fn=_log_iter,
            log_interval=10,
        )
        # losses include the volume penalty; do one extra forward to
        # recover the pure compliance value at the final iterate.
        _, aux_final = loss_components(rho_opt, t)
        final_compliance = float(aux_final["compliance"])
    else:
        lr = float(optim_cfg.get("lr", 1e-2))
        max_iters = int(optim_cfg.get("max_iters", 200))
        patience = int(optim_cfg.get("patience", 30))
        rho_opt, losses, diag = _run_optim(
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

    compliances = aux_history.get("compliance", [])
    vol_fracs = aux_history.get("vol_frac", [])
    final_vol_frac = vol_fracs[-1] if vol_fracs else float(jnp.mean(rho_opt))
    n_iters = len(losses)
    converged = n_iters < max_iters
    grad_norms = (diag or {}).get("grad_norms")

    snapshots: dict[str, np.ndarray] = {"rho_final": np.asarray(rho_opt)}
    if history:
        snapshots["rho_history"] = np.asarray(history)

    return {
        "metrics": {
            "compliances": compliances,
            "vol_fracs": vol_fracs,
            "final_compliance": final_compliance,
            "final_vol_frac": final_vol_frac,
            "n_iters": n_iters,
            "converged": converged,
            "grad_norms": grad_norms,
        },
        "snapshots": snapshots,
        "shared": {"rho_init": np.asarray(rho_init)},
    }
