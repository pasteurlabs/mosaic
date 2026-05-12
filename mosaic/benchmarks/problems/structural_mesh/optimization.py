"""SIMP topology-optimization runner and helpers for the structural-mesh problem.

This module hosts the topology-optimization harness (``run_topopt``) and its
supporting Adam / L-BFGS inner loops. The harness minimises FEA compliance
subject to a soft volume-fraction constraint via SIMP-style density updates.

Only the structural-mesh problem exercises this code path; the generic
optimization helpers (``_run_optim``, ``_run_lbfgs``) are imported from
``mosaic.benchmarks.problems.shared.optimization``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mosaic.benchmarks.core.config import Problem, has_vjp
from mosaic.benchmarks.core.io import (
    experiment_dir,
    results_dir,
    save_field_snapshots_npz,
    save_harness_result,
)
from mosaic.benchmarks.core.runner import per_solver_loop

# JAX-traced loss_fn closures capture this reference at trace time;
# using the tracer-aware wrapper ensures primitive binding sees the
# active trace.
from mosaic.benchmarks.core.tracer_apply import apply_tesseract
from mosaic.benchmarks.core.utils import extract_runs, iter_runs
from mosaic.benchmarks.problems.shared.optimization import (  # noqa: F401
    _run_lbfgs,
    _run_optim,
)

_SUITE = "optimization"


def _topopt_solvers(cfg: Problem) -> list[str]:
    """Solvers that can differentiate compliance.

    Checks spec.compliance_differentiable first (explicit override), then
    spec.differentiable (general flag), then falls back to runtime VJP endpoint
    detection.  Uses getattr with None defaults so that SolverSpec subclasses
    that omit these optional fields are handled gracefully.
    """
    result = []
    for spec in cfg.solvers:
        cd = getattr(spec, "compliance_differentiable", None)
        if cd is None:
            cd = getattr(spec, "differentiable", None)
        if cd is None:
            # Fall back to runtime VJP endpoint detection (same as active_differentiable_solvers)
            cd = has_vjp(spec)
        if cd:
            result.append(spec.name)
    return result


def _topopt_adam_loop(
    loss_components,
    rho_init,
    *,
    t,
    lr: float = 1e-2,
    max_iters: int = 200,
    patience: int = 30,
    x_min: float = 1e-3,
    snap_interval: int = 0,
    **_unused,  # absorb kwargs only the L-BFGS loop uses (e.g. ``name``)
) -> tuple[jax.Array, dict]:
    """Adam loop for topopt: per-iter compliance & vol_frac tracking + clip + patience.

    ``loss_components(rho, t)`` must return ``(loss, compliance)`` so the loop
    can record raw compliance separately from the (compliance + vol_penalty)
    loss the optimiser sees.
    """
    del _unused  # signature-parity slot, intentionally discarded
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(rho_init)
    rho_opt = rho_init
    compliances: list[float] = []
    vol_fracs: list[float] = []
    grad_norms: list[float] = []
    history: list = []
    best_c, no_improve = jnp.inf, 0

    def _loss(rho, _t=t):
        return loss_components(rho, _t)

    for i in range(max_iters):
        (_, compliance_val), g = jax.value_and_grad(_loss, has_aux=True)(rho_opt)
        grad_norms.append(float(jnp.linalg.norm(g.ravel())))
        updates, opt_state = optimizer.update(g, opt_state)
        rho_opt = jnp.clip(optax.apply_updates(rho_opt, updates), x_min, 1.0)
        c = float(compliance_val)
        compliances.append(c)
        vol_fracs.append(float(jnp.mean(rho_opt)))
        if snap_interval > 0 and (i + 1) % snap_interval == 0:
            history.append(np.array(rho_opt))
        if c < best_c:
            best_c, no_improve = c, 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    return rho_opt, {
        "compliances": compliances,
        "vol_fracs": vol_fracs,
        "final_compliance": compliances[-1] if compliances else float("nan"),
        "final_vol_frac": vol_fracs[-1] if vol_fracs else float(jnp.mean(rho_opt)),
        "n_iters": len(compliances),
        "converged": no_improve >= patience,
        "grad_norms": grad_norms,
        "history": history,
    }


def _topopt_lbfgs_loop(
    loss_components,
    rho_init,
    *,
    t,
    name: str,
    max_iters: int = 100,
    x_min: float = 1e-3,
    snap_interval: int = 0,
    **_unused,  # absorb kwargs only the Adam loop uses (lr, patience)
) -> tuple[jax.Array, dict]:
    """L-BFGS loop for topopt: scalar loss + clip; vol_fracs not tracked per-iter.

    ``loss_components(rho, t)`` returns ``(loss, compliance)``; the loop wraps
    it to a scalar for L-BFGS and runs one extra forward call afterwards to
    recover the pure compliance value (``losses[-1]`` includes the vol
    penalty).
    """
    del _unused  # signature-parity slot, intentionally discarded
    history: list = []

    def _loss_scalar(rho, _t=t):
        loss, _ = loss_components(rho, _t)
        return loss

    def _log_iter(i, loss_val, _n=name):
        from mosaic.benchmarks.core.console import console

        console.print(
            f"  [green]{_n}[/] topopt_bfgs iter {i}/{max_iters} loss={loss_val:.4g}"
        )

    rho_opt, losses, lbfgs_diag = _run_lbfgs(
        _loss_scalar,
        rho_init,
        max_iters=max_iters,
        snap_interval=snap_interval,
        history=history if snap_interval > 0 else None,
        log_fn=_log_iter,
        log_interval=10,
        clip_fn=lambda rho: jnp.clip(rho, x_min, 1.0),
    )

    try:
        _, final_compliance = loss_components(rho_opt, t)
        final_compliance = float(final_compliance)
    except Exception:
        final_compliance = losses[-1] if losses else float("nan")

    return rho_opt, {
        "compliances": losses,
        "vol_fracs": [],
        "final_compliance": final_compliance,
        "final_vol_frac": float(jnp.mean(rho_opt)),
        "n_iters": len(losses),
        "converged": len(losses) < max_iters,
        "grad_norms": (lbfgs_diag or {}).get("grad_norms"),
        "history": history,
    }


def _run_topopt_impl(
    cfg: Problem,
    tags: dict[str, str],
    exp_key: str,
    harness_fn,
    *,
    make_ic,
    make_inputs,
    runs=None,
    _optim_loop=_topopt_adam_loop,
    **overrides,
) -> dict:
    """Shared body for ``run_topopt`` and its L-BFGS variant.

    The optimiser is selected by passing a loop function via ``_optim_loop``.
    Both candidate loops (:func:`_topopt_adam_loop`, :func:`_topopt_lbfgs_loop`)
    return ``(rho_final, info)`` with the same ``info`` dict shape so the
    downstream save/result code is optimiser-agnostic.
    """
    if not runs:
        raise NotImplementedError(
            f"_run_topopt_impl requires runs= payload for {exp_key!r} "
            f"(not configured for '{cfg.name}')"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(make_ic)))
        seed = ic_cfg.get("seed", 0)
        phys = run.get("physics", {})
        optim_cfg = run.get("optim", {})
        v_frac = phys.get("v_frac")
        compliance_key = phys.get("compliance_key", "compliance")
        penalty_weight = phys.get("penalty_weight", 100.0)
        x_min = phys.get("x_min", 1e-3)
        snap_interval = int(phys.get("snap_interval", 0))
        ic_subdir = ic_name if n_runs > 1 else ""

        if v_frac is None:
            raise NotImplementedError(
                f"'{exp_key}' requires physics.v_frac in runs payload "
                f"(not configured for '{cfg.name}')"
            )

        # Only forward optimiser knobs that the YAML actually sets; each loop
        # function carries its own default so Adam's max_iters=200 vs BFGS's
        # max_iters=100 (etc.) stay correct without the wrappers having to
        # know which optimiser is in use.
        loop_kwargs = {
            k: optim_cfg[k] for k in ("lr", "max_iters", "patience") if k in optim_cfg
        }

        rho_init = make_ic[ic_name](rho_0=v_frac, seed=seed, **phys)

        by_solver: dict = {}
        rho_snaps: dict = {}
        rho_histories: dict = {}
        gpu_ids = overrides.get("gpu_ids")

        def _topopt_work(name: str, t) -> None:
            def loss_components(rho, _t):
                inp = make_inputs(name, rho, **phys)
                compliance = apply_tesseract(_t, inp)[compliance_key]
                vol_penalty = penalty_weight * (jnp.mean(rho) - v_frac) ** 2
                return compliance + vol_penalty, compliance

            rho_opt, info = _optim_loop(
                loss_components,
                rho_init,
                t=t,
                name=name,
                x_min=x_min,
                snap_interval=snap_interval,
                **loop_kwargs,
            )

            rho_snaps[name] = np.array(rho_opt)
            rho_histories[name] = info["history"]
            by_solver[name] = {
                "compliances": info["compliances"],
                "vol_fracs": info["vol_fracs"],
                "final_compliance": info["final_compliance"],
                "final_vol_frac": info["final_vol_frac"],
                "n_iters": info["n_iters"],
                "converged": info["converged"],
                "grad_norms": info["grad_norms"],
            }

        _wall_times = per_solver_loop(
            cfg,
            tags,
            _topopt_solvers(cfg),
            _topopt_work,
            gpu_ids=gpu_ids,
            print_done=False,
        )

        exp_subdir = f"{exp_key}/{ic_subdir}" if ic_subdir else exp_key
        out_dir = experiment_dir(
            results_dir(),
            cfg.name,
            _SUITE,
            exp_subdir,
            suffix="_debug" if overrides.get("debug") else "",
        )
        solver_names = list(rho_snaps.keys())
        per_solver: dict[str, dict[str, np.ndarray]] = {}
        for sname in solver_names:
            entry: dict[str, np.ndarray] = {"rho_final:": np.asarray(rho_snaps[sname])}
            if rho_histories[sname]:
                entry["rho_history:"] = np.asarray(rho_histories[sname])
            per_solver[sname] = entry
        save_field_snapshots_npz(
            out_dir,
            solver_names,
            per_solver,
            shared_arrays={"rho_init": np.array(rho_init)},
            filename="topopt_fields.npz",
            prefixes=("rho_final", "rho_history"),
        )

        result = {"by_solver": by_solver, "params": run}
        save_harness_result(
            result,
            cfg=cfg,
            suite=_SUITE,
            exp_subdir=exp_subdir,
            harness_fn=harness_fn,
            wall_time_s=_wall_times,
            debug=bool(overrides.get("debug")),
        )
        if n_runs > 1:
            all_results[ic_name] = result
        else:
            all_results = result

    return all_results


def run_topopt(
    cfg: Problem,
    tags: dict[str, str],
    *,
    make_ic,
    make_inputs,
    error_fn,
    output_key: str,
    domain_extent: float,
    optimizer: str = "adam",
    runs=None,
    **overrides,
) -> dict:
    """Topology optimisation: minimise compliance subject to a volume fraction constraint.

    ``optimizer`` selects the inner optimiser:

      * ``"adam"`` — Adam gradient descent with ρ clipped to [x_min, 1] and a
        soft volume penalty (default).
      * ``"bfgs"`` — L-BFGS with zoom line-search.

    Designed for static FEA problems (structural-mesh) where IC recovery is
    degenerate.

    Problem-semantics state is passed explicitly (``error_fn``, ``output_key``,
    ``domain_extent`` are accepted for signature parity with the other public
    harnesses but unused by topopt). See :func:`run_recovery_constant_ic` for
    the field semantics.

    Returns:
        {"by_solver": {solver: {"compliances", "vol_fracs", "final_compliance",
                                "final_vol_frac", "n_iters", "converged"}},
         "params": run}
        or {ic_name: <above>} when multiple runs are configured.
    """
    del error_fn, output_key, domain_extent  # unused by topopt; kept for parity
    optim_loop = _topopt_lbfgs_loop if optimizer == "bfgs" else _topopt_adam_loop
    exp_key = "topopt_bfgs" if optimizer == "bfgs" else "topopt"
    return _run_topopt_impl(
        cfg,
        tags,
        exp_key,
        run_topopt,
        runs=runs,
        _optim_loop=optim_loop,
        make_ic=make_ic,
        make_inputs=make_inputs,
        **overrides,
    )
