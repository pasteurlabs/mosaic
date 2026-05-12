"""Conductivity-recovery runner + helpers for thermal-mesh.

Recovers the SIMP density field ``rho`` from temperature observations by
minimising ``identification_error = ||T(rho) - T_target||^2``. Supports
two inner optimisers (Adam and L-BFGS) selected via the ``optimizer`` kwarg
to :func:`run_conductivity_recovery`.

This module lives under the ``thermal_mesh`` problem package because the
recovery loop is specific to thermal-mesh's SIMP conductivity setup; the
generic gradient-descent / L-BFGS primitives it depends on (``_run_optim``,
``_run_lbfgs``) remain in :mod:`mosaic.benchmarks.problems.shared.optimization`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import (
    experiment_dir,
    results_dir,
    save_harness_result,
    save_npz_merged,
)
from mosaic.benchmarks.core.runner import per_solver_loop

# JAX-traced loss_fn closures capture this reference at trace time;
# using the tracer-aware wrapper ensures primitive binding sees the
# active trace.
from mosaic.benchmarks.core.tracer_apply import apply_tesseract
from mosaic.benchmarks.core.utils import (
    active_differentiable_solvers,
    extract_runs,
    iter_runs,
)
from mosaic.benchmarks.problems.shared.optimization import (
    _run_lbfgs,
    _run_optim,  # noqa: F401 — part of the shared-primitive surface this module relies on
)

_SUITE = "optimization"


def _run_conductivity_adam_loop(
    loss_components,
    rho_opt,
    *,
    lr: float = 1e-2,
    max_iters: int = 500,
    patience: int = 50,
    snap_interval: int = 0,
    x_min: float = 1e-3,
    rho_history: list,
    name: str,
) -> tuple[jax.Array, list[float], dict]:
    """Adam optimisation loop for conductivity recovery.

    ``loss_components(rho)`` must return ``(loss, err)`` with ``has_aux=True``
    semantics so ``err`` (raw identification error) is tracked separately from
    the (loss + penalty) the optimiser sees.

    Returns ``(rho_opt, errors, info)`` where ``info`` has keys
    ``grad_norms`` and ``converged``. ``rho_history`` is appended in place.
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(rho_opt)
    best_err = jnp.inf
    no_improve = 0
    errors: list[float] = []
    grad_norms_adam: list[float] = []

    for i in range(max_iters):
        try:
            (_, err_val), g = jax.value_and_grad(loss_components, has_aux=True)(rho_opt)
            grad_norms_adam.append(float(jnp.linalg.norm(g.ravel())))
            updates, opt_state = optimizer.update(g, opt_state)
            # Project rho back into [x_min, 1] after each step.
            rho_opt = jnp.clip(optax.apply_updates(rho_opt, updates), x_min, 1.0)
            e = float(err_val)
            errors.append(e)
            if snap_interval > 0 and (i + 1) % snap_interval == 0:
                rho_history.append(np.array(rho_opt))
            if e < best_err:
                best_err, no_improve = e, 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break
        except Exception as exc:
            from mosaic.benchmarks.core.console import print_warn

            print_warn(f"{name} conductivity_recovery iter {i} failed: {exc}")
            break
    return (
        rho_opt,
        errors,
        {
            "grad_norms": grad_norms_adam,
            "converged": no_improve >= patience,
        },
    )


def _run_conductivity_lbfgs_loop(
    loss_components,
    rho_opt,
    *,
    max_iters: int = 500,
    snap_interval: int = 0,
    x_min: float = 1e-3,
    rho_history: list,
    **_unused,  # absorb Adam-only kwargs (lr, patience, name)
) -> tuple[jax.Array, list[float], dict]:
    """L-BFGS loop for conductivity recovery.

    Accepts ``loss_components(rho)`` returning ``(loss, err)`` for signature
    parity with the Adam loop; internally wraps it to a scalar for L-BFGS.
    """
    del _unused  # signature-parity slot, intentionally discarded

    def _loss_scalar(rho):
        loss, _ = loss_components(rho)
        return loss

    rho_opt, errors, lbfgs_diag = _run_lbfgs(
        _loss_scalar,
        rho_opt,
        max_iters=max_iters,
        snap_interval=snap_interval,
        history=rho_history if snap_interval > 0 else None,
        log_interval=10,
        clip_fn=lambda r: jnp.clip(r, x_min, 1.0),
    )
    return (
        rho_opt,
        errors,
        {
            "grad_norms": (lbfgs_diag or {}).get("grad_norms"),
            # L-BFGS terminates early via its internal stopping criterion; len(losses)
            # below max_iters means convergence, equal means the budget was exhausted.
            "converged": len(errors) < max_iters,
        },
    )


def _merge_rho_fields_npz(
    out_dir,
    rho_init,
    rho_truth,
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
        if rho_histories[sname]:
            payload[f"rho_history_{sname}"] = np.asarray(rho_histories[sname])
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


def _run_conductivity_recovery_impl(
    cfg: Problem,
    tags: dict[str, str],
    exp_key: str,
    harness_fn,
    *,
    make_ic,
    make_inputs,
    runs=None,
    _optim_loop=_run_conductivity_adam_loop,
    **overrides,
) -> dict:
    """Shared body for ``run_conductivity_recovery`` and its L-BFGS variant.

    The optimiser is selected by passing a loop function via ``_optim_loop``;
    both candidate loops accept the same ``loss_components(rho) -> (loss, err)``
    closure and return ``(rho, errors, info)`` with a uniform info dict
    (``grad_norms``, ``converged``).
    """
    if not runs:
        raise NotImplementedError(
            f"_run_conductivity_recovery_impl requires runs= payload for {exp_key!r} "
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
        compliance_key = phys.get("compliance_key", "identification_error")
        penalty_weight = float(phys.get("penalty_weight", 0.0))
        x_min = float(phys.get("x_min", 1e-3))
        snap_interval = int(phys.get("snap_interval", 0))
        ic_subdir = ic_name if n_runs > 1 else ""
        gpu_ids = overrides.get("gpu_ids")

        # Only forward optimiser knobs that the YAML sets; each loop carries
        # its own default so Adam's max_iters=500/patience=50 vs BFGS's
        # max_iters=500 (no patience) stay correct.
        loop_kwargs = {
            k: optim_cfg[k] for k in ("lr", "max_iters", "patience") if k in optim_cfg
        }

        rho_init = jnp.array(make_ic[ic_name](seed=seed, **phys))

        # Ground-truth conductivity field: prefer "two_gaussians_rho", then "two_gaussians".
        truth_ic_name: str | None = None
        for candidate in ("two_gaussians_rho", "two_gaussians"):
            if candidate in make_ic:
                truth_ic_name = candidate
                break
        if truth_ic_name is not None:
            rho_truth = np.asarray(
                make_ic[truth_ic_name](seed=seed, **phys), dtype=np.float32
            )
        else:
            rho_truth = None

        by_solver: dict = {}
        rho_snaps: dict = {}
        rho_histories: dict = {}

        candidate_solvers = active_differentiable_solvers(cfg, "optimization", exp_key)

        def _conductivity_recovery_work(name: str, t) -> None:
            rho_history: list = []

            _loss_phys = {k: v for k, v in phys.items() if k != "rho_0"}

            def loss_components(rho, _t=t):
                inp = make_inputs(name, rho, **_loss_phys)
                out = apply_tesseract(_t, inp)
                err = out.get(compliance_key)
                if err is None:
                    raise RuntimeError(
                        f"Solver '{name}' did not return '{compliance_key}'"
                    )
                penalty = (
                    penalty_weight * jnp.mean(rho**2) if penalty_weight > 0 else 0.0
                )
                return jnp.squeeze(err) + penalty, jnp.squeeze(err)

            rho_opt, errors, info = _optim_loop(
                loss_components,
                rho_init,
                snap_interval=snap_interval,
                x_min=x_min,
                rho_history=rho_history,
                name=name,
                **loop_kwargs,
            )

            rho_snaps[name] = np.array(rho_opt)
            rho_histories[name] = rho_history
            by_solver[name] = {
                "errors": errors,
                "initial_error": errors[0] if errors else None,
                "final_error": errors[-1] if errors else None,
                "error_reduction_pct": (
                    100.0 * (errors[0] - errors[-1]) / (abs(errors[0]) + 1e-30)
                    if len(errors) >= 2
                    else 0.0
                ),
                "n_iters": len(errors),
                "converged": info["converged"],
                "grad_norms": info["grad_norms"],
            }

        _wall_times = per_solver_loop(
            cfg,
            tags,
            candidate_solvers,
            _conductivity_recovery_work,
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
        _merge_rho_fields_npz(
            out_dir, rho_init, rho_truth, solver_names, rho_snaps, rho_histories
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


def run_conductivity_recovery(
    cfg: Problem,
    tags: dict[str, str],
    *,
    make_ic,
    make_inputs,
    optimizer: str = "adam",
    runs=None,
    exp_key: str = "conductivity_recovery",
    **overrides,
) -> dict:
    """Conductivity-field recovery: recover rho from temperature observations.

    Optimises the SIMP density field (rho, clipped to [x_min, 1]) to minimise
    identification_error = ||T(rho) - T_target||^2. The target temperature is
    produced by forward-solving with a two-Gaussian ground-truth conductivity
    and zero volumetric source (Neumann BC only).

    ``optimizer`` selects the inner optimiser:

      * ``"adam"`` — vanilla Adam (default).
      * ``"bfgs"`` — L-BFGS with zoom line-search.

    Problem-semantics state is passed explicitly (see
    :func:`run_recovery_constant_ic`).

    Each run dict in ``runs`` must contain:
        ic:     {name, seed}         — IC generator for initial rho (e.g. "uniform")
        physics: {nx, ny, nz, Lx, Ly, Lz, rho_0, Q_total, compliance_key,
                  penalty_weight, x_min, snap_interval, target_rho_from_two_gaussians}
        optim:  {lr, max_iters, patience}

    Returns:
        {"by_solver": {solver: {"errors", "final_error", "n_iters", "converged"}},
         "params": run}
        or {ic_name: <above>} when multiple runs are configured.
    """
    optim_loop = (
        _run_conductivity_lbfgs_loop
        if optimizer == "bfgs"
        else _run_conductivity_adam_loop
    )
    return _run_conductivity_recovery_impl(
        cfg,
        tags,
        exp_key,
        run_conductivity_recovery,
        runs=runs,
        _optim_loop=optim_loop,
        make_ic=make_ic,
        make_inputs=make_inputs,
        **overrides,
    )
