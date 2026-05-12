"""Drag-optimisation runner and helpers for the navier-stokes grid problem.

This module hosts the ``run_drag_opt`` entry point and its private helpers,
which optimise an inflow velocity profile to minimise drag on an embedded
obstacle. The code is ns-grid-specific (drag is a fluid-flow quantity tied to
this problem's tesseract outputs), so it lives next to the problem rather than
in ``shared/optimization.py``.

The two inner optimisation primitives ``_run_optim`` (Adam) and ``_run_lbfgs``
(L-BFGS with zoom line-search) remain in ``shared/optimization.py`` and are
imported here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import (
    experiment_dir,
    results_dir,
    save_experiment,
    save_json,
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
from mosaic.benchmarks.problems.shared.optimization import _run_lbfgs

_SUITE = "optimization"


def _drag_opt_out_dir(cfg: Problem, ic_subdir: str, debug: bool, exp_name: str):
    """Compute the on-disk output directory for a drag_opt[_bfgs] run."""
    suffix = "_debug" if debug else ""
    if ic_subdir:
        parent = experiment_dir(
            results_dir(), cfg.name, _SUITE, exp_name, suffix=suffix
        )
        out_dir = parent / ic_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    return experiment_dir(results_dir(), cfg.name, _SUITE, exp_name, suffix=suffix)


def _drag_capture_flow(
    name: str,
    t,
    profile,
    phys: dict,
    *,
    make_inputs,
    domain_extent: float,
) -> np.ndarray | None:
    """Forward the (initial or final) profile and return the velocity field if any."""
    try:
        de = phys.get("domain_extent", domain_extent)
        inp = make_inputs(
            name,
            profile,
            domain_extent=de,
            **{k: v for k, v in phys.items() if k != "domain_extent"},
        )
        out = apply_tesseract(t, inp)
        vel = out.get("result")
        if vel is not None:
            return np.array(vel)
    except Exception:
        pass
    return None


def _drag_build_solver_entry(
    drags: list,
    flow_rates: list,
    grad_norms: list,
    n_iters_total: int,
    converged: bool,
    in_progress: bool = False,
    error: str | None = None,
) -> dict:
    """Assemble a single solver's drag-opt result dict (used for partial + final)."""
    entry = {
        "drags": drags,
        "flow_rates": flow_rates,
        "initial_drag": drags[0] if drags else None,
        "final_drag": drags[-1] if drags else None,
        "drag_reduction_pct": (
            100.0 * (abs(drags[0]) - abs(drags[-1])) / (abs(drags[0]) + 1e-30)
            if len(drags) >= 2
            else 0.0
        ),
        "n_iters": n_iters_total,
        "converged": converged,
        "grad_norms": grad_norms,
    }
    if in_progress:
        entry["in_progress"] = True
    if error is not None:
        entry["error"] = error
    return entry


@dataclass
class _DragAdamConfig:
    """Hyper-parameters for ``_run_drag_adam_loop`` (groups loop knobs)."""

    lr: float
    max_iters: int
    patience: int
    flow_penalty_weight: float
    snap_interval: int
    U_mean: float


def _run_drag_adam_loop(
    name: str,
    t,
    profile_init,
    phys: dict,
    adam_cfg: _DragAdamConfig,
    by_solver: dict,
    write_partial,
    *,
    make_inputs,
    domain_extent: float,
) -> tuple:
    """Run the Adam drag-optimisation loop for one solver.

    Returns ``(profile, profile_history, drags, flow_rates, grad_norms,
    non_finite_grad, no_improve_at_end)``.
    """
    profile = profile_init
    optimizer = optax.adam(adam_cfg.lr)
    opt_state = optimizer.init(profile)
    drags: list[float] = []
    flow_rates: list[float] = []
    best_drag = jnp.inf
    no_improve = 0
    profile_history: list = []
    grad_norms: list[float] = []
    U_mean = adam_cfg.U_mean
    flow_penalty_weight = adam_cfg.flow_penalty_weight
    snap_interval = adam_cfg.snap_interval
    max_iters = adam_cfg.max_iters
    patience = adam_cfg.patience

    def loss_fn(p, _t=t):
        # domain_extent may already be inside phys (drag_opt sets it to 1.0);
        # avoid passing it twice by letting phys take precedence.
        _de = phys.get("domain_extent", domain_extent)
        inp = make_inputs(
            name,
            p,
            domain_extent=_de,
            **{k: v for k, v in phys.items() if k != "domain_extent"},
        )
        out = apply_tesseract(_t, inp)
        drag_val = out.get("drag")
        if drag_val is None:
            raise RuntimeError(
                f"Solver '{name}' did not return 'drag' — "
                "ensure the tesseract implements drag computation."
            )
        flow_penalty = flow_penalty_weight * (jnp.mean(p) - U_mean) ** 2
        return jnp.abs(jnp.squeeze(drag_val)) + flow_penalty, jnp.squeeze(drag_val)

    non_finite_grad = False
    for i in range(max_iters):
        (_, drag_val), g = jax.value_and_grad(loss_fn, has_aux=True)(profile)
        if not jnp.isfinite(g).all():
            from mosaic.benchmarks.core.console import print_warn

            print_warn(
                f"{name} drag_opt iter {i}: non-finite gradient detected "
                f"(max|g|={float(jnp.max(jnp.abs(g))):.3e}); "
                "aborting optimisation loop"
            )
            non_finite_grad = True
            break
        grad_norms.append(float(jnp.linalg.norm(g.ravel())))
        updates, opt_state = optimizer.update(g, opt_state)
        profile = jnp.clip(optax.apply_updates(profile, updates), 0.0, 3.0 * U_mean)
        d = float(drag_val)
        drags.append(d)
        flow_rates.append(float(jnp.mean(profile)))
        if snap_interval > 0 and (i + 1) % snap_interval == 0:
            profile_history.append(np.asarray(profile))
        if abs(d) < best_drag:
            best_drag, no_improve = abs(d), 0
        else:
            no_improve += 1
        by_solver[name] = _drag_build_solver_entry(
            drags,
            flow_rates,
            grad_norms,
            len(drags),
            converged=False,
            in_progress=True,
        )
        if (i + 1) % 10 == 0:
            write_partial()
        if no_improve >= patience:
            break
    return (
        profile,
        profile_history,
        drags,
        flow_rates,
        grad_norms,
        non_finite_grad,
        no_improve,
    )


def _drag_bfgs_final_drag(
    name: str,
    t,
    profile,
    phys: dict,
    losses: list,
    *,
    make_inputs,
    domain_extent: float,
) -> tuple[float, np.ndarray | None]:
    """Forward the final profile to get the drag scalar and velocity field.

    Falls back to ``losses[-1]`` (or NaN) if the forward call fails.
    Returns ``(final_drag, velocity_field_or_None)``.
    """
    try:
        de = phys.get("domain_extent", domain_extent)
        inp_f = make_inputs(
            name,
            profile,
            domain_extent=de,
            **{k: v for k, v in phys.items() if k != "domain_extent"},
        )
        out_f = apply_tesseract(t, inp_f)
        final_drag = float(jnp.squeeze(out_f.get("drag", jnp.array(float("nan")))))
        vel = out_f.get("result")
        return final_drag, (np.array(vel) if vel is not None else None)
    except Exception:
        return (losses[-1] if losses else float("nan")), None


def _merge_drag_profiles_npz(
    out_dir,
    profile_init,
    profile_snaps: dict,
    profile_histories: dict,
) -> None:
    """Save profiles.npz with merge so single-solver reruns don't wipe peers."""
    payload: dict[str, np.ndarray] = {"initial": np.array(profile_init)}
    for k, v in profile_snaps.items():
        payload[f"final_{k}"] = v
    for k, v in profile_histories.items():
        payload[f"profile_history_{k}"] = v
    save_npz_merged(
        out_dir / "profiles.npz",
        payload,
        keep_old=lambda k: k.startswith(("final_", "profile_history_")),
    )


def _merge_drag_flow_fields_npz(
    out_dir,
    flow_init_snaps: dict,
    flow_snaps: dict,
) -> None:
    """Save flow_fields.npz with merge logic (per-solver and canonical entries)."""
    if not (flow_snaps or flow_init_snaps):
        return
    payload: dict[str, np.ndarray] = {}
    for sn, v in flow_init_snaps.items():
        payload[f"flow_initial_{sn}"] = v
    for sn, v in flow_snaps.items():
        payload[f"flow_final_{sn}"] = v
    # Canonical ``flow_initial`` is set only on the first write — preserve
    # whatever ``save_npz_merged`` finds on disk and seed from the new
    # per-solver entries when no prior file exists.
    ff_path = out_dir / "flow_fields.npz"
    if not ff_path.exists() and flow_init_snaps:
        payload["flow_initial"] = next(iter(flow_init_snaps.values()))
    save_npz_merged(ff_path, payload)


# ── drag_opt: shared body, Adam vs L-BFGS as an argument ─────────────────────


def _drag_opt_adam_loop(  # noqa: PLR0913 — explicit-deps signature
    name: str,
    t,
    profile_init,
    phys: dict,
    *,
    lr: float = 1e-3,
    max_iters: int = 150,
    patience: int = 30,
    flow_penalty_weight: float = 50.0,
    snap_interval: int = 0,
    U_mean: float = 0.5,
    by_solver: dict,
    write_partial,
    make_inputs,
    domain_extent: float,
) -> tuple[jax.Array, list, dict]:
    """Adam loop for drag_opt; supports per-iter partial-checkpoint flushes.

    Returns ``(profile_final, profile_history, by_solver_entry)``.
    """
    (
        profile,
        profile_history,
        drags,
        flow_rates,
        grad_norms,
        non_finite_grad,
        no_improve,
    ) = _run_drag_adam_loop(
        name,
        t,
        profile_init,
        phys,
        _DragAdamConfig(
            lr=lr,
            max_iters=max_iters,
            patience=patience,
            flow_penalty_weight=flow_penalty_weight,
            snap_interval=snap_interval,
            U_mean=U_mean,
        ),
        by_solver,
        write_partial,
        make_inputs=make_inputs,
        domain_extent=domain_extent,
    )
    entry = _drag_build_solver_entry(
        drags,
        flow_rates,
        grad_norms,
        len(drags),
        converged=(not non_finite_grad) and (no_improve >= patience),
        error="non-finite gradients" if non_finite_grad else None,
    )
    return profile, profile_history, entry


def _drag_opt_lbfgs_loop(
    name: str,
    t,
    profile_init,
    phys: dict,
    *,
    max_iters: int = 50,
    flow_penalty_weight: float = 50.0,
    snap_interval: int = 0,
    U_mean: float = 0.5,
    make_inputs,
    domain_extent: float,
    **_unused,  # absorb Adam-only kwargs (by_solver, write_partial, lr, patience)
) -> tuple[jax.Array, list, dict]:
    """L-BFGS loop for drag_opt; no partial checkpointing.

    Returns ``(profile_final, profile_history, by_solver_entry)``. Computes
    ``final_drag`` via one extra forward pass after convergence so the entry
    reflects the post-clip drag rather than the last loss (which includes the
    flow penalty).
    """
    del _unused  # signature-parity slot, intentionally discarded
    profile_history: list = []

    def loss_fn(p, _t=t):
        _de = phys.get("domain_extent", domain_extent)
        inp = make_inputs(
            name,
            p,
            domain_extent=_de,
            **{k: v for k, v in phys.items() if k != "domain_extent"},
        )
        out = apply_tesseract(_t, inp)
        drag_val = out.get("drag")
        if drag_val is None:
            raise RuntimeError(f"Solver '{name}' did not return 'drag'")
        flow_penalty = flow_penalty_weight * (jnp.mean(p) - U_mean) ** 2
        return jnp.abs(jnp.squeeze(drag_val)) + flow_penalty

    def _log_iter(i, loss_val, _n=name):
        from mosaic.benchmarks.core.console import console

        console.print(
            f"  [green]{_n}[/] drag_opt_bfgs iter {i}/{max_iters} loss={loss_val:.4g}"
        )

    profile, losses, lbfgs_diag = _run_lbfgs(
        loss_fn,
        profile_init,
        max_iters=max_iters,
        snap_interval=snap_interval,
        history=profile_history if snap_interval > 0 else None,
        log_fn=_log_iter,
        log_interval=10,
        clip_fn=lambda p: jnp.clip(p, 0.0, 3.0 * U_mean),
    )

    final_drag, _ = _drag_bfgs_final_drag(
        name,
        t,
        profile,
        phys,
        losses,
        make_inputs=make_inputs,
        domain_extent=domain_extent,
    )
    initial_drag = losses[0] if losses else None
    entry = {
        "drags": losses,
        "flow_rates": [],
        "initial_drag": initial_drag,
        "final_drag": final_drag,
        "drag_reduction_pct": (
            100.0 * (abs(losses[0]) - abs(final_drag)) / (abs(losses[0]) + 1e-30)
            if losses
            else 0.0
        ),
        "n_iters": len(losses),
        "converged": len(losses) < max_iters,
        "grad_norms": (lbfgs_diag or {}).get("grad_norms"),
    }
    return profile, profile_history, entry


def _run_drag_opt_impl(
    cfg: Problem,
    tags: dict[str, str],
    exp_key: str,
    harness_fn,
    *,
    make_ic,
    make_inputs,
    domain_extent: float,
    runs=None,
    _optim_loop=_drag_opt_adam_loop,
    _supports_partial: bool = True,
    **overrides,
) -> dict:
    """Shared body for ``run_drag_opt`` and its L-BFGS variant.

    ``_optim_loop`` selects the optimiser via the same contract used by
    :func:`_run_topopt_impl`: it returns ``(profile, profile_history, entry)``
    where ``entry`` is the fully-formed ``by_solver[name]`` dict.

    ``_supports_partial`` toggles the result_partial.json checkpointing the
    Adam variant relies on for long PICT runs. The L-BFGS variant doesn't use
    it, so the impl skips constructing the partial-write callback and lock.
    """
    if not runs:
        raise NotImplementedError(
            f"_run_drag_opt_impl requires runs= payload for {exp_key!r} "
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
        flow_penalty_weight = optim_cfg.get("flow_penalty_weight", 50.0)
        snap_interval = int(optim_cfg.get("snap_interval", 0))
        U_mean = float(phys.get("U_mean", 0.5))
        run_name = run.get("name", ic_name)
        ic_subdir = run_name if n_runs > 1 else ""
        gpu_ids = overrides.get("gpu_ids")

        # Forward only knobs that the YAML sets; each loop carries its own
        # default so Adam's max_iters=150 vs BFGS's max_iters=50 stay correct.
        loop_kwargs = {
            k: optim_cfg[k] for k in ("lr", "max_iters", "patience") if k in optim_cfg
        }

        profile_init = jnp.array(make_ic[ic_name](L=domain_extent, seed=seed, **phys))

        by_solver: dict = {}
        profile_snaps: dict = {}
        profile_histories: dict = {}
        # Velocity fields per solver, shape (N, N, 1, 2).
        flow_init_snaps: dict = {}
        flow_snaps: dict = {}

        # Partial-checkpoint plumbing (Adam path only). Adam mutates ``by_solver``
        # in-place inside its loop and calls ``write_partial`` periodically; the
        # L-BFGS loop ignores both arguments via its signature.
        write_partial = None
        if _supports_partial:
            partial_out_dir = _drag_opt_out_dir(
                cfg, ic_subdir, bool(overrides.get("debug")), exp_key
            )
            _partial_lock = threading.Lock()

            def write_partial() -> None:
                if not by_solver:
                    return
                payload = {
                    "by_solver": by_solver,
                    "run_name": run_name,
                    "U_mean": U_mean,
                    "params": run,
                }
                with _partial_lock:
                    save_json(payload, partial_out_dir / "result_partial.json")

        def _drag_opt_work(name: str, t) -> None:
            vel0 = _drag_capture_flow(
                name,
                t,
                profile_init,
                phys,
                make_inputs=make_inputs,
                domain_extent=domain_extent,
            )
            if vel0 is not None:
                flow_init_snaps[name] = vel0

            profile, profile_history, entry = _optim_loop(
                name,
                t,
                profile_init,
                phys,
                flow_penalty_weight=flow_penalty_weight,
                snap_interval=snap_interval,
                U_mean=U_mean,
                by_solver=by_solver,
                write_partial=write_partial,
                make_inputs=make_inputs,
                domain_extent=domain_extent,
                **loop_kwargs,
            )

            profile_snaps[name] = np.array(profile)
            if profile_history:
                profile_histories[name] = np.asarray(profile_history)
            by_solver[name] = entry
            if write_partial is not None:
                write_partial()
            vel = _drag_capture_flow(
                name,
                t,
                profile,
                phys,
                make_inputs=make_inputs,
                domain_extent=domain_extent,
            )
            if vel is not None:
                flow_snaps[name] = vel

        _drag_exp = f"{exp_key}/{run_name}" if run_name else exp_key
        drag_opt_solvers = active_differentiable_solvers(cfg, "optimization", _drag_exp)
        _wall_times = per_solver_loop(
            cfg,
            tags,
            drag_opt_solvers,
            _drag_opt_work,
            gpu_ids=gpu_ids,
            print_done=False,
        )

        out_dir = _drag_opt_out_dir(
            cfg, ic_subdir, bool(overrides.get("debug")), exp_key
        )
        result = {
            "by_solver": by_solver,
            "run_name": run_name,
            "U_mean": U_mean,
            "params": run,
        }
        if not by_solver:
            from mosaic.benchmarks.core.console import print_warn

            print_warn(
                f"{harness_fn.__name__}: by_solver is empty (all solvers excluded or "
                f"skipped) — skipping result.json save to preserve existing data"
            )
            if n_runs > 1:
                all_results[run_name] = result
            else:
                all_results = result
            continue
        save_experiment(
            result,
            out_dir,
            cfg=cfg,
            harness_fn=harness_fn,
            wall_time_s=_wall_times,
        )
        _merge_drag_profiles_npz(
            out_dir, profile_init, profile_snaps, profile_histories
        )
        _merge_drag_flow_fields_npz(out_dir, flow_init_snaps, flow_snaps)
        if n_runs > 1:
            all_results[run_name] = result
        else:
            all_results = result

    return all_results


def run_drag_opt(
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
    exp_key: str = "drag_opt",
    **overrides,
) -> dict:
    """Inflow profile optimisation: minimise drag on an embedded obstacle.

    Optimises the ``inflow_profile`` input field (1-D inlet velocity u_x(y)) to
    minimise the scalar ``drag`` output. A flow-rate conservation penalty is
    added to prevent the optimiser from trivially reducing drag by zeroing the
    inflow: L = drag + flow_penalty_weight * (mean(profile) - U_mean)².

    ``optimizer`` selects the inner optimiser:

      * ``"adam"`` — vanilla Adam (default).
      * ``"bfgs"`` — L-BFGS with zoom line-search.

    Problem-semantics state is passed explicitly. ``error_fn`` and
    ``output_key`` are accepted for signature parity with the other public
    harnesses but unused (drag is read from a fixed ``"drag"`` output and the
    loss is built in-line from ``mean(profile)``).

    Each run dict in ``runs`` must contain:
        name: str               — used as result subdir when multiple runs present
        ic: {name, seed}        — IC generator returning 1-D profile, shape (N,)
        physics: {N, nu, dt, steps, domain_extent, U_mean, obstacle, ...}
        optim: {lr, max_iters, patience, flow_penalty_weight}
    """
    del error_fn, output_key  # unused by drag_opt; kept for parity
    optim_loop = _drag_opt_lbfgs_loop if optimizer == "bfgs" else _drag_opt_adam_loop
    supports_partial = optimizer != "bfgs"
    return _run_drag_opt_impl(
        cfg,
        tags,
        exp_key,
        run_drag_opt,
        runs=runs,
        _optim_loop=optim_loop,
        _supports_partial=supports_partial,
        make_ic=make_ic,
        make_inputs=make_inputs,
        domain_extent=domain_extent,
        **overrides,
    )
