# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drag-optimization kernel and helpers for the navier-stokes grid problem.

This module hosts the ``drag_opt`` kernel and its private helpers, which
optimize the inlet velocity profile of the DFG 2D-1 channel
(:mod:`.cylinder`) to minimize the drag on its cylinder. The code is
ns-grid-specific (drag is a fluid-flow quantity tied to this problem's
tesseract outputs), so it lives next to the problem rather than in
``shared/optimization.py``.

The flow rate is held at the benchmark's. The solver sees the optimized
profile projected onto non-negative profiles of mean speed Ū and peak at most
the benchmark's U_m (:func:`_project`). Drag is reported as the coefficient
Cd = 2 F_x / (Ū² D), and the loss is Cd / Cd₀ with Cd₀ that of the
benchmark's parabolic profile.

The inner Adam loop, ``_run_optim``, lives in ``shared/optimization.py``.
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_jax import apply_tesseract

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.core.io import (
    PartialResultWriter,
    experiment_dir,
    results_dir,
    save_npz_merged,
)
from mosaic.benchmarks.core.utils import active_differentiable_solvers
from mosaic.benchmarks.problems.shared.optimization import _run_optim

from . import cylinder

logger = logging.getLogger(__name__)

_SUITE = "optimization"


# Largest initial Cd error, relative to the reference, for which the drag
# optimization still starts from the benchmark flow. Further off, the solver
# is solving a different problem and its optimum would mean nothing.
CD0_TOLERANCE = 0.15


# Upper bound on the inlet speed: the peak of the benchmark's own parabola.
# With the flow rate fixed at Ū = 2 U_m / 3, no admissible profile leaves more
# than a third of the inlet at rest. Looser bounds admit narrow jets that
# bypass the cylinder and drive a recirculation onto it, which settles slowly,
# can push Cd below zero, and makes the optimum incomparable across solvers.
U_BOUND = cylinder.U_MAX


def _project(v: Any) -> Any:
    """Euclidean projection of ``v`` onto {0 ≤ u ≤ U_BOUND, mean(u) = Ū}.

    The projection is ``clip(v + c, 0, U_BOUND)`` with the shift ``c`` found by
    bisection. Its gradient is the exact Jacobian of the projection. Entries
    at a bound get none, and the free entries lose their mean, since ``c``
    moves to restore the flow rate.
    """
    vs = jax.lax.stop_gradient(v)

    def bisect(_: int, bounds: tuple) -> tuple:
        lo, hi = bounds
        c = 0.5 * (lo + hi)
        above = jnp.mean(jnp.clip(vs + c, 0.0, U_BOUND)) > cylinder.U_MEAN
        return jnp.where(above, lo, c), jnp.where(above, c, hi)

    lo, hi = jax.lax.fori_loop(0, 60, bisect, (-jnp.max(vs), U_BOUND - jnp.min(vs)))
    c = 0.5 * (lo + hi)
    free = (vs + c > 0.0) & (vs + c < U_BOUND)
    n_free = jnp.maximum(jnp.sum(free), 1)
    shift = c - jnp.sum(jnp.where(free, v - vs, 0.0)) / n_free
    return jnp.clip(v + shift, 0.0, U_BOUND)


def _drag_forward(
    t: Any, profile: Any, phys: dict, *, name: str, make_inputs: Any
) -> tuple[Any, Any]:
    """Run the solver on the projected inlet ``profile`` and return ``(Cd, velocity)``."""
    inp = make_inputs(name, _project(profile), **phys)
    out = apply_tesseract(t, inp)
    force = out.get("body_force")
    if force is None:
        raise RuntimeError(
            f"Solver '{name}' did not return 'body_force'; obstacle runs must "
            "return the force history on the obstacle."
        )
    return cylinder.drag_coefficient(force), out.get("result")


def _drag_capture_flow(
    t: Any, profile: Any, phys: dict, *, name: str, make_inputs: Any
) -> tuple[float, np.ndarray | None]:
    """Float-valued :func:`_drag_forward` that returns ``(nan, None)`` on failure."""
    try:
        cd, vel = _drag_forward(t, profile, phys, name=name, make_inputs=make_inputs)
        return float(cd), (np.array(vel) if vel is not None else None)
    except Exception:
        logger.debug("drag-capture forward for %s failed", name, exc_info=True)
    return float("nan"), None


def _drag_build_solver_entry(
    drags: list,
    grad_norms: list,
    n_iters_total: int,
    converged: bool,
    in_progress: bool = False,
    error: str | None = None,
) -> dict:
    """Assemble a single solver's drag-opt result dict (used for partial + final).

    ``drags`` is the drag coefficient Cd at each iteration.
    """
    entry = {
        "drags": drags,
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


def _merge_drag_profiles_npz(
    out_dir: Any,
    profile_init: Any,
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
    out_dir: Any,
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


# ── drag_opt: Adam loop ──────────────────────────────────────────────────────


def _drag_opt_adam_loop(
    name: str,
    t: Any,
    profile_init: Any,
    phys: dict,
    *,
    cd0: float,
    lr: float = 1e-2,
    max_iters: int = 100,
    patience: int = 10,
    min_rel_improvement: float = 1e-3,
    snap_interval: int = 0,
    by_solver: dict,
    write_partial: Any,
    make_inputs: Any,
) -> tuple[jax.Array, list, dict]:
    """Adam loop for drag_opt; supports per-iter partial-checkpoint flushes.

    Thin wrapper around :func:`_run_optim` with ``has_aux=True``. The loss
    closure returns ``(Cd / cd0, {"cd": Cd})`` so the per-iter drag
    coefficient flows through the shared ``aux_history`` plumbing. Partial-
    checkpoint flushes happen via the ``log_fn`` callback every 10 iterations.

    Returns ``(profile_final, profile_history, by_solver_entry)``.
    """

    def loss_fn(p: Any, _t: Any = t) -> Any:
        cd, _ = _drag_forward(_t, p, phys, name=name, make_inputs=make_inputs)
        return cd / cd0, {"cd": cd}

    aux_history: dict = {}
    profile_history: list = []

    def _log_iter(_i: Any, _loss_val: Any) -> None:
        # ``_run_optim`` invokes us every ``log_interval`` iters. Rebuild
        # the cross-solver shim entry from the live aux traces and flush a
        # partial result. Grad-norms are kept inside ``_run_optim``'s local
        # ``diag`` dict and only surfaced post-loop, so partial entries
        # written during the run carry an empty grad_norms list — the final
        # flush in the kernel replaces the entry with the converged one.
        drag_trace = list(aux_history.get("cd", []))
        by_solver[name] = _drag_build_solver_entry(
            drag_trace, [], len(drag_trace), converged=False, in_progress=True
        )
        write_partial()

    profile, losses, diag = _run_optim(
        loss_fn,
        profile_init,
        lr=lr,
        max_iters=max_iters,
        patience=patience,
        min_rel_improvement=min_rel_improvement,
        has_aux=True,
        aux_history=aux_history,
        clip_fn=_project,
        snap_interval=snap_interval,
        history=profile_history if snap_interval > 0 else None,
        log_fn=_log_iter,
        log_interval=10,
    )

    drags = list(aux_history.get("cd", []))
    grad_norms = list((diag or {}).get("grad_norms") or [])
    n_iters = len(losses)
    entry = _drag_build_solver_entry(
        drags, grad_norms, n_iters, converged=n_iters < max_iters
    )
    return profile, profile_history, entry


# ── kernel + aggregate ───────────────────────────────────────────────────────


def _drag_opt_aggregate(
    by_solver: Any,
    *,
    run: Any,
    cfg: Any,
    out_dir: Any,
    snapshots: Any,
    shared_extras: Any,
    **_kw: Any,
) -> dict:
    """Aggregate per-solver drag_opt output → result dict + profiles/flow npz.

    The framework hands us:
      * ``by_solver`` — ``{name: <solver entry dict>}`` from each kernel's
        ``metrics`` return.
      * ``snapshots`` — ``{name: {"profile_final:": arr,
        "profile_history:": arr, "flow_initial:": arr, "flow_final:": arr}}``
        from each kernel's ``snapshots`` return. Empty-suffix prefixed keys
        carry the ``"prefix:"`` form used by
        :func:`mosaic.benchmarks.core.experiment._absorb`.
      * ``shared_extras`` — ``{"profile_initial": arr}`` shared across solvers.

    Writes ``profiles.npz`` (initial profile + per-solver final/history) and
    ``flow_fields.npz`` (per-solver flow snapshots) via the existing
    merge-aware helpers so single-solver reruns don't wipe peer entries.
    Returns the canonical ``{by_solver, run_name, U_mean, params}`` result
    dict (writing ``result.json`` is the framework's job).
    """
    # Strip the framework's ``"prefix:"`` suffix convention; sweep_mode="none"
    # so the suffix after ``":"`` is always empty.
    profile_snaps: dict[str, np.ndarray] = {}
    profile_histories: dict[str, np.ndarray] = {}
    flow_init_snaps: dict[str, np.ndarray] = {}
    flow_snaps: dict[str, np.ndarray] = {}
    for name, suf_map in snapshots.items():
        for k, arr in suf_map.items():
            prefix = k.split(":", 1)[0]
            if prefix == "profile_final":
                profile_snaps[name] = np.asarray(arr)
            elif prefix == "profile_history":
                profile_histories[name] = np.asarray(arr)
            elif prefix == "flow_initial":
                flow_init_snaps[name] = np.asarray(arr)
            elif prefix == "flow_final":
                flow_snaps[name] = np.asarray(arr)

    profile_init = shared_extras.get("profile_initial")
    U_mean = cylinder.U_MEAN
    run_name = run.get("name", "")

    from mosaic.benchmarks.core.experiment import (
        _build_result_envelope,
        _flatten_by_solver,
    )

    if not by_solver:
        from mosaic.benchmarks.core.console import print_warn

        print_warn(
            "drag_opt: by_solver is empty (all solvers excluded or "
            "skipped) — skipping NPZ writes to preserve existing data"
        )
        return _build_result_envelope(
            cfg=cfg,
            suite=_kw.get("suite", "optimization"),
            exp_key=_kw.get("exp_key", "drag_opt"),
            run=run,
            sweep_key=None,
            sweep_values=None,
            results=_flatten_by_solver(by_solver, None),
            extras={"run_name": run_name, "U_mean": U_mean},
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if profile_init is not None:
        _merge_drag_profiles_npz(
            out_dir, profile_init, profile_snaps, profile_histories
        )
    _merge_drag_flow_fields_npz(out_dir, flow_init_snaps, flow_snaps)

    return _build_result_envelope(
        cfg=cfg,
        suite=_kw.get("suite", "optimization"),
        exp_key=_kw.get("exp_key", "drag_opt"),
        run=run,
        sweep_key=None,
        sweep_values=None,
        results=_flatten_by_solver(by_solver, None),
        extras={"run_name": run_name, "U_mean": U_mean},
    )


@kernel(
    sweep_mode="none",
    selector_fn=active_differentiable_solvers,
    aggregate_fn=_drag_opt_aggregate,
    snapshot_filename="profiles.npz",
    snapshot_prefixes=(
        "profile_final",
        "profile_history",
        "flow_initial",
        "flow_final",
    ),
)
def drag_opt(t: Any, ctx: KernelContext) -> dict:
    """One solver's full inflow-profile drag optimization.

    Starts from the benchmark's parabolic inlet profile and checks that the
    solver's drag coefficient for it, Cd₀, is within :data:`CD0_TOLERANCE`
    of the reference. Otherwise it stops there and records the failure. Then
    runs Adam on the loss ``Cd / Cd₀``, captures the initial and final flow
    fields, and hands the per-solver entry + snapshots back to the framework.

    Every 10 iterations the kernel merges its current entry into
    ``result_partial.json`` (under a FileLock so peer solvers running on
    other workers don't clobber it).
    """
    run = ctx.run
    phys = run.get("physics", {})
    optim_cfg = run.get("optim", {})
    snap_interval = int(optim_cfg.get("snap_interval", 0))

    ic_cfg = run.get("ic", {})
    ic_name = ic_cfg.get("name", next(iter(ctx.cfg.make_ic)))
    profile_init = jnp.array(ctx.cfg.make_ic[ic_name](seed=ctx.seed, **phys))

    # ── Initial flow: the benchmark itself ─────────────────────────────────
    cd0, vel0 = _drag_capture_flow(
        t, profile_init, phys, name=ctx.name, make_inputs=ctx.make_inputs
    )
    cd0_err = abs(cd0 - cylinder.CD_REF) / cylinder.CD_REF
    if not cd0_err <= CD0_TOLERANCE:
        return {
            "metrics": {
                "valid": False,
                "error": (
                    f"initial Cd {cd0:.4g} is not within {CD0_TOLERANCE:.0%} of "
                    f"the reference {cylinder.CD_REF}; not optimizing"
                ),
            }
        }

    # Forward only the optimizer knobs the config sets, so the loop's own
    # defaults fill in the rest.
    loop_kwargs = {
        k: optim_cfg[k]
        for k in ("lr", "max_iters", "patience", "min_rel_improvement")
        if k in optim_cfg
    }

    # ── Partial-result checkpointing ───────────────────────────────────────
    # Resolve out_dir the same way the framework does so the partial JSON
    # lands next to the final ``result.json``.
    partial_out_dir = experiment_dir(
        results_dir(),
        ctx.cfg.name,
        _SUITE,
        "drag_opt",
        suffix="_debug" if run.get("debug") else "",
    )

    # The Adam inner loop mutates a tiny ``by_solver`` dict in place and
    # invokes ``write_partial()`` every 10 iters. We keep that contract
    # by giving the loop a one-key dict shim that forwards through to the
    # cross-solver FileLock-backed writer.
    by_solver_shim: dict = {}
    partial_writer = PartialResultWriter(
        partial_out_dir,
        base_payload={
            "run_name": run.get("name", ""),
            "U_mean": cylinder.U_MEAN,
            "params": run,
        },
    )

    def write_partial() -> None:
        partial_writer.write(ctx.name, by_solver_shim.get(ctx.name))

    # ── Optimisation loop ──────────────────────────────────────────────────
    profile, profile_history, entry = _drag_opt_adam_loop(
        ctx.name,
        t,
        profile_init,
        phys,
        cd0=cd0,
        snap_interval=snap_interval,
        by_solver=by_solver_shim,
        write_partial=write_partial,
        make_inputs=ctx.make_inputs,
        **loop_kwargs,
    )

    # ── Final flow field capture ───────────────────────────────────────────
    cd_final, vel_final = _drag_capture_flow(
        t, profile, phys, name=ctx.name, make_inputs=ctx.make_inputs
    )
    if np.isfinite(cd_final):
        # ``drags[k]`` is Cd after k updates. The final profile adds one more.
        entry = _drag_build_solver_entry(
            [*entry["drags"], cd_final],
            entry["grad_norms"],
            entry["n_iters"],
            converged=entry["converged"],
        )
    # Final flush — replace the in-progress entry with the converged one.
    by_solver_shim[ctx.name] = entry
    write_partial()

    snaps: dict[str, np.ndarray] = {"profile_final": np.asarray(_project(profile))}
    if profile_history:
        snaps["profile_history"] = np.asarray([_project(p) for p in profile_history])
    if vel0 is not None:
        snaps["flow_initial"] = vel0
    if vel_final is not None:
        snaps["flow_final"] = vel_final

    return {
        "metrics": entry,
        "snapshots": snaps,
        "shared": {"profile_initial": np.asarray(_project(profile_init))},
    }
