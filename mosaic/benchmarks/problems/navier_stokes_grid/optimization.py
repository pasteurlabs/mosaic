# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drag-optimisation kernel and helpers for the navier-stokes grid problem.

This module hosts the ``drag_opt`` kernel and its private helpers, which
optimise an inflow velocity profile to minimise drag on an embedded
obstacle. The code is ns-grid-specific (drag is a fluid-flow quantity tied
to this problem's tesseract outputs), so it lives next to the problem
rather than in ``shared/optimization.py``.

The two inner optimisation primitives ``_run_optim`` (Adam) and
``_run_lbfgs`` (L-BFGS with zoom line-search) live in
``shared/optimization.py``; both are imported here.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.core.io import (
    PartialResultWriter,
    experiment_dir,
    results_dir,
    save_npz_merged,
)

# JAX-traced loss_fn closures capture this reference at trace time;
# using the tracer-aware wrapper ensures primitive binding sees the
# active trace.
from mosaic.benchmarks.core.tracer_apply import apply_tesseract
from mosaic.benchmarks.core.utils import active_differentiable_solvers
from mosaic.benchmarks.problems.shared.optimization import _run_lbfgs, _run_optim

_SUITE = "optimization"


def _drag_capture_flow(
    name: str,
    t: Any,
    profile: Any,
    phys: dict,
    *,
    make_inputs: Any,
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


def _drag_bfgs_final_drag(
    name: str,
    t: Any,
    profile: Any,
    phys: dict,
    losses: list,
    *,
    make_inputs: Any,
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


# ── drag_opt: shared body, Adam vs L-BFGS as an argument ─────────────────────


def _drag_opt_adam_loop(
    name: str,
    t: Any,
    profile_init: Any,
    phys: dict,
    *,
    lr: float = 1e-3,
    max_iters: int = 150,
    patience: int = 30,
    flow_penalty_weight: float = 50.0,
    snap_interval: int = 0,
    U_mean: float = 0.5,
    by_solver: dict,
    write_partial: Any,
    make_inputs: Any,
    domain_extent: float,
) -> tuple[jax.Array, list, dict]:
    """Adam loop for drag_opt; supports per-iter partial-checkpoint flushes.

    Thin wrapper around :func:`_run_optim` with ``has_aux=True``. The loss
    closure returns ``(|drag| + flow_penalty, {"drag": ..., "flow_rate": ...})``
    so the per-iter drag and flow-rate traces flow through the shared
    ``aux_history`` plumbing. Partial-checkpoint flushes happen via the
    ``log_fn`` callback every 10 iterations.

    Returns ``(profile_final, profile_history, by_solver_entry)``.
    """

    def loss_fn(p: Any, _t: Any = t) -> Any:
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
        return (
            jnp.abs(jnp.squeeze(drag_val)) + flow_penalty,
            {"drag": jnp.squeeze(drag_val), "flow_rate": jnp.mean(p)},
        )

    aux_history: dict = {}
    profile_history: list = []

    def _log_iter(_i: Any, _loss_val: Any) -> None:
        # ``_run_optim`` invokes us every ``log_interval`` iters. Rebuild
        # the cross-solver shim entry from the live aux traces and flush a
        # partial result. Grad-norms are kept inside ``_run_optim``'s local
        # ``diag`` dict and only surfaced post-loop, so partial entries
        # written during the run carry an empty grad_norms list — the final
        # flush in the kernel replaces the entry with the converged one.
        drag_trace = list(aux_history.get("drag", []))
        by_solver[name] = _drag_build_solver_entry(
            drag_trace,
            list(aux_history.get("flow_rate", [])),
            [],
            len(drag_trace),
            converged=False,
            in_progress=True,
        )
        write_partial()

    profile, losses, diag = _run_optim(
        loss_fn,
        profile_init,
        lr=lr,
        max_iters=max_iters,
        patience=patience,
        has_aux=True,
        aux_history=aux_history,
        clip_fn=lambda p: jnp.clip(p, 0.0, 3.0 * U_mean),
        snap_interval=snap_interval,
        history=profile_history if snap_interval > 0 else None,
        log_fn=_log_iter,
        log_interval=10,
    )

    drags = list(aux_history.get("drag", []))
    flow_rates = list(aux_history.get("flow_rate", []))
    grad_norms = list((diag or {}).get("grad_norms") or [])
    n_iters = len(losses)
    entry = _drag_build_solver_entry(
        drags,
        flow_rates,
        grad_norms,
        n_iters,
        converged=n_iters < max_iters,
    )
    return profile, profile_history, entry


def _drag_opt_lbfgs_loop(
    name: str,
    t: Any,
    profile_init: Any,
    phys: dict,
    *,
    max_iters: int = 50,
    flow_penalty_weight: float = 50.0,
    snap_interval: int = 0,
    U_mean: float = 0.5,
    make_inputs: Any,
    domain_extent: float,
    **_unused: Any,  # absorb Adam-only kwargs (by_solver, write_partial, lr, patience)
) -> tuple[jax.Array, list, dict]:
    """L-BFGS loop for drag_opt; no partial checkpointing.

    Returns ``(profile_final, profile_history, by_solver_entry)``. Computes
    ``final_drag`` via one extra forward pass after convergence so the entry
    reflects the post-clip drag rather than the last loss (which includes the
    flow penalty).
    """
    del _unused  # signature-parity slot, intentionally discarded
    profile_history: list = []

    def loss_fn(p: Any, _t: Any = t) -> Any:
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

    def _log_iter(i: Any, loss_val: Any, _n: Any = name) -> None:
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


# ── kernel + aggregate ───────────────────────────────────────────────────────


def _drag_opt_aggregate(
    by_solver: Any,
    *,
    run: Any,
    cfg: Any,
    out_dir: Any,
    snapshots: Any,
    shared_extras: Any,
    **_: Any,
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
    del cfg  # not needed; out_dir already resolved
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
    U_mean = float(run.get("physics", {}).get("U_mean", 0.5))
    run_name = run.get("name", "")

    if not by_solver:
        from mosaic.benchmarks.core.console import print_warn

        print_warn(
            "drag_opt: by_solver is empty (all solvers excluded or "
            "skipped) — skipping NPZ writes to preserve existing data"
        )
        return {
            "by_solver": by_solver,
            "run_name": run_name,
            "U_mean": U_mean,
            "params": run,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    if profile_init is not None:
        _merge_drag_profiles_npz(
            out_dir, profile_init, profile_snaps, profile_histories
        )
    _merge_drag_flow_fields_npz(out_dir, flow_init_snaps, flow_snaps)

    return {
        "by_solver": by_solver,
        "run_name": run_name,
        "U_mean": U_mean,
        "params": run,
    }


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
    """One solver's full inflow-profile drag optimisation.

    Builds the initial 1-D inlet velocity profile from the IC factory,
    optionally captures the initial flow field, runs Adam or L-BFGS
    (selected via ``run["optimizer"]``) to minimise drag subject to a
    soft flow-rate conservation penalty, captures the final flow field,
    and hands the per-solver entry + snapshots back to the framework.

    Adam supports partial-result checkpointing — every 10 iterations the
    kernel merges its current entry into ``result_partial.json`` (under a
    FileLock so peer solvers running on other workers don't clobber it).
    L-BFGS doesn't checkpoint.

    Loss: ``L = |drag| + flow_penalty_weight * (mean(profile) - U_mean)²``.
    The penalty stops the optimiser from trivially zeroing the inflow.
    """
    run = ctx.run
    phys_run = run.get("physics", {})
    optim_cfg = run.get("optim", {})
    optimizer_name = run.get("optimizer", "adam")

    flow_penalty_weight = optim_cfg.get("flow_penalty_weight", 50.0)
    snap_interval = int(optim_cfg.get("snap_interval", 0))
    U_mean = float(phys_run.get("U_mean", 0.5))

    ic_cfg = run.get("ic", {})
    ic_name = ic_cfg.get("name", next(iter(ctx.cfg.make_ic)))
    profile_init = jnp.array(
        ctx.cfg.make_ic[ic_name](L=ctx.domain_extent, seed=ctx.seed, **phys_run)
    )

    # phys for forward / loss closures: physics fields + canonical
    # ``domain_extent`` (kept consistent with the old harness behaviour).
    phys = {**phys_run}
    phys.setdefault("domain_extent", ctx.domain_extent)

    # Forward only the optimiser knobs the YAML sets; each loop carries
    # its own default so Adam's max_iters=500 vs BFGS's max_iters=50 stay
    # correct without the caller knowing which optimiser is in use.
    loop_kwargs = {
        k: optim_cfg[k] for k in ("lr", "max_iters", "patience") if k in optim_cfg
    }
    optim_loop = (
        _drag_opt_lbfgs_loop if optimizer_name == "bfgs" else _drag_opt_adam_loop
    )
    supports_partial = optimizer_name != "bfgs"

    # ── Partial-result checkpointing (Adam only) ───────────────────────────
    # Resolve out_dir the same way the framework does so the partial JSON
    # lands next to the final ``result.json``. Both drag_opt registrations
    # in this problem's config follow the ``optimization/drag_opt[_bfgs]``
    # naming convention, so the exp_name is implied by the optimiser flag.
    run_name = run.get("name", "")
    debug_flag = bool(run.get("debug"))
    exp_name = "drag_opt_bfgs" if optimizer_name == "bfgs" else "drag_opt"
    partial_out_dir = experiment_dir(
        results_dir(),
        ctx.cfg.name,
        _SUITE,
        exp_name,
        suffix="_debug" if debug_flag else "",
    )

    # The Adam inner loop mutates a tiny ``by_solver`` dict in place and
    # invokes ``write_partial()`` every 10 iters. We keep that contract
    # by giving the loop a one-key dict shim that forwards through to the
    # cross-solver FileLock-backed writer.
    by_solver_shim: dict = {}
    if supports_partial:
        partial_writer = PartialResultWriter(
            partial_out_dir,
            base_payload={"run_name": run_name, "U_mean": U_mean, "params": run},
        )

        def write_partial() -> None:
            partial_writer.write(ctx.name, by_solver_shim.get(ctx.name))

    else:

        def write_partial() -> None:
            return

    # ── Initial flow field capture ─────────────────────────────────────────
    vel0 = _drag_capture_flow(
        ctx.name,
        t,
        profile_init,
        phys,
        make_inputs=ctx.make_inputs,
        domain_extent=ctx.domain_extent,
    )

    # ── Optimisation loop ──────────────────────────────────────────────────
    profile, profile_history, entry = optim_loop(
        ctx.name,
        t,
        profile_init,
        phys,
        flow_penalty_weight=flow_penalty_weight,
        snap_interval=snap_interval,
        U_mean=U_mean,
        by_solver=by_solver_shim,
        write_partial=write_partial,
        make_inputs=ctx.make_inputs,
        domain_extent=ctx.domain_extent,
        **loop_kwargs,
    )
    if supports_partial:
        # Final flush — replace the in-progress entry with the converged one.
        by_solver_shim[ctx.name] = entry
        write_partial()

    # ── Final flow field capture ───────────────────────────────────────────
    vel_final = _drag_capture_flow(
        ctx.name,
        t,
        profile,
        phys,
        make_inputs=ctx.make_inputs,
        domain_extent=ctx.domain_extent,
    )

    snaps: dict[str, np.ndarray] = {"profile_final": np.asarray(profile)}
    if profile_history:
        snaps["profile_history"] = np.asarray(profile_history)
    if vel0 is not None:
        snaps["flow_initial"] = vel0
    if vel_final is not None:
        snaps["flow_final"] = vel_final

    return {
        "metrics": entry,
        "snapshots": snaps,
        "shared": {"profile_initial": np.asarray(profile_init)},
    }
