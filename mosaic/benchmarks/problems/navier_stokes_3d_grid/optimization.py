"""IC-recovery runner and helpers for the ns-3d-grid problem.

This module hosts the gradient-descent / L-BFGS based initial-condition
recovery pipeline.  The runner ``run_recovery`` (plus its private helpers
and the ``_RecoveryRunCtx`` dataclass) used to live in
``mosaic.benchmarks.problems.shared.optimization`` but is consumed only by
the navier-stokes 3D grid problem, so it was moved here.

The inner optimiser primitives ``_run_optim`` (Adam with patience-based
early stopping) and ``_run_lbfgs`` (L-BFGS with zoom line-search) remain
in ``shared/`` since they are reused by topology-optimisation and other
suites; they are imported below.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import (
    experiment_dir,
    results_dir,
    save_experiment,
    save_field_snapshots_npz,
    save_json,
)
from mosaic.benchmarks.core.runner import per_solver_loop

# JAX-traced loss_fn closures capture this reference at trace time;
# using the tracer-aware wrapper ensures primitive binding sees the
# active trace.
from mosaic.benchmarks.core.tracer_apply import apply_tesseract
from mosaic.benchmarks.core.utils import (
    active_differentiable_solvers,
    extract_runs,
    is_valid,
    iter_runs,
)
from mosaic.benchmarks.problems.shared.optimization import _run_lbfgs, _run_optim

_SUITE = "optimization"


# ── Divergence-free helpers ───────────────────────────────────────────────────


def _is_velocity_field(u: np.ndarray) -> bool:
    """True when u looks like a periodic velocity field (last dim 2 or 3)."""
    return u.shape[-1] in (2, 3)


def _max_divergence(u: np.ndarray, domain_extent: float) -> float:
    """Max |∇·u| computed spectrally (exact for periodic fields).

    Uses the same FFT machinery as ``_project_divergence_free``:
    ``∇·u = IFFT(Σ_i  i k_i  û_i)``.  For a spectrally divergence-free field
    this returns near-machine-precision values (~1e-14), unlike central
    finite differences which give O(0.1) residuals at high wavenumbers.
    """
    nd = u.shape[-1]
    squeeze = u.ndim > nd + 1
    v = u.reshape((*u.shape[:nd], nd)) if squeeze else u

    N = v.shape[0]
    k1d = np.fft.fftfreq(N) * N * (2.0 * np.pi / domain_extent)
    grids = np.meshgrid(*([k1d] * nd), indexing="ij")

    spatial_axes = tuple(range(nd))
    # Upcast to float64 before FFT: avoids complex64 precision loss when input is float32.
    v_hat = np.fft.fftn(v.astype(np.float64), axes=spatial_axes)

    div_hat = sum(1j * grids[i] * v_hat[..., i] for i in range(nd))
    div = np.fft.ifftn(div_hat, axes=spatial_axes).real
    return float(np.max(np.abs(div)))


def _project_divergence_free(u: np.ndarray, domain_extent: float) -> np.ndarray:
    """Helmholtz-project a periodic velocity field onto ∇·u = 0.

    Spectral projection: û_df_i = û_i − k_i (k·û) / |k|²
    Handles both 2D (N, N, 1, 2) and 3D (N, N, N, 3) shapes.
    """
    nd = u.shape[-1]
    squeeze = u.ndim > nd + 1  # True for (N,N,1,2); False for (N,N,N,3)
    v = u.reshape((*u.shape[:nd], nd)) if squeeze else u

    N = v.shape[0]
    # Physical wavenumbers for a periodic box of size domain_extent.
    k1d = np.fft.fftfreq(N) * N * (2.0 * np.pi / domain_extent)
    grids = np.meshgrid(*([k1d] * nd), indexing="ij")  # nd arrays, each (N,...,N)
    k2 = sum(k**2 for k in grids)
    k2_safe = np.where(k2 == 0.0, 1.0, k2)  # avoid ÷0 at k=0

    spatial_axes = tuple(range(nd))
    v_hat = np.fft.fftn(v, axes=spatial_axes)  # (..., nd), complex

    k_dot_u = sum(grids[i] * v_hat[..., i] for i in range(nd))  # scalar field

    v_hat_df = np.stack(
        [v_hat[..., i] - grids[i] * k_dot_u / k2_safe for i in range(nd)],
        axis=-1,
    )

    v_df = np.fft.ifftn(v_hat_df, axes=spatial_axes).real
    v_df = v_df.reshape(u.shape) if squeeze else v_df
    return v_df.astype(u.dtype)


def _project_ic_with_log(
    raw: jax.Array,
    label: str,
    is_vel: bool,
    domain_extent: float,
    console,
) -> jax.Array:
    """Project to divergence-free and log; no-op for non-velocity fields."""
    if not is_vel:
        return raw
    raw_np = np.asarray(raw)
    div_before = _max_divergence(raw_np, domain_extent)
    projected = _project_divergence_free(raw_np, domain_extent)
    div_after = _max_divergence(projected, domain_extent)
    console.print(
        f"  [dim]{label}[/dim]  "
        f"max|∇·u| {div_before:.2e} → {div_after:.2e} (after projection)"
    )
    return jnp.asarray(projected)


def _build_per_seed_ics(
    ic_name: str,
    ic_seeds: list[int],
    phys: dict,
    perturb_sigma: float,
    ic_init_type: str,
    console,
    *,
    make_ic,
    domain_extent: float,
) -> tuple[dict, dict, dict]:
    """Build per-seed (true IC, initial IC, max divergence) dicts.

    Returns ``(ic_true_dict, ic_init_dict, max_div_dict)``. For velocity fields
    both true and perturbed ICs are Helmholtz-projected onto ∇·u = 0.
    """
    ic_true_dict: dict[int, jax.Array] = {}
    ic_init_dict: dict[int, jax.Array] = {}
    max_div_dict: dict[int, float | None] = {}

    for s in ic_seeds:
        ic_k = jnp.array(make_ic[ic_name](L=domain_extent, seed=s, **phys))
        is_vel_k = _is_velocity_field(np.asarray(ic_k))
        ic_k = _project_ic_with_log(
            ic_k, f"IC seed={s} (true)", is_vel_k, domain_extent, console
        )
        max_div_dict[s] = (
            _max_divergence(np.asarray(ic_k), domain_extent) if is_vel_k else None
        )
        ic_true_dict[s] = ic_k
        if ic_init_type == "zeros":
            ic_init_dict[s] = jnp.zeros_like(ic_k)
        else:
            noise_seed = s + 1000
            if is_vel_k:
                noise = jnp.array(
                    make_ic[ic_name](L=domain_extent, seed=noise_seed, **phys)
                )
                raw_init = ic_k + perturb_sigma * noise
            else:
                raw_init = ic_k + perturb_sigma * jax.random.normal(
                    jax.random.PRNGKey(noise_seed), ic_k.shape, dtype=jnp.float32
                )
            ic_init_dict[s] = _project_ic_with_log(
                raw_init,
                f"IC seed={s} (perturbed, σ={perturb_sigma})",
                is_vel_k,
                domain_extent,
                console,
            )
    return ic_true_dict, ic_init_dict, max_div_dict


def _build_sigma_perturbed_ics(
    ic_true: jax.Array,
    sweep_values: list,
    seed: int,
    is_vel: bool,
    console,
    *,
    domain_extent: float,
) -> dict:
    """For sigma sweep: build dict[sigma -> perturbed IC] (div-free projected)."""
    sigma_ics: dict = {}
    for sv in sweep_values:
        sigma_val = float(sv)
        key_sv = jax.random.fold_in(jax.random.PRNGKey(seed), int(sigma_val * 1000))
        raw = ic_true + sigma_val * jax.random.normal(
            key_sv, ic_true.shape, dtype=jnp.float32
        )
        sigma_ics[sv] = _project_ic_with_log(
            raw, f"σ={sigma_val}", is_vel, domain_extent, console
        )
    return sigma_ics


def _compute_targets_for_val(
    name: str,
    exp_key: str,
    val,
    sweep_key: str,
    ic_seeds: list[int],
    ic_true_dict: dict,
    phys: dict,
    t,
    *,
    make_inputs,
    output_key: str,
    domain_extent: float,
) -> dict:
    """Forward-solve each seed's true IC to produce target outputs.

    Skips seeds whose forward pass fails or produces invalid output.
    Returns ``{seed: target_array}``.
    """
    targets: dict[int, jax.Array] = {}
    for s in ic_seeds:
        try:
            inp_true = make_inputs(
                name,
                ic_true_dict[s],
                domain_extent=domain_extent,
                **{**phys, sweep_key: val},
            )
            tgt = apply_tesseract(t, inp_true)[output_key]
        except Exception as exc:
            from mosaic.benchmarks.core.console import print_warn

            print_warn(
                f"{name} {exp_key} target forward failed at {sweep_key}={val} ic_seed={s}: {exc}"
            )
            continue
        if is_valid(tgt):
            targets[s] = tgt
    return targets


def _aggregate_trial_results(
    trial_results: list[dict],
    primary_ic_err_hist: list[float],
    val,
    perturb_sigma: float,
    is_sigma_sweep: bool,
    is_multi_seed: bool,
    failure_threshold: float,
    max_div,
    ic_true,
) -> dict:
    """Aggregate per-seed trial results into a single by_sweep entry."""
    all_fice = [r["final_ic_error"] for r in trial_results]
    first = trial_results[0]
    first_diag = first.get("diag") or {}
    return {
        "errors": first["errors"],
        "grad_norms": first_diag.get("grad_norms"),
        "grad_divs": first_diag.get("grad_divs"),
        "ic_divs": first_diag.get("ic_divs"),
        "ic_error_history": primary_ic_err_hist,
        "ic_error_init": float(np.mean([r["ic_error_init"] for r in trial_results])),
        "final_ic_error": float(np.mean(all_fice)),
        "final_ic_error_std": float(np.std(all_fice)) if is_multi_seed else None,
        "final_ic_error_trials": all_fice if is_multi_seed else None,
        "final_ic_div": float(
            np.nanmean(
                [
                    r["final_ic_div"]
                    for r in trial_results
                    if r["final_ic_div"] is not None
                ]
            )
        )
        if any(r["final_ic_div"] is not None for r in trial_results)
        else None,
        "converged": np.mean(
            [r["final_ic_error"] < failure_threshold for r in trial_results]
        )
        > 0.5,
        "perturb_sigma": float(val) if is_sigma_sweep else perturb_sigma,
        "max_div_ic": max_div if ic_true.ndim == 4 else None,
        "n_trials": len(trial_results),
        "final_loss": float(
            np.mean([r["errors"][-1] for r in trial_results if r.get("errors")])
        ),
        "final_loss_trials": [r["errors"][-1] for r in trial_results if r.get("errors")]
        if is_multi_seed
        else None,
    }


def _build_recovery_visualization_stacks(
    name: str,
    all_ic_opts: dict,
    sweep_values: list,
    sweep_key: str,
    phys: dict,
    is_sigma_sweep: bool,
    sigma_ics: dict | None,
    ic_true,
    t,
    *,
    make_inputs,
    output_key: str,
    domain_extent: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Stack per-sigma IC/final-rec arrays plus per-sigma perturbed-forward.

    Returns ``(ic_stack, final_rec_stack, perturbed_stack)``; any may be ``None``.
    """
    ic_stack, fr_stack = [], []
    for v in sweep_values:
        io = all_ic_opts.get(v)
        if io is not None:
            ic_stack.append(np.asarray(io))
            kw = phys if is_sigma_sweep else {**phys, sweep_key: v}
            try:
                inp = make_inputs(name, io, domain_extent=domain_extent, **kw)
                fr_stack.append(np.asarray(apply_tesseract(t, inp)[output_key]))
            except Exception:
                fr_stack.append(np.zeros_like(np.asarray(io)))
        else:
            ic_stack.append(np.zeros_like(np.asarray(ic_true)))
            fr_stack.append(np.zeros_like(np.asarray(ic_true)))

    ic_arr = np.stack(ic_stack) if ic_stack else None
    fr_arr = np.stack(fr_stack) if fr_stack else None

    perturb_arr = None
    if is_sigma_sweep and sigma_ics:
        fp_stack = []
        for v in sweep_values:
            ip = sigma_ics.get(v)
            if ip is not None:
                try:
                    inp_p = make_inputs(name, ip, domain_extent=domain_extent, **phys)
                    fp_stack.append(np.asarray(apply_tesseract(t, inp_p)[output_key]))
                except Exception:
                    fp_stack.append(np.zeros_like(np.asarray(ic_true)))
            else:
                fp_stack.append(np.zeros_like(np.asarray(ic_true)))
        if fp_stack:
            perturb_arr = np.stack(fp_stack)
    return ic_arr, fr_arr, perturb_arr


def _compute_recovery_failure_values(by_sweep: dict, sweep_values: list) -> dict:
    """For each solver: first sweep value where the optimizer failed to converge."""
    failure_values: dict = {}
    for name, s_results in by_sweep.items():
        fail_val = None
        for val in sweep_values:
            r = s_results.get(val)
            if r is None or not r["converged"]:
                fail_val = val
                break
        failure_values[name] = fail_val
    return failure_values


def _build_recovery_shared_arrays(
    rep_val,
    sweep_values,
    ic_true,
    rep_ic_init,
    solver_names: list[str],
    final_states_gt: dict,
    is_sigma_sweep: bool,
    sigma_ics: dict,
) -> dict:
    """Assemble the ``shared_arrays`` dict for ``save_field_snapshots_npz``."""
    shared_final_gt = (
        np.asarray(final_states_gt[solver_names[0]])
        if solver_names and solver_names[0] in final_states_gt
        else None
    )
    shared: dict = {
        "rep_val": np.array([rep_val]),
        "sweep_values": np.array(sweep_values, dtype=float),
        "ic_true": np.asarray(ic_true),
        "ic_init": np.asarray(rep_ic_init)
        if rep_ic_init is not None
        else np.asarray(ic_true),
    }
    if shared_final_gt is not None:
        shared["final_gt_shared"] = shared_final_gt
    if is_sigma_sweep and sigma_ics:
        shared["ic_perturbed_all"] = np.stack(
            [np.asarray(sigma_ics[sv]) for sv in sweep_values]
        )
    return shared


def _save_recovery_outputs(
    out_dir,
    solver_names: list[str],
    per_solver_arrays: dict,
    shared: dict,
    result: dict,
    cfg: Problem,
    harness_fn,
    wall_times: dict[str, float],
    by_sweep: dict,
    exp_key: str,
) -> None:
    """Save the per-solver npz fields and result.json (skip if by_sweep empty)."""
    save_field_snapshots_npz(
        out_dir,
        solver_names,
        per_solver_arrays,
        shared_arrays=shared,
        filename="recovery_fields.npz",
        prefixes=(
            "ic_rec",
            "ic_history",
            "final_gt",
            "final_rec",
            "final_rep_val",
            "ic_rec_all",
            "final_rec_all",
            "final_perturbed_all",
        ),
    )
    if not by_sweep:
        from mosaic.benchmarks.core.console import print_warn

        print_warn(
            f"{exp_key}: by_sweep is empty (all solvers excluded or skipped) — "
            "skipping result.json save to preserve existing data"
        )
    else:
        save_experiment(
            result, out_dir, cfg=cfg, harness_fn=harness_fn, wall_time_s=wall_times
        )


def _build_recovery_per_solver_arrays(
    solver_names: list[str],
    ic_snaps: dict,
    ic_histories: dict,
    final_states_gt: dict,
    final_states_rec: dict,
    final_states_rep_val: dict,
    all_ic_snaps: dict,
    all_final_rec_snaps: dict,
    all_final_perturbed_snaps: dict,
) -> dict[str, dict[str, np.ndarray]]:
    """Build the per-solver dict of named ndarray entries for the npz save."""
    per_solver_arrays: dict[str, dict[str, np.ndarray]] = {}
    for sname in solver_names:
        entry: dict[str, np.ndarray] = {
            "ic_rec:": np.asarray(ic_snaps[sname]),
        }
        if sname in ic_histories:
            entry["ic_history:"] = np.asarray(ic_histories[sname])
        if sname in final_states_gt:
            entry["final_gt:"] = np.asarray(final_states_gt[sname])
        if sname in final_states_rec:
            entry["final_rec:"] = np.asarray(final_states_rec[sname])
        if sname in final_states_rep_val:
            entry["final_rep_val:"] = np.array([final_states_rep_val[sname]])
        if sname in all_ic_snaps:
            entry["ic_rec_all:"] = np.asarray(all_ic_snaps[sname])
        if sname in all_final_rec_snaps:
            entry["final_rec_all:"] = np.asarray(all_final_rec_snaps[sname])
        if sname in all_final_perturbed_snaps:
            entry["final_perturbed_all:"] = np.asarray(all_final_perturbed_snaps[sname])
        per_solver_arrays[sname] = entry
    return per_solver_arrays


@dataclass
class _RecoveryRunCtx:
    """Per-run state for `_run_recovery_long_impl` worker helpers.

    Encapsulates the otherwise huge set of captured locals so the helper
    functions can be lifted out of closures without a 30-parameter signature.
    The mutable accumulator dicts (``by_sweep``, ``ic_snaps`` etc.) are
    populated in place by worker callbacks running on the GPU pool.
    """

    cfg: Problem
    run: dict
    exp_key: str
    # Config snapshot
    sweep_key: str
    sweep_values: list
    phys: dict
    snap_interval: int
    lr: float
    max_iters: int
    patience: int
    perturb_sigma: float
    failure_threshold: float
    record_diagnostics: bool
    is_sigma_sweep: bool
    is_multi_seed: bool
    is_vel: bool
    rep_val: float
    primary_seed: int
    ic_seeds: list
    # Per-seed dicts
    ic_true_dict: dict
    ic_init_dict: dict
    sigma_ics: dict
    # Derived
    max_div: float | None
    ic_true: jax.Array
    optim_fn: object
    div_fn: object | None
    grad_proj_fn: object | None
    out_dir: object
    partial_lock: threading.Lock
    # Problem-semantics state (explicit dependencies)
    make_ic: object = None
    make_inputs: object = None
    error_fn: object = None
    output_key: str = ""
    domain_extent: float = 0.0
    # Accumulators (mutated in place by workers)
    by_sweep: dict = field(default_factory=dict)
    ic_snaps: dict = field(default_factory=dict)
    ic_init_snaps: dict = field(default_factory=dict)
    ic_histories: dict = field(default_factory=dict)
    final_states_gt: dict = field(default_factory=dict)
    final_states_rec: dict = field(default_factory=dict)
    final_states_rep_val: dict = field(default_factory=dict)
    all_ic_snaps: dict = field(default_factory=dict)
    all_final_rec_snaps: dict = field(default_factory=dict)
    all_final_perturbed_snaps: dict = field(default_factory=dict)
    wall_times: dict = field(default_factory=dict)


def _write_recovery_partial(ctx: _RecoveryRunCtx) -> None:
    """Snapshot ``by_sweep`` into ``result_partial.json`` under the run's lock."""
    partial = {
        "sweep_key": ctx.sweep_key,
        "by_sweep": ctx.by_sweep,
        "params": ctx.run,
    }
    with ctx.partial_lock:
        save_json(partial, ctx.out_dir / "result_partial.json")


def _run_one_seed_trial(
    ctx: _RecoveryRunCtx,
    name: str,
    t,
    color: str,
    val,
    s: int,
    ic_init_k,
    ic_true_k,
    target_k,
    is_primary: bool,
) -> dict | None:
    """Run a single per-seed optimisation trial; returns a result dict or None."""
    from mosaic.benchmarks.core.console import console

    phys = ctx.phys
    sweep_key = ctx.sweep_key
    snap_interval = ctx.snap_interval
    max_iters = ctx.max_iters

    def loss_fn(ic, _t=t, _target=target_k, _val=val):
        _phys_kw = phys if ctx.is_sigma_sweep else {**phys, sweep_key: _val}
        inp = ctx.make_inputs(
            name,
            ic,
            domain_extent=ctx.domain_extent,
            **_phys_kw,
        )
        return jnp.mean((apply_tesseract(_t, inp)[ctx.output_key] - _target) ** 2)

    _hist: list | None = (
        [] if (snap_interval > 0 and val == ctx.rep_val and is_primary) else None
    )
    _ic_err_hist: list[float] = []
    ic_error_init = float(ctx.error_fn(ic_init_k, ic_true_k))
    seed_tag = f" [ic_seed={s}]" if ctx.is_multi_seed else ""
    console.print(
        f"  [{color}]{name}[/] {sweep_key}={val}{seed_tag} "
        f"optim start (init_err={ic_error_init:.4g})"
    )

    def _log_iter(i, loss, _n=name, _c=color, _v=val, _st=seed_tag):
        console.print(
            f"  [{_c}]{_n}[/] {sweep_key}={_v}{_st} iter {i}/{max_iters} loss={loss:.4g}"
        )

    try:
        ic_opt, errors, diag = ctx.optim_fn(
            loss_fn,
            ic_init_k,
            ctx.lr,
            max_iters,
            ctx.patience,
            snap_interval=snap_interval if snap_interval > 0 else 0,
            history=_hist,
            snap_error_fn=lambda ic, _ict=ic_true_k: float(ctx.error_fn(ic, _ict)),
            error_history=_ic_err_hist if (snap_interval > 0 and is_primary) else None,
            log_fn=_log_iter,
            record_diagnostics=ctx.record_diagnostics,
            div_fn=ctx.div_fn,
            grad_proj_fn=ctx.grad_proj_fn,
        )
    except Exception as exc:
        from mosaic.benchmarks.core.console import print_warn

        print_warn(
            f"{name} {ctx.exp_key} optim failed at {sweep_key}={val}{seed_tag}: {exc}"
        )
        return None
    final_ic_error = ctx.error_fn(ic_opt, ic_true_k)
    final_ic_div = (
        _max_divergence(np.asarray(ic_opt), ctx.domain_extent) if ctx.is_vel else None
    )
    console.print(
        f"  [{color}]{name}[/] {sweep_key}={val}{seed_tag} done "
        f"iters={len(errors)} final_loss={errors[-1]:.4g} ic_err={final_ic_error:.4g}"
    )
    return {
        "errors": errors,
        "diag": diag,
        "ic_error_history": _ic_err_hist if is_primary else [],
        "ic_error_init": ic_error_init,
        "final_ic_error": float(final_ic_error),
        "final_ic_div": final_ic_div,
        "ic_opt": ic_opt,
        "target": target_k,
        "ic_init": ic_init_k,
        "phys_kw": phys if ctx.is_sigma_sweep else {**phys, sweep_key: val},
        "ic_seed": s,
        "_hist": _hist,
        "_ic_err_hist": _ic_err_hist,
    }


def _process_recovery_sweep_val(
    ctx: _RecoveryRunCtx,
    name: str,
    t,
    color: str,
    val,
    sigma_target,
    all_ic_opts: dict,
    best_conv: dict,
) -> dict:
    """Process one sweep value: build targets, run per-seed trials, aggregate.

    ``sigma_target`` is the pre-computed common target for sigma sweep (else
    ``None``). Updates ``ctx.by_sweep`` / vis snaps in place and returns the
    updated ``best_conv`` dict.
    """
    if ctx.is_sigma_sweep:
        _ic_init = ctx.sigma_ics[val]
        seeds_for_val: list[int] = [ctx.primary_seed]
        target_for_seed: dict[int, jax.Array] = {ctx.primary_seed: sigma_target}
    else:
        _ic_init = None
        seeds_for_val = ctx.ic_seeds
        target_for_seed = _compute_targets_for_val(
            name,
            ctx.exp_key,
            val,
            ctx.sweep_key,
            ctx.ic_seeds,
            ctx.ic_true_dict,
            ctx.phys,
            t,
            make_inputs=ctx.make_inputs,
            output_key=ctx.output_key,
            domain_extent=ctx.domain_extent,
        )
        if not target_for_seed:
            ctx.by_sweep[name][val] = None
            _write_recovery_partial(ctx)
            return best_conv

    trial_results: list[dict] = []
    primary_hist: list | None = None
    primary_ic_err_hist: list[float] = []

    for s in seeds_for_val:
        if s not in target_for_seed:
            continue
        ic_true_k = ctx.ic_true_dict[s]
        ic_init_k = ctx.ic_init_dict[s] if not ctx.is_sigma_sweep else _ic_init
        target_k = target_for_seed[s]
        is_primary = s == ctx.primary_seed
        trial = _run_one_seed_trial(
            ctx, name, t, color, val, s, ic_init_k, ic_true_k, target_k, is_primary
        )
        if trial is None:
            continue
        if is_primary and trial["_hist"]:
            primary_hist = trial["_hist"]
        if is_primary and trial["_ic_err_hist"]:
            primary_ic_err_hist = trial["_ic_err_hist"]
        trial_results.append(trial)

    if not trial_results:
        ctx.by_sweep[name][val] = None
        _write_recovery_partial(ctx)
        return best_conv

    ctx.by_sweep[name][val] = _aggregate_trial_results(
        trial_results,
        primary_ic_err_hist,
        val,
        ctx.perturb_sigma,
        ctx.is_sigma_sweep,
        ctx.is_multi_seed,
        ctx.failure_threshold,
        ctx.max_div,
        ctx.ic_true,
    )
    _write_recovery_partial(ctx)

    prim_trial = next(
        (r for r in trial_results if r["ic_seed"] == ctx.primary_seed),
        trial_results[0],
    )
    ic_opt = prim_trial["ic_opt"]
    all_ic_opts[val] = ic_opt
    if val == ctx.rep_val:
        ctx.ic_snaps[name] = ic_opt
        ctx.ic_init_snaps[name] = prim_trial["ic_init"]
        if primary_hist:
            ctx.ic_histories[name] = np.asarray(primary_hist)
    if prim_trial["final_ic_error"] < ctx.failure_threshold:
        return {
            "val": val,
            "ic_opt": ic_opt,
            "target": prim_trial["target"],
            "ic_init": prim_trial["ic_init"],
            "phys_kw": prim_trial["phys_kw"],
        }
    return best_conv


def _collect_final_states(ctx: _RecoveryRunCtx, name: str, t, best_conv: dict) -> None:
    """Use the hardest converged val (or rep_val) to record final GT / rec states."""
    fv = best_conv.get("val", ctx.rep_val)
    fic = best_conv.get("ic_opt", ctx.ic_snaps.get(name))
    ftgt = best_conv.get("target")
    fphys = best_conv.get("phys_kw", ctx.phys)
    if fic is None or ftgt is None:
        return
    ctx.final_states_gt[name] = np.asarray(ftgt)
    ctx.final_states_rep_val[name] = fv
    try:
        inp_rec = ctx.make_inputs(
            name,
            fic,
            domain_extent=ctx.domain_extent,
            **fphys,
        )
        ctx.final_states_rec[name] = np.asarray(
            apply_tesseract(t, inp_rec)[ctx.output_key]
        )
    except Exception:
        pass


def _recovery_long_work(ctx: _RecoveryRunCtx, name: str, t) -> None:
    """Per-solver worker: run the full sweep optimisation pipeline.

    Wall-time bookkeeping is handled by :func:`per_solver_loop` in the caller.
    """
    from mosaic.benchmarks.core.console import console

    color = ctx.cfg.solver(name).color
    ctx.by_sweep[name] = {}
    best_conv: dict = {}
    all_ic_opts: dict = {}
    console.print(
        f"  [{color}]{name}[/] {ctx.exp_key} starting ({len(ctx.sweep_values)} sweep values, "
        f"max_iters={ctx.max_iters})"
    )

    sigma_target = None
    if ctx.is_sigma_sweep:
        sigma_target = _precompute_sigma_target(ctx, name, t, color)
        if sigma_target is None:
            for val in ctx.sweep_values:
                ctx.by_sweep[name][val] = None
            return

    for val in ctx.sweep_values:
        best_conv = _process_recovery_sweep_val(
            ctx, name, t, color, val, sigma_target, all_ic_opts, best_conv
        )

    _collect_final_states(ctx, name, t, best_conv)

    ic_arr, fr_arr, perturb_arr = _build_recovery_visualization_stacks(
        name,
        all_ic_opts,
        ctx.sweep_values,
        ctx.sweep_key,
        ctx.phys,
        ctx.is_sigma_sweep,
        ctx.sigma_ics if ctx.is_sigma_sweep else None,
        ctx.ic_true,
        t,
        make_inputs=ctx.make_inputs,
        output_key=ctx.output_key,
        domain_extent=ctx.domain_extent,
    )
    if ic_arr is not None:
        ctx.all_ic_snaps[name] = ic_arr
    if fr_arr is not None:
        ctx.all_final_rec_snaps[name] = fr_arr
    if perturb_arr is not None:
        ctx.all_final_perturbed_snaps[name] = perturb_arr


def _precompute_sigma_target(ctx: _RecoveryRunCtx, name: str, t, color: str):
    """For sigma sweep: forward the primary true IC to produce the shared target.

    Returns the target array, or ``None`` if forward failed or output invalid.
    """
    try:
        inputs_true_fixed = ctx.make_inputs(
            name,
            ctx.ic_true,
            domain_extent=ctx.domain_extent,
            **ctx.phys,
        )
        target = apply_tesseract(t, inputs_true_fixed)[ctx.output_key]
    except Exception as exc:
        from mosaic.benchmarks.core.console import print_warn

        print_warn(
            f"{name} {ctx.exp_key} target forward failed: {exc} — "
            f"marking all {len(ctx.sweep_values)} sweep values as None"
        )
        return None
    if not is_valid(target):
        return None
    return target


def _run_recovery_long_impl(
    cfg: Problem,
    tags: dict[str, str],
    exp_key: str,
    harness_fn,
    *,
    make_ic,
    make_inputs,
    error_fn,
    output_key: str,
    domain_extent: float,
    runs=None,
    _optim_fn=_run_optim,
    _project_grads: bool = False,
    **overrides,
) -> dict:
    """Shared implementation for run_recovery_long and variants."""
    if not runs:
        raise NotImplementedError(
            f"_run_recovery_long_impl requires runs= payload for {exp_key!r} "
            f"(not configured for '{cfg.name}')"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        result = _run_recovery_for_one_run(
            cfg=cfg,
            tags=tags,
            exp_key=exp_key,
            harness_fn=harness_fn,
            run=run,
            n_runs=n_runs,
            overrides=overrides,
            optim_fn=_optim_fn,
            project_grads=_project_grads,
            make_ic=make_ic,
            make_inputs=make_inputs,
            error_fn=error_fn,
            output_key=output_key,
            domain_extent=domain_extent,
        )
        if n_runs > 1:
            ic_name = run.get("ic", {}).get("name", next(iter(make_ic)))
            all_results[ic_name] = result
        else:
            all_results = result

    return all_results


def _run_recovery_for_one_run(  # noqa: PLR0913 — explicit-deps signature
    *,
    cfg: Problem,
    tags: dict[str, str],
    exp_key: str,
    harness_fn,
    run: dict,
    n_runs: int,
    overrides: dict,
    optim_fn,
    project_grads: bool,
    make_ic,
    make_inputs,
    error_fn,
    output_key: str,
    domain_extent: float,
) -> dict:
    """Inner body of ``_run_recovery_long_impl``: process a single ``run`` entry."""
    ic_cfg = run.get("ic", {})
    ic_name = ic_cfg.get("name", next(iter(make_ic)))
    seed = ic_cfg.get("seed", 0)
    sweep_cfg = run.get("sweep", {})
    sweep_key = sweep_cfg.get("key", "steps")
    sweep_values = sweep_cfg.get("values", [])
    optim_cfg = run.get("optim", {})
    perturb_sigma = optim_cfg.get("perturb_sigma", 0.3)
    lr = optim_cfg.get("lr", 3e-3)
    max_iters = optim_cfg.get("max_iters", 1000)
    patience = optim_cfg.get("patience", 100)
    failure_threshold = optim_cfg.get("failure_threshold", 0.5)
    snap_interval = int(optim_cfg.get("snap_interval", 0))
    # Multi-seed: list of IC seeds; defaults to single seed from ic_cfg.
    ic_seeds: list[int] = optim_cfg.get("ic_seeds", [seed])
    record_diagnostics: bool = bool(optim_cfg.get("record_diagnostics", True))
    ic_init_type: str = optim_cfg.get("ic_init_type", "perturb")
    phys = run.get("physics", {})
    _is_sigma_sweep = sweep_key == "perturb_sigma"
    _is_multi_seed = len(ic_seeds) > 1
    ic_subdir = ic_name if n_runs > 1 else ""
    _optim_fn = optim_fn
    _project_grads = project_grads

    if not sweep_values:
        raise NotImplementedError(
            f"'{exp_key}' requires sweep.values in runs payload "
            f"(not configured for '{cfg.name}')"
        )

    from mosaic.benchmarks.core.console import console

    # ── Build per-seed IC true + perturbed IC ─────────────────────────────
    _ic_true_dict, _ic_init_dict, _max_div_dict = _build_per_seed_ics(
        ic_name,
        ic_seeds,
        phys,
        perturb_sigma,
        ic_init_type,
        console,
        make_ic=make_ic,
        domain_extent=domain_extent,
    )

    # Primary IC / init for single-seed and for visualization (always seed 0 / ic_seeds[0])
    _primary_seed = ic_seeds[0]
    ic_true = _ic_true_dict[_primary_seed]
    max_div = _max_div_dict[_primary_seed]
    _is_vel = _is_velocity_field(np.asarray(ic_true))

    # div_fn for diagnostics: computes max|∇·u| per-iteration inside _run_optim
    _div_fn = (
        (lambda u: _max_divergence(np.asarray(u), domain_extent))
        if (_is_vel and record_diagnostics)
        else None
    )

    # Gradient projection: Helmholtz-project onto ∇·g = 0 before handing to
    # the optimiser. Only meaningful for velocity fields; ignored otherwise.
    _grad_proj_fn = (
        (lambda g: _project_divergence_free(g, domain_extent))
        if (_project_grads and _is_vel)
        else None
    )

    _sigma_ics: dict = {}
    if not _is_sigma_sweep:
        ic_init = _ic_init_dict[_primary_seed]
    else:
        ic_init = None
        _sigma_ics = _build_sigma_perturbed_ics(
            ic_true,
            sweep_values,
            seed,
            _is_vel,
            console,
            domain_extent=domain_extent,
        )

    rep_val = sweep_values[len(sweep_values) // 2]
    gpu_ids = overrides.get("gpu_ids")

    out_dir = experiment_dir(
        results_dir(),
        cfg.name,
        _SUITE,
        f"{exp_key}/{ic_subdir}" if ic_subdir else exp_key,
        suffix="_debug" if overrides.get("debug") else "",
    )

    ctx = _RecoveryRunCtx(
        cfg=cfg,
        run=run,
        exp_key=exp_key,
        sweep_key=sweep_key,
        sweep_values=sweep_values,
        phys=phys,
        snap_interval=snap_interval,
        lr=lr,
        max_iters=max_iters,
        patience=patience,
        perturb_sigma=perturb_sigma,
        failure_threshold=failure_threshold,
        record_diagnostics=record_diagnostics,
        is_sigma_sweep=_is_sigma_sweep,
        is_multi_seed=_is_multi_seed,
        is_vel=_is_vel,
        rep_val=rep_val,
        primary_seed=_primary_seed,
        ic_seeds=ic_seeds,
        ic_true_dict=_ic_true_dict,
        ic_init_dict=_ic_init_dict,
        sigma_ics=_sigma_ics,
        max_div=max_div,
        ic_true=ic_true,
        optim_fn=_optim_fn,
        div_fn=_div_fn,
        grad_proj_fn=_grad_proj_fn,
        out_dir=out_dir,
        partial_lock=threading.Lock(),
        make_ic=make_ic,
        make_inputs=make_inputs,
        error_fn=error_fn,
        output_key=output_key,
        domain_extent=domain_extent,
    )

    ctx.wall_times = per_solver_loop(
        cfg,
        tags,
        active_differentiable_solvers(cfg, "optimization", exp_key),
        lambda name, t: _recovery_long_work(ctx, name, t),
        gpu_ids=gpu_ids,
        print_done=False,
    )

    by_sweep = ctx.by_sweep
    ic_snaps = ctx.ic_snaps
    ic_init_snaps = ctx.ic_init_snaps
    ic_histories = ctx.ic_histories
    final_states_gt = ctx.final_states_gt
    final_states_rec = ctx.final_states_rec
    final_states_rep_val = ctx.final_states_rep_val
    all_ic_snaps = ctx.all_ic_snaps
    all_final_rec_snaps = ctx.all_final_rec_snaps
    all_final_perturbed_snaps = ctx.all_final_perturbed_snaps
    _wall_times = ctx.wall_times

    failure_values = _compute_recovery_failure_values(by_sweep, sweep_values)

    solver_names = list(ic_snaps.keys())
    _rep_ic_init = (
        ic_init_snaps.get(solver_names[0], ic_init)
        if ic_init_snaps and solver_names
        else ic_init
    )
    per_solver_arrays = _build_recovery_per_solver_arrays(
        solver_names,
        ic_snaps,
        ic_histories,
        final_states_gt,
        final_states_rec,
        final_states_rep_val,
        all_ic_snaps,
        all_final_rec_snaps,
        all_final_perturbed_snaps,
    )
    shared = _build_recovery_shared_arrays(
        rep_val,
        sweep_values,
        ic_true,
        _rep_ic_init,
        solver_names,
        final_states_gt,
        _is_sigma_sweep,
        _sigma_ics,
    )

    result = {
        "sweep_key": sweep_key,
        "by_sweep": by_sweep,
        "failure_values": failure_values,
        "params": run,
    }
    _save_recovery_outputs(
        out_dir,
        solver_names,
        per_solver_arrays,
        shared,
        result,
        cfg,
        harness_fn,
        _wall_times,
        by_sweep,
        exp_key,
    )
    return result


def run_recovery(
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
    exp_key: str = "recovery",
    **overrides,
) -> dict:
    """IC recovery from a zero initial guess (cold start).

    ``optimizer`` selects the inner optimiser:

      * ``"adam"``       — vanilla Adam (default).
      * ``"bfgs"``       — L-BFGS with zoom line-search.
      * ``"bfgs_proj"``  — L-BFGS with the gradient Helmholtz-projected
        onto the ∇·g = 0 subspace each iteration (keeps the search
        direction compatible with incompressibility for velocity-field
        problems).

    Problem-semantics state (``make_ic``, ``make_inputs``, ``error_fn``,
    ``output_key``, ``domain_extent``) is passed explicitly. ``cfg``
    retains its runtime-registry role only.
    """
    optim_fn = _run_lbfgs if optimizer.startswith("bfgs") else _run_optim
    project = optimizer == "bfgs_proj"
    return _run_recovery_long_impl(
        cfg,
        tags,
        exp_key,
        run_recovery,
        runs=runs,
        _optim_fn=optim_fn,
        _project_grads=project,
        make_ic=make_ic,
        make_inputs=make_inputs,
        error_fn=error_fn,
        output_key=output_key,
        domain_extent=domain_extent,
        **overrides,
    )
