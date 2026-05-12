"""Optimization suite: IC recovery via gradient descent.

Only runs solvers where SolverSpec.differentiable is True.

Run from the terminal:
    mosaic run <problem> optimization [--experiments EXPR] [--plots-only]
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mosaic.benchmarks.core.config import Problem, has_vjp
from mosaic.benchmarks.core.io import (
    experiment_dir,
    results_dir,
    save_experiment,
    save_field_snapshots_npz,
    save_harness_result,
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
    is_valid,
    iter_runs,
)

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


def _run_optim(  # noqa: PLR0913 — explicit-deps signature
    loss_fn,
    init_x,
    lr: float,
    max_iters: int,
    patience: int,
    *,
    snap_interval: int = 0,
    history: list | None = None,
    snap_error_fn=None,
    error_history: list | None = None,
    log_fn=None,
    log_interval: int = 20,
    record_diagnostics: bool = False,
    div_fn=None,
    grad_proj_fn=None,
):
    """Adam with patience-based early stopping.

    Returns ``(final_x, losses, diag)`` where ``losses`` is the per-iteration
    loss list and ``diag`` is a dict with optional diagnostic time-series
    (all ``None`` when ``record_diagnostics=False``):

    - ``grad_norms``: per-iter ``‖∇L‖₂``
    - ``grad_divs``:  per-iter ``max|∇·g|`` (only when ``div_fn`` provided)
    - ``ic_divs``:    per-iter ``max|∇·u|`` (only when ``div_fn`` provided)
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(init_x)
    x = init_x
    losses, best, no_improve = [], jnp.inf, 0
    # grad_norms always recorded; grad_divs/ic_divs only when record_diagnostics
    diag: dict = {"grad_norms": []}
    if record_diagnostics:
        diag["grad_divs"] = [] if div_fn else None
        diag["ic_divs"] = [] if div_fn else None
    for i in range(max_iters):
        loss, g = jax.value_and_grad(loss_fn)(x)
        diag["grad_norms"].append(float(jnp.linalg.norm(g.ravel())))
        if record_diagnostics and div_fn is not None:
            diag["grad_divs"].append(div_fn(np.asarray(g)))
            diag["ic_divs"].append(div_fn(np.asarray(x)))
        updates, opt_state = optimizer.update(g, opt_state)
        x = optax.apply_updates(x, updates)
        loss_val = float(loss)
        losses.append(loss_val)
        if loss_val < best:
            best, no_improve = loss_val, 0
        else:
            no_improve += 1
        if snap_interval > 0 and (i + 1) % snap_interval == 0:
            if history is not None:
                history.append(np.asarray(x))
            if snap_error_fn is not None and error_history is not None:
                error_history.append(snap_error_fn(np.asarray(x)))
        if log_fn is not None and (i + 1) % log_interval == 0:
            log_fn(i + 1, loss_val)
        if no_improve >= patience:
            break
    return x, losses, diag


def _run_lbfgs(  # noqa: PLR0913 — explicit-deps signature
    loss_fn,
    init_x,
    lr=None,
    max_iters: int = 100,
    patience=None,
    *,
    record_diagnostics: bool = False,
    div_fn=None,
    snap_interval: int = 0,
    history: list | None = None,
    snap_error_fn=None,
    error_history: list | None = None,
    log_fn=None,
    log_interval: int = 10,
    clip_fn=None,
    grad_proj_fn=None,
):
    """L-BFGS with zoom line-search.

    ``clip_fn`` is called after each update to project ``x`` back into the
    feasible set (e.g. ``jnp.clip(x, x_min, 1)`` for density fields).
    ``grad_proj_fn``, when provided, is called on the gradient (as a numpy
    array) before it is handed to L-BFGS, e.g. Helmholtz projection onto the
    divergence-free subspace for velocity-field optimisation.
    Returns ``(final_x, losses, None)`` — same shape as ``_run_optim``.

    The ``lr``, ``patience``, ``record_diagnostics``, and ``div_fn`` parameters
    are accepted but ignored, so call sites that pass them (e.g.
    ``_run_recovery_long_impl``) work without modification.

    Signature matches ``_run_optim(loss_fn, init_x, lr, max_iters, patience, ...)``
    so the two are interchangeable at call sites.
    """
    solver = optax.lbfgs()
    opt_state = solver.init(init_x)
    x = init_x
    losses: list[float] = []
    grad_norms: list[float] = []
    value_and_grad = optax.value_and_grad_from_state(loss_fn)
    for i in range(max_iters):
        value, grad = value_and_grad(x, state=opt_state)
        if grad_proj_fn is not None:
            grad = jnp.array(grad_proj_fn(np.asarray(grad)))
        grad_norms.append(float(jnp.linalg.norm(grad.ravel())))
        updates, opt_state = solver.update(
            grad, opt_state, x, value=value, grad=grad, value_fn=loss_fn
        )
        x = optax.apply_updates(x, updates)
        if clip_fn is not None:
            x = clip_fn(x)
        loss_val = float(value)
        losses.append(loss_val)
        if snap_interval > 0 and (i + 1) % snap_interval == 0:
            if history is not None:
                history.append(np.asarray(x))
            if snap_error_fn is not None and error_history is not None:
                error_history.append(snap_error_fn(np.asarray(x)))
        if log_fn is not None and (i + 1) % log_interval == 0:
            log_fn(i + 1, loss_val)
        if grad_norms[-1] < 1e-7:
            break
    return x, losses, {"grad_norms": grad_norms}


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


# ── Topology optimisation ─────────────────────────────────────────────────────


# ── topopt: shared body, Adam vs L-BFGS as an argument ───────────────────────


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


# ── Conductivity recovery (thermal-mesh) ─────────────────────────────────────


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
