# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""IC-recovery kernel for the ns-3d-grid problem.

The IC-recovery pipeline used to be a bespoke ``run_recovery`` harness that
managed its own per-solver and per-sweep loops. It now lives as a
:func:`recovery` kernel + :func:`_recovery_aggregate` aggregate plugged
into :func:`mosaic.benchmarks.core.experiment.run_experiment`.

Layout:

* :func:`recovery` — per-(solver, sweep-value) kernel. The framework
  iterates ``sweep_values`` (``sweep_mode="default"``); each invocation
  handles ONE sweep point. The kernel still iterates ``ic_seeds`` (the
  inner per-seed trial loop) and aggregates per-seed trial results into
  the single ``by_sweep[name][val]`` entry it returns as ``metrics``.
* :func:`_recovery_aggregate` — cross-solver post-pass. The framework
  hands back ``by_solver[name] = {val: entry}`` (already shaped as
  ``by_sweep``); the aggregate computes ``failure_values``, repacks the
  per-(solver, val) visualisation snapshots into the legacy
  ``recovery_fields.npz`` schema, and returns the final
  ``{sweep_key, by_sweep, failure_values, params}`` result dict.
* Inner helpers (``_build_per_seed_ics``, ``_compute_targets_for_val``,
  ``_run_one_seed_trial``, …) are unchanged science primitives shared
  between the kernel body and (previously) the runner.

The inner optimiser primitives ``_run_optim`` (Adam with patience-based
early stopping) and ``_run_lbfgs`` (L-BFGS with zoom line-search) remain
in ``shared/`` since they are reused by topology-optimisation and other
suites.
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
    save_field_snapshots_npz,
    try_load_json,
)

# JAX-traced loss_fn closures capture this reference at trace time;
# using the tracer-aware wrapper ensures primitive binding sees the
# active trace.
from mosaic.benchmarks.core.tracer_apply import apply_tesseract
from mosaic.benchmarks.core.utils import is_valid
from mosaic.benchmarks.problems.shared.optimization import _run_lbfgs, _run_optim

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


def _project_divergence_free_jax(u: jax.Array, domain_extent: float) -> jax.Array:
    """JAX-traceable variant of :func:`_project_divergence_free`.

    Used as ``grad_proj_fn`` inside the JIT-compiled L-BFGS step so the
    whole iteration is one trace.
    """
    nd = u.shape[-1]
    squeeze = u.ndim > nd + 1
    v = u.reshape((*u.shape[:nd], nd)) if squeeze else u

    N = v.shape[0]
    k1d = jnp.fft.fftfreq(N) * N * (2.0 * jnp.pi / domain_extent)
    grids = jnp.meshgrid(*([k1d] * nd), indexing="ij")
    k2 = sum(k**2 for k in grids)
    k2_safe = jnp.where(k2 == 0.0, 1.0, k2)

    spatial_axes = tuple(range(nd))
    v_hat = jnp.fft.fftn(v, axes=spatial_axes)

    k_dot_u = sum(grids[i] * v_hat[..., i] for i in range(nd))

    v_hat_df = jnp.stack(
        [v_hat[..., i] - grids[i] * k_dot_u / k2_safe for i in range(nd)],
        axis=-1,
    )

    v_df = jnp.fft.ifftn(v_hat_df, axes=spatial_axes).real
    v_df = v_df.reshape(u.shape) if squeeze else v_df
    return v_df.astype(u.dtype)


def _project_ic_with_log(
    raw: jax.Array,
    label: str,
    is_vel: bool,
    domain_extent: float,
    console: Any,
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
    console: Any,
    *,
    make_ic: Any,
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


def _build_sigma_perturbed_ic(
    ic_true: jax.Array,
    sigma_val: float,
    seed: int,
    is_vel: bool,
    console: Any,
    *,
    domain_extent: float,
) -> jax.Array:
    """Build the perturbed IC at a single sigma value (div-free projected)."""
    key_sv = jax.random.fold_in(jax.random.PRNGKey(seed), int(sigma_val * 1000))
    raw = ic_true + sigma_val * jax.random.normal(
        key_sv, ic_true.shape, dtype=jnp.float32
    )
    return _project_ic_with_log(raw, f"σ={sigma_val}", is_vel, domain_extent, console)


def _compute_targets_for_val(
    name: str,
    exp_key: str,
    val: Any,
    sweep_key: str,
    ic_seeds: list[int],
    ic_true_dict: dict,
    phys: dict,
    t: Any,
    *,
    make_inputs: Any,
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
    val: Any,
    perturb_sigma: float,
    is_sigma_sweep: bool,
    is_multi_seed: bool,
    failure_threshold: float,
    max_div: Any,
    ic_true: Any,
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


# ── Per-seed trial body ───────────────────────────────────────────────────────


def _run_one_seed_trial(
    *,
    name: str,
    t: Any,
    color: str,
    val: Any,
    s: int,
    ic_init_k: Any,
    ic_true_k: Any,
    target_k: Any,
    is_primary: bool,
    phys: dict,
    sweep_key: str,
    is_sigma_sweep: bool,
    is_multi_seed: bool,
    is_vel: bool,
    rep_val: Any,
    snap_interval: int,
    lr: float,
    max_iters: int,
    patience: int,
    record_diagnostics: bool,
    optim_fn: Any,
    div_fn: Any,
    grad_proj_fn: Any,
    make_inputs: Any,
    error_fn: Any,
    output_key: str,
    domain_extent: float,
    exp_key: str,
) -> dict | None:
    """Run a single per-seed optimisation trial; returns a result dict or None."""
    from mosaic.benchmarks.core.console import console

    def loss_fn(ic: Any, _t: Any = t, _target: Any = target_k, _val: Any = val) -> Any:
        _phys_kw = phys if is_sigma_sweep else {**phys, sweep_key: _val}
        inp = make_inputs(
            name,
            ic,
            domain_extent=domain_extent,
            **_phys_kw,
        )
        return jnp.mean((apply_tesseract(_t, inp)[output_key] - _target) ** 2)

    _hist: list | None = (
        [] if (snap_interval > 0 and val == rep_val and is_primary) else None
    )
    _ic_err_hist: list[float] = []
    ic_error_init = float(error_fn(ic_init_k, ic_true_k))
    seed_tag = f" [ic_seed={s}]" if is_multi_seed else ""
    console.print(
        f"  [{color}]{name}[/] {sweep_key}={val}{seed_tag} "
        f"optim start (init_err={ic_error_init:.4g})"
    )

    def _log_iter(
        i: Any,
        loss: Any,
        _n: Any = name,
        _c: Any = color,
        _v: Any = val,
        _st: Any = seed_tag,
    ) -> None:
        console.print(
            f"  [{_c}]{_n}[/] {sweep_key}={_v}{_st} iter {i}/{max_iters} loss={loss:.4g}"
        )

    try:
        ic_opt, errors, diag = optim_fn(
            loss_fn,
            ic_init_k,
            lr,
            max_iters,
            patience,
            snap_interval=snap_interval if snap_interval > 0 else 0,
            history=_hist,
            snap_error_fn=lambda ic, _ict=ic_true_k: float(error_fn(ic, _ict)),
            error_history=_ic_err_hist if (snap_interval > 0 and is_primary) else None,
            log_fn=_log_iter,
            record_diagnostics=record_diagnostics,
            div_fn=div_fn,
            grad_proj_fn=grad_proj_fn,
        )
    except Exception as exc:
        from mosaic.benchmarks.core.console import print_warn

        print_warn(
            f"{name} {exp_key} optim failed at {sweep_key}={val}{seed_tag}: {exc}"
        )
        return None
    final_ic_error = error_fn(ic_opt, ic_true_k)
    final_ic_div = (
        _max_divergence(np.asarray(ic_opt), domain_extent) if is_vel else None
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
        "phys_kw": phys if is_sigma_sweep else {**phys, sweep_key: val},
        "ic_seed": s,
        "_hist": _hist,
        "_ic_err_hist": _ic_err_hist,
    }


def _precompute_sigma_target(
    *,
    name: str,
    t: Any,
    ic_true: Any,
    phys: dict,
    make_inputs: Any,
    output_key: str,
    domain_extent: float,
    exp_key: str,
) -> Any:
    """For sigma sweep: forward the primary true IC to produce the shared target.

    Returns the target array, or ``None`` if forward failed or output invalid.
    """
    try:
        inputs_true_fixed = make_inputs(
            name,
            ic_true,
            domain_extent=domain_extent,
            **phys,
        )
        target = apply_tesseract(t, inputs_true_fixed)[output_key]
    except Exception as exc:
        from mosaic.benchmarks.core.console import print_warn

        print_warn(f"{name} {exp_key} target forward failed: {exc}")
        return None
    if not is_valid(target):
        return None
    return target


# ── Aggregate ────────────────────────────────────────────────────────────────


def _decode_snap_key(key: str) -> tuple[str, str]:
    """Split a ``"prefix:suffix"`` framework snapshot key.

    The framework's ``_absorb`` writes per-call snapshots under
    ``f"{prefix}:{idx}"`` where ``idx`` is the sweep-iteration index
    (stringified). Empty suffix means a single (non-per-val) array.
    """
    if ":" in key:
        p, s = key.split(":", 1)
        return p, s
    return key, ""


def _collect_per_val_arrays(
    suf_map: dict[str, np.ndarray], prefix: str, n_vals: int, fallback: np.ndarray
) -> list[np.ndarray]:
    """Order per-val snapshots ``prefix:0..N-1`` into a list, zero-padding gaps."""
    out: list[np.ndarray] = []
    for idx in range(n_vals):
        arr = suf_map.get(f"{prefix}:{idx}")
        out.append(np.asarray(arr) if arr is not None else np.zeros_like(fallback))
    return out


def _build_solver_snap_layout(
    suf_map: dict[str, np.ndarray],
    sweep_values: list,
    by_sweep_name: dict,
    failure_threshold: float,
    rep_val: Any,
    fallback_ic: np.ndarray,
    is_sigma_sweep: bool,
) -> dict[str, np.ndarray]:
    """Repack per-(solver, val) framework snapshots into the legacy NPZ schema.

    The kernel emits one ``ic_rec`` / ``final_gt`` / ``final_rec`` array per
    sweep value (framework adds ``:<idx>`` suffix). The on-disk schema wants:

      * ``ic_rec_<j>``        — IC at rep_val (showcase)
      * ``ic_history_<j>``    — history at rep_val
      * ``final_gt_<j>``      — target at the hardest converged val
      * ``final_rec_<j>``     — forward of that IC at that val
      * ``final_rep_val_<j>`` — that val itself
      * ``ic_rec_all_<j>``    — stack across all sweep values
      * ``final_rec_all_<j>`` — stack across all sweep values
      * ``final_perturbed_all_<j>`` — sigma-sweep only, stack of perturbed forwards

    Returns the empty-suffix ``"prefix:"`` keyed dict expected by
    :func:`save_field_snapshots_npz` (positional layout).
    """
    n_vals = len(sweep_values)
    rep_idx = n_vals // 2

    # Stack ``_all`` arrays across all sweep values.
    ic_all = _collect_per_val_arrays(suf_map, "ic_rec", n_vals, fallback_ic)
    fr_all = _collect_per_val_arrays(suf_map, "final_rec", n_vals, fallback_ic)
    perturb_all = (
        _collect_per_val_arrays(suf_map, "final_perturbed", n_vals, fallback_ic)
        if is_sigma_sweep
        else None
    )

    # ``best_conv``: pick the hardest sweep value where the trial converged;
    # legacy semantics walked the sweep in registration order and overwrote
    # so the last converged val wins (NOT a strict ``max`` — keeps the same
    # ordering as the iteration order).
    best_idx: int | None = None
    for idx, val in enumerate(sweep_values):
        entry = by_sweep_name.get(val)
        if entry is None:
            continue
        if (
            entry.get("final_ic_error") is not None
            and entry["final_ic_error"] < failure_threshold
        ):
            best_idx = idx
    showcase_idx = best_idx if best_idx is not None else rep_idx
    showcase_val = sweep_values[showcase_idx]

    out: dict[str, np.ndarray] = {}
    # Single-array showcases (empty suffix → ``ic_rec_<j>``).
    rep_ic = suf_map.get(f"ic_rec:{rep_idx}")
    if rep_ic is not None:
        out["ic_rec:"] = np.asarray(rep_ic)
    rep_hist = suf_map.get(f"ic_history:{rep_idx}")
    if rep_hist is not None:
        out["ic_history:"] = np.asarray(rep_hist)
    # ``final_gt``/``final_rec``/``final_rep_val`` come from the best-conv val
    # (showcase_val). Per-val ``final_gt:<idx>`` / ``final_rec:<idx>`` arrays
    # are emitted by the kernel; pick the showcase index.
    sc_gt = suf_map.get(f"final_gt:{showcase_idx}")
    sc_rec = suf_map.get(f"final_rec:{showcase_idx}")
    if sc_gt is not None:
        out["final_gt:"] = np.asarray(sc_gt)
    if sc_rec is not None:
        out["final_rec:"] = np.asarray(sc_rec)
    if sc_gt is not None or sc_rec is not None:
        out["final_rep_val:"] = np.array([showcase_val])

    out["ic_rec_all:"] = np.stack(ic_all)
    out["final_rec_all:"] = np.stack(fr_all)
    if perturb_all is not None:
        out["final_perturbed_all:"] = np.stack(perturb_all)
    return out


def _recovery_aggregate(
    by_solver: Any,
    *,
    run: Any,
    cfg: Any,
    out_dir: Any,
    snapshots: Any,
    shared_extras: Any,
    ic: Any,
    sweep_values: Any,
    sweep_key: Any,
    snapshot_filename: Any,
    snapshot_prefixes: Any,
    **_kw: Any,
) -> dict:
    """Cross-solver post-pass: failure_values + per-solver NPZ + result dict.

    The framework already iterated ``sweep_values`` via ``sweep_mode="default"``
    so ``by_solver[name]`` is the per-solver ``{val: entry}`` map (i.e.
    ``by_sweep[name]``). Per-val visualisation snapshots arrive on
    ``snapshots[name][f"{prefix}:{idx}"]``; we repack them into the
    legacy ``recovery_fields.npz`` empty-suffix schema.
    """
    del ic  # already covered by shared_extras["ic_true"] / ["ic_init"]
    by_sweep: dict = {name: per for name, per in by_solver.items() if per}

    optim_cfg = run.get("optim", {})
    failure_threshold = optim_cfg.get("failure_threshold", 0.5)
    rep_val = sweep_values[len(sweep_values) // 2] if sweep_values else None

    failure_values = _compute_recovery_failure_values(by_sweep, sweep_values)

    # Build per-solver legacy NPZ layout. Use ic_true (from shared_extras) as
    # the zero-padding template for any missing per-val arrays so the stacked
    # ``ic_rec_all`` shape is well-defined even with mid-sweep failures.
    ic_true_arr = np.asarray(shared_extras.get("ic_true"))
    is_sigma_sweep = sweep_key == "perturb_sigma"
    per_solver_arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, suf_map in snapshots.items():
        per_solver_arrays[name] = _build_solver_snap_layout(
            suf_map,
            sweep_values,
            by_sweep.get(name, {}),
            failure_threshold,
            rep_val,
            ic_true_arr,
            is_sigma_sweep,
        )
    solver_names = list(snapshots.keys())

    # Shared NPZ payload: framework-collected ``shared`` values from the first
    # solver to report (rep_val, sweep_values, ic_true, ic_init, …). For sigma
    # sweep we also build ``ic_perturbed_all`` by stacking the per-val
    # ``ic_perturbed`` arrays from whichever solver's snapshot dict contains
    # them (each solver emits the same set; first non-empty wins).
    shared_dict: dict = {k: np.asarray(v) for k, v in shared_extras.items()}
    if is_sigma_sweep and snapshots:
        first_suf = next(iter(snapshots.values()))
        perturbed_stack = _collect_per_val_arrays(
            first_suf, "ic_perturbed", len(sweep_values), ic_true_arr
        )
        if perturbed_stack:
            shared_dict["ic_perturbed_all"] = np.stack(perturbed_stack)

    # Empty-by_sweep preservation: the legacy code skipped result.json save
    # entirely when no solver returned any sweep value. Reproduce by handing
    # back the existing result dict (the framework's save_harness_result will
    # then idempotently re-stamp staleness fields without overwriting data).
    if not by_sweep:
        from mosaic.benchmarks.core.console import print_warn

        # NPZ is written unconditionally — matches the original
        # _save_recovery_outputs ordering.
        if per_solver_arrays:
            save_field_snapshots_npz(
                out_dir,
                solver_names,
                per_solver_arrays,
                shared_arrays=shared_dict,
                filename=snapshot_filename,
                prefixes=snapshot_prefixes,
            )
        print_warn(
            f"{out_dir.name}: by_sweep is empty (all solvers excluded or skipped) — "
            "skipping result.json save to preserve existing data"
        )
        existing = try_load_json(out_dir / "result.json") or {
            "sweep_key": sweep_key,
            "by_sweep": {},
            "failure_values": failure_values,
            "params": run,
        }
        return existing

    save_field_snapshots_npz(
        out_dir,
        solver_names,
        per_solver_arrays,
        shared_arrays=shared_dict,
        filename=snapshot_filename,
        prefixes=snapshot_prefixes,
    )

    from mosaic.benchmarks.core.experiment import (
        _build_result_envelope,
        _flatten_by_solver,
    )

    return _build_result_envelope(
        cfg=cfg,
        suite=_kw.get("suite", "optimization"),
        exp_key=_kw.get("exp_key", "recovery"),
        run=run,
        sweep_key=sweep_key,
        sweep_values=sweep_values,
        results=_flatten_by_solver(by_sweep, sweep_key),
        extras={"failure_values": failure_values},
    )


# ── Kernel ───────────────────────────────────────────────────────────────────


@kernel(
    sweep_mode="default",
    ic_sweep=False,
    aggregate_fn=_recovery_aggregate,
    catch_label="recovery failed",
    snapshot_filename="recovery_fields.npz",
    snapshot_prefixes=(
        "ic_rec",
        "ic_history",
        "final_gt",
        "final_rec",
        "final_rep_val",
        "ic_rec_all",
        "final_rec_all",
        "final_perturbed_all",
        "ic_perturbed",
        "final_perturbed",
    ),
)
def recovery(t: Any, ctx: KernelContext) -> dict:
    """One (solver, sweep-value) IC-recovery point.

    Builds per-seed (true, perturbed) ICs, computes the target for the
    current sweep value, runs per-seed gradient-based optimisation against
    each target-forward output, aggregates the per-seed trials into a
    single ``by_sweep`` entry, and emits the per-val visualisation
    snapshots (``ic_rec``, ``final_gt``, ``final_rec``, plus the
    ``ic_history`` history at ``val == rep_val``, and the sigma-only
    ``ic_perturbed`` / ``final_perturbed`` for downstream stacking).

    The framework owns the sweep loop (``sweep_mode="default"``); each
    invocation handles exactly one ``ctx.sweep_value``. Per-seed
    iteration remains inside the kernel since seeds are a trial dimension,
    not a sweep dimension.

    The ``run["optimizer"]`` key selects the inner optimiser:

      * ``"adam"``       — vanilla Adam (default).
      * ``"bfgs"``       — L-BFGS with zoom line-search.
      * ``"bfgs_proj"``  — L-BFGS with the gradient Helmholtz-projected
        onto the ∇·g = 0 subspace each iteration (keeps the search
        direction compatible with incompressibility for velocity-field
        problems).
    """
    from mosaic.benchmarks.core.console import console

    cfg = ctx.cfg
    run = ctx.run
    name = ctx.name
    color = cfg.solver(name).color
    domain_extent = ctx.domain_extent
    make_inputs = ctx.make_inputs
    output_key = ctx.output_key
    make_ic = cfg.make_ic
    error_fn = cfg.error_fn

    # ── Unpack run config ────────────────────────────────────────────────
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
    ic_seeds: list[int] = optim_cfg.get("ic_seeds", [seed])
    record_diagnostics: bool = bool(optim_cfg.get("record_diagnostics", True))
    ic_init_type: str = optim_cfg.get("ic_init_type", "perturb")
    phys = run.get("physics", {})
    is_sigma_sweep = sweep_key == "perturb_sigma"
    is_multi_seed = len(ic_seeds) > 1

    # Experiment key flows in via the run dict — the framework folds any
    # add_experiment kwarg not consumed by run_experiment into each run's
    # payload, so config.py passes ``_exp_key="..."`` per registration.
    exp_key = run.get("_exp_key", "recovery")

    if not sweep_values:
        raise NotImplementedError(
            f"recovery kernel requires sweep.values in runs payload "
            f"(not configured for {cfg.name!r})"
        )

    # The current sweep point. Framework-provided; falls back to the first
    # sweep_value as a defensive default (should never trigger under
    # sweep_mode="default").
    val = ctx.sweep_value if ctx.sweep_value is not None else sweep_values[0]
    rep_val = sweep_values[len(sweep_values) // 2]
    is_rep = val == rep_val

    # ── Build per-seed IC true + perturbed IC (cheap; doesn't depend on val) ─
    ic_true_dict, ic_init_dict, max_div_dict = _build_per_seed_ics(
        ic_name,
        ic_seeds,
        phys,
        perturb_sigma,
        ic_init_type,
        console,
        make_ic=make_ic,
        domain_extent=domain_extent,
    )

    primary_seed = ic_seeds[0]
    ic_true = ic_true_dict[primary_seed]
    max_div = max_div_dict[primary_seed]
    is_vel = _is_velocity_field(np.asarray(ic_true))

    # div_fn for diagnostics: computes max|∇·u| per-iteration inside _run_optim
    div_fn = (
        (lambda u: _max_divergence(np.asarray(u), domain_extent))
        if (is_vel and record_diagnostics)
        else None
    )

    # Optimiser dispatch from run["optimizer"].
    optimizer = run.get("optimizer", "adam")
    optim_fn = _run_lbfgs if optimizer.startswith("bfgs") else _run_optim
    project_grads = optimizer == "bfgs_proj"

    # Gradient projection: Helmholtz-project onto ∇·g = 0 before handing to
    # the optimiser. Only meaningful for velocity fields; ignored otherwise.
    grad_proj_fn = (
        (lambda g: _project_divergence_free_jax(g, domain_extent))
        if (project_grads and is_vel)
        else None
    )

    # Resolve out_dir from the framework's standard layout so partial-result
    # writes land alongside the final ``result.json``. We re-derive rather
    # than reading off ``ctx`` because :class:`KernelContext` doesn't (yet)
    # surface the experiment key; the run dict carries ``_exp_key`` from the
    # add_experiment registration.
    suite = "optimization"
    out_dir = experiment_dir(
        results_dir(),
        cfg.name,
        suite,
        exp_key,
    )
    partial_writer = PartialResultWriter(
        out_dir, base_payload={"sweep_key": sweep_key, "params": run}
    )

    # ── Per-val target setup ────────────────────────────────────────────
    if is_sigma_sweep:
        sigma_target = _precompute_sigma_target(
            name=name,
            t=t,
            ic_true=ic_true,
            phys=phys,
            make_inputs=make_inputs,
            output_key=output_key,
            domain_extent=domain_extent,
            exp_key=exp_key,
        )
        if sigma_target is None:
            # Persist a None entry for this val and ask the framework to
            # short-circuit the remaining sweep (sigma_target failure is a
            # function of name + ic_true + phys, not val — it would fail
            # identically for every subsequent val).
            partial_writer.write(name, None)
            return {"metrics": None, "stop_sweep": True}
        ic_init_at_val = _build_sigma_perturbed_ic(
            ic_true,
            float(val),
            seed,
            is_vel,
            console,
            domain_extent=domain_extent,
        )
        seeds_for_val: list[int] = [primary_seed]
        target_for_seed: dict[int, jax.Array] = {primary_seed: sigma_target}
    else:
        ic_init_at_val = ic_init_dict[primary_seed]
        seeds_for_val = ic_seeds
        target_for_seed = _compute_targets_for_val(
            name,
            exp_key,
            val,
            sweep_key,
            ic_seeds,
            ic_true_dict,
            phys,
            t,
            make_inputs=make_inputs,
            output_key=output_key,
            domain_extent=domain_extent,
        )
        if not target_for_seed:
            partial_writer.write(name, None)
            return {"metrics": None}

    console.print(
        f"  [{color}]{name}[/] {exp_key} {sweep_key}={val} starting "
        f"(max_iters={max_iters})"
    )

    # ── Per-seed trial loop ──────────────────────────────────────────────
    trial_results: list[dict] = []
    primary_hist: list | None = None
    primary_ic_err_hist: list[float] = []

    for s in seeds_for_val:
        if s not in target_for_seed:
            continue
        ic_true_k = ic_true_dict[s]
        ic_init_k = ic_init_dict[s] if not is_sigma_sweep else ic_init_at_val
        target_k = target_for_seed[s]
        is_primary = s == primary_seed
        trial = _run_one_seed_trial(
            name=name,
            t=t,
            color=color,
            val=val,
            s=s,
            ic_init_k=ic_init_k,
            ic_true_k=ic_true_k,
            target_k=target_k,
            is_primary=is_primary,
            phys=phys,
            sweep_key=sweep_key,
            is_sigma_sweep=is_sigma_sweep,
            is_multi_seed=is_multi_seed,
            is_vel=is_vel,
            rep_val=rep_val,
            snap_interval=snap_interval,
            lr=lr,
            max_iters=max_iters,
            patience=patience,
            record_diagnostics=record_diagnostics,
            optim_fn=optim_fn,
            div_fn=div_fn,
            grad_proj_fn=grad_proj_fn,
            make_inputs=make_inputs,
            error_fn=error_fn,
            output_key=output_key,
            domain_extent=domain_extent,
            exp_key=exp_key,
        )
        if trial is None:
            continue
        if is_primary and trial["_hist"]:
            primary_hist = trial["_hist"]
        if is_primary and trial["_ic_err_hist"]:
            primary_ic_err_hist = trial["_ic_err_hist"]
        trial_results.append(trial)

    if not trial_results:
        partial_writer.write(name, None)
        return {"metrics": None}

    entry = _aggregate_trial_results(
        trial_results,
        primary_ic_err_hist,
        val,
        perturb_sigma,
        is_sigma_sweep,
        is_multi_seed,
        failure_threshold,
        max_div,
        ic_true,
    )

    # ── Partial-result checkpoint ────────────────────────────────────────
    # Merge this val's entry into the on-disk ``by_sweep[name]`` slice so a
    # crash mid-sweep preserves the previous vals' progress. We read +
    # merge + write inside the lock-acquiring writer; the read-before-write
    # for THIS solver is race-free because each solver's worker is the sole
    # writer of its own ``by_solver[name]`` entry.
    existing = try_load_json(out_dir / "result_partial.json") or {}
    prev_for_solver = (existing.get("by_solver") or {}).get(name) or {}
    # JSON keys round-trip as strings; legacy code kept them keyed by the
    # raw value in-memory. Coerce both forms when merging so we don't
    # duplicate entries across runs.
    coerced: dict = {}
    for k, v in prev_for_solver.items():
        try:
            coerced[type(val)(k)] = v
        except (TypeError, ValueError):
            coerced[k] = v
    coerced[val] = entry
    partial_writer.write(name, coerced)

    # ── Per-val snapshots ────────────────────────────────────────────────
    prim_trial = next(
        (r for r in trial_results if r["ic_seed"] == primary_seed),
        trial_results[0],
    )
    ic_opt = prim_trial["ic_opt"]

    # Per-val visualisation arrays. ``ic_rec`` is the optimised IC at this
    # val; ``final_gt`` is the target; ``final_rec`` is the forward of the
    # optimised IC. The aggregate stacks these per solver into
    # ``ic_rec_all`` / ``final_rec_all`` and picks single-array showcases
    # (``ic_rec``, ``final_gt``, ``final_rec``, ``final_rep_val``) from the
    # per-val arrays at rep_val / best-conv val.
    snaps: dict[str, np.ndarray] = {
        "ic_rec": np.asarray(ic_opt),
        "final_gt": np.asarray(prim_trial["target"]),
    }
    fphys = prim_trial["phys_kw"]
    try:
        inp_rec = make_inputs(name, ic_opt, domain_extent=domain_extent, **fphys)
        snaps["final_rec"] = np.asarray(apply_tesseract(t, inp_rec)[output_key])
    except Exception:
        snaps["final_rec"] = np.zeros_like(np.asarray(ic_opt))

    if is_rep and primary_hist:
        snaps["ic_history"] = np.asarray(primary_hist)

    if is_sigma_sweep:
        # ``ic_perturbed``: the perturbed IC at this sigma (for the shared
        # ``ic_perturbed_all`` stack). ``final_perturbed``: forward-solve of
        # that perturbed IC (for the per-solver ``final_perturbed_all`` stack).
        snaps["ic_perturbed"] = np.asarray(ic_init_at_val)
        try:
            inp_p = make_inputs(
                name, ic_init_at_val, domain_extent=domain_extent, **phys
            )
            snaps["final_perturbed"] = np.asarray(apply_tesseract(t, inp_p)[output_key])
        except Exception:
            snaps["final_perturbed"] = np.zeros_like(np.asarray(ic_true))

    # ── Shared NPZ payload ───────────────────────────────────────────────
    # Framework's shared_extras uses setdefault, so only the first solver's
    # contribution sticks. ``ic_init`` prefers the actual init at rep_val so
    # the visualisation matches the showcase; non-rep calls still set it as
    # a fallback (any single contribution suffices since the shared key is
    # deterministic across solvers).
    shared: dict[str, np.ndarray] = {
        "rep_val": np.array([rep_val]),
        "sweep_values": np.array(sweep_values, dtype=float),
        "ic_true": np.asarray(ic_true),
    }
    if is_rep:
        shared["ic_init"] = np.asarray(ic_init_at_val)

    return {
        "metrics": entry,
        "snapshots": snaps,
        "shared": shared,
    }
