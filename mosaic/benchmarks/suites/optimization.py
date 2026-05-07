"""Optimization suite: IC recovery via gradient descent.

Only runs solvers where SolverSpec.differentiable is True.

Run from the terminal:
    cd mosaic
    python -m benchmarks.suites.optimization [--experiment EXPR] [--no-plots]
"""

from __future__ import annotations

import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mosaic.benchmarks.core.config import ProblemConfig
from mosaic.benchmarks.core.runner import run_with_gpu_pool
from mosaic.benchmarks.core.utils import (
    _diff_solvers,
    _has_vjp,
    experiment_dir,
    extract_runs,
    is_valid,
    iter_runs,
    results_dir,
    save_experiment,
    save_gradient_fields_npz,
    save_json,
)

# JAX-traced loss_fn closures capture this reference at trace time;
# using the tracer-aware wrapper ensures primitive binding sees the
# active trace.
from mosaic.benchmarks.core.tracer_apply import apply_tesseract

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
    v = u.reshape(u.shape[:nd] + (nd,)) if squeeze else u

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
    v = u.reshape(u.shape[:nd] + (nd,)) if squeeze else u

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


def _run_optim(
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


def _run_lbfgs(
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


def _topopt_solvers(cfg: ProblemConfig) -> list[str]:
    """Solvers that can differentiate compliance.

    Checks spec.compliance_differentiable first (explicit override), then
    spec.differentiable (general flag), then falls back to runtime VJP endpoint
    detection.  Uses getattr with None defaults so that SolverSpec subclasses
    that omit these optional fields are handled gracefully.
    """
    result = []
    for name, spec in cfg.solvers.items():
        cd = getattr(spec, "compliance_differentiable", None)
        if cd is None:
            cd = getattr(spec, "differentiable", None)
        if cd is None:
            # Fall back to runtime VJP endpoint detection (same as _diff_solvers)
            cd = _has_vjp(spec)
        if cd:
            result.append(name)
    return result


def _run_recovery_long_impl(
    cfg: ProblemConfig,
    tags: dict[str, str],
    exp_key: str,
    harness_fn,
    *,
    _optim_fn=_run_optim,
    _project_grads: bool = False,
    **overrides,
) -> dict:
    """Shared implementation for run_recovery_long and variants."""
    runs = cfg.inverse_defaults.get(exp_key, [])
    if not runs:
        raise NotImplementedError(
            f"No '{exp_key}' inverse_defaults configured for '{cfg.name}'"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
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

        if not sweep_values:
            raise NotImplementedError(
                f"'{exp_key}' requires sweep.values in inverse_defaults "
                f"(not configured for '{cfg.name}')"
            )

        from mosaic.benchmarks.core.console import console

        # ── IC helpers ────────────────────────────────────────────────────────
        def _project_ic(raw: jax.Array, label: str, is_vel: bool) -> jax.Array:
            """Project to divergence-free and log; no-op for non-velocity fields."""
            if not is_vel:
                return raw
            raw_np = np.asarray(raw)
            div_before = _max_divergence(raw_np, cfg.domain_extent)
            projected = _project_divergence_free(raw_np, cfg.domain_extent)
            div_after = _max_divergence(projected, cfg.domain_extent)
            console.print(
                f"  [dim]{label}[/dim]  "
                f"max|∇·u| {div_before:.2e} → {div_after:.2e} (after projection)"
            )
            return jnp.asarray(projected)

        # ── Build per-seed IC true + perturbed IC ─────────────────────────────
        # Each ic_seed gives a different ground-truth IC.
        # The perturbed IC is the true IC + σ·noise, projected back to div-free.
        _ic_true_dict: dict[int, jax.Array] = {}
        _ic_init_dict: dict[int, jax.Array] = {}
        _max_div_dict: dict[int, float | None] = {}

        for _s in ic_seeds:
            _ic_k = jnp.array(
                cfg.make_ic[ic_name](L=cfg.domain_extent, seed=_s, **phys)
            )
            _is_vel_k = _is_velocity_field(np.asarray(_ic_k))
            # Project true IC
            _ic_k = _project_ic(_ic_k, f"IC seed={_s} (true)", _is_vel_k)
            _max_div_dict[_s] = (
                _max_divergence(np.asarray(_ic_k), cfg.domain_extent)
                if _is_vel_k
                else None
            )
            _ic_true_dict[_s] = _ic_k
            if ic_init_type == "zeros":
                _ic_init_dict[_s] = jnp.zeros_like(_ic_k)
            else:
                # Perturbed IC: for velocity fields use the same div-free generator with a
                # different seed so the perturbation is exactly div-free by construction.
                # For non-velocity fields fall back to Gaussian noise + projection.
                _noise_seed = _s + 1000
                if _is_vel_k:
                    _noise = jnp.array(
                        cfg.make_ic[ic_name](
                            L=cfg.domain_extent, seed=_noise_seed, **phys
                        )
                    )
                    _raw_init = _ic_k + perturb_sigma * _noise
                else:
                    _raw_init = _ic_k + perturb_sigma * jax.random.normal(
                        jax.random.PRNGKey(_noise_seed), _ic_k.shape, dtype=jnp.float32
                    )
                _ic_init_dict[_s] = _project_ic(
                    _raw_init, f"IC seed={_s} (perturbed, σ={perturb_sigma})", _is_vel_k
                )

        # Primary IC / init for single-seed and for visualization (always seed 0 / ic_seeds[0])
        _primary_seed = ic_seeds[0]
        ic_true = _ic_true_dict[_primary_seed]
        max_div = _max_div_dict[_primary_seed]
        _is_vel = _is_velocity_field(np.asarray(ic_true))

        # div_fn for diagnostics: computes max|∇·u| per-iteration inside _run_optim
        _div_fn = (
            (lambda u: _max_divergence(np.asarray(u), cfg.domain_extent))
            if (_is_vel and record_diagnostics)
            else None
        )

        # Gradient projection: Helmholtz-project onto ∇·g = 0 before handing to
        # the optimiser. Only meaningful for velocity fields; ignored otherwise.
        _grad_proj_fn = (
            (lambda g: _project_divergence_free(g, cfg.domain_extent))
            if (_project_grads and _is_vel)
            else None
        )

        # Keep a single _make_div_free_ic shim for the sigma-sweep path
        def _make_div_free_ic(raw: jax.Array, label: str) -> jax.Array:
            return _project_ic(raw, label, _is_vel)

        if not _is_sigma_sweep:
            ic_init = _ic_init_dict[_primary_seed]
        else:
            ic_init = None
            _sigma_ics: dict = {}
            for _sv in sweep_values:
                _sigma_val = float(_sv)
                _key_sv = jax.random.fold_in(
                    jax.random.PRNGKey(seed), int(_sigma_val * 1000)
                )
                _raw = ic_true + _sigma_val * jax.random.normal(
                    _key_sv, ic_true.shape, dtype=jnp.float32
                )
                _sigma_ics[_sv] = _make_div_free_ic(_raw, f"σ={_sigma_val}")

        rep_val = sweep_values[len(sweep_values) // 2]
        ic_snaps: dict = {}
        ic_init_snaps: dict = {}
        ic_histories: dict = {}
        final_states_gt: dict = {}
        final_states_rec: dict = {}
        final_states_rep_val: dict = {}  # per-solver val actually used for final states
        all_ic_snaps: dict = {}  # per-solver stacked (n_sigmas, *ic_shape)
        all_final_rec_snaps: dict = {}  # per-solver stacked (n_sigmas, *field_shape)
        all_final_perturbed_snaps: dict = {}  # per-solver stacked (n_sigmas, *field_shape)
        by_sweep: dict = {}
        _wall_times: dict[str, float] = {}
        gpu_ids = overrides.get("gpu_ids")

        out_dir = experiment_dir(
            results_dir(),
            cfg.name,
            _SUITE,
            f"{exp_key}/{ic_subdir}" if ic_subdir else exp_key,
            suffix="_debug" if overrides.get("debug") else "",
        )
        _partial_lock = threading.Lock()

        def _write_partial() -> None:
            partial = {
                "sweep_key": sweep_key,
                "by_sweep": by_sweep,
                "params": run,
            }
            with _partial_lock:
                save_json(partial, out_dir / "result_partial.json")

        def _recovery_long_work(name: str, t) -> None:
            from mosaic.benchmarks.core.console import console

            _color = cfg.solvers[name].color
            _t0 = time.perf_counter()
            by_sweep[name] = {}
            # Track the last converged (val, ic_opt, target, ic_init) for final
            # state plots — use the hardest case that still converged rather than
            # the blind median rep_val which may be a failed recovery.
            _best_conv: dict = {}
            _all_ic_opts: dict = {}  # val -> ic_opt for all completed sweep values
            console.print(
                f"  [{_color}]{name}[/] {exp_key} starting ({len(sweep_values)} sweep values, max_iters={max_iters})"
            )

            if _is_sigma_sweep:
                try:
                    inputs_true_fixed = cfg.make_inputs(
                        name,
                        ic_true,
                        domain_extent=cfg.domain_extent,
                        **phys,
                    )
                    target = apply_tesseract(t, inputs_true_fixed)[cfg.output_key]
                except Exception as exc:
                    from mosaic.benchmarks.core.console import print_warn

                    print_warn(
                        f"{name} {exp_key} target forward failed: {exc} — "
                        f"marking all {len(sweep_values)} sweep values as None"
                    )
                    for val in sweep_values:
                        by_sweep[name][val] = None
                    _wall_times[name] = time.perf_counter() - _t0
                    return
                if not is_valid(target):
                    for val in sweep_values:
                        by_sweep[name][val] = None
                    _wall_times[name] = time.perf_counter() - _t0
                    return

            for val in sweep_values:
                if _is_sigma_sweep:
                    # sigma sweep: single IC seed, target pre-computed above
                    _ic_init = _sigma_ics[val]
                    _seeds_for_val: list[int] = [_primary_seed]
                    _target_for_seed: dict[int, jax.Array] = {_primary_seed: target}  # type: ignore[assignment]
                else:
                    # steps / other sweep: compute target per ic_seed
                    _seeds_for_val = ic_seeds
                    _target_for_seed = {}
                    for _s in ic_seeds:
                        try:
                            _inp_true = cfg.make_inputs(
                                name,
                                _ic_true_dict[_s],
                                domain_extent=cfg.domain_extent,
                                **{**phys, sweep_key: val},
                            )
                            _tgt = apply_tesseract(t, _inp_true)[cfg.output_key]
                        except Exception as exc:
                            from mosaic.benchmarks.core.console import print_warn

                            print_warn(
                                f"{name} {exp_key} target forward failed at {sweep_key}={val} ic_seed={_s}: {exc}"
                            )
                            continue
                        if is_valid(_tgt):
                            _target_for_seed[_s] = _tgt

                    if not _target_for_seed:
                        by_sweep[name][val] = None
                        _write_partial()
                        continue

                # ── Per-seed optimization trials ──────────────────────────────
                # For the primary (first) seed, capture history/snaps for visualization.
                _trial_results: list[dict] = []
                _primary_hist: list | None = None
                _primary_ic_err_hist: list[float] = []

                for _s in _seeds_for_val:
                    if _s not in _target_for_seed:
                        continue
                    _ic_true_k = _ic_true_dict[_s]
                    _ic_init_k = _ic_init_dict[_s] if not _is_sigma_sweep else _ic_init  # type: ignore[assignment]
                    _target_k = _target_for_seed[_s]
                    _is_primary = _s == _primary_seed

                    def loss_fn(ic, _t=t, _target=_target_k, _val=val):
                        _phys_kw = (
                            phys if _is_sigma_sweep else {**phys, sweep_key: _val}
                        )
                        inp = cfg.make_inputs(
                            name,
                            ic,
                            domain_extent=cfg.domain_extent,
                            **_phys_kw,
                        )
                        return jnp.mean(
                            (apply_tesseract(_t, inp)[cfg.output_key] - _target) ** 2
                        )

                    _hist: list | None = (
                        []
                        if (snap_interval > 0 and val == rep_val and _is_primary)
                        else None
                    )
                    _ic_err_hist: list[float] = (
                        [] if (snap_interval > 0 and _is_primary) else []
                    )
                    ic_error_init = float(cfg.error_fn(_ic_init_k, _ic_true_k))
                    seed_tag = f" [ic_seed={_s}]" if _is_multi_seed else ""
                    console.print(
                        f"  [{_color}]{name}[/] {sweep_key}={val}{seed_tag} "
                        f"optim start (init_err={ic_error_init:.4g})"
                    )

                    def _log_iter(i, loss, _n=name, _c=_color, _v=val, _st=seed_tag):
                        console.print(
                            f"  [{_c}]{_n}[/] {sweep_key}={_v}{_st} iter {i}/{max_iters} loss={loss:.4g}"
                        )

                    try:
                        ic_opt, errors, diag = _optim_fn(
                            loss_fn,
                            _ic_init_k,
                            lr,
                            max_iters,
                            patience,
                            snap_interval=snap_interval if snap_interval > 0 else 0,
                            history=_hist,
                            snap_error_fn=lambda ic, _ict=_ic_true_k: float(
                                cfg.error_fn(ic, _ict)
                            ),
                            error_history=_ic_err_hist
                            if (snap_interval > 0 and _is_primary)
                            else None,
                            log_fn=_log_iter,
                            record_diagnostics=record_diagnostics,
                            div_fn=_div_fn,
                            grad_proj_fn=_grad_proj_fn,
                        )
                    except Exception as exc:
                        from mosaic.benchmarks.core.console import print_warn

                        print_warn(
                            f"{name} {exp_key} optim failed at {sweep_key}={val}{seed_tag}: {exc}"
                        )
                        continue
                    final_ic_error = cfg.error_fn(ic_opt, _ic_true_k)
                    final_ic_div = (
                        _max_divergence(np.asarray(ic_opt), cfg.domain_extent)
                        if _is_vel
                        else None
                    )
                    console.print(
                        f"  [{_color}]{name}[/] {sweep_key}={val}{seed_tag} done "
                        f"iters={len(errors)} final_loss={errors[-1]:.4g} ic_err={final_ic_error:.4g}"
                    )
                    _trial_results.append(
                        {
                            "errors": errors,
                            "diag": diag,
                            "ic_error_history": _ic_err_hist if _is_primary else [],
                            "ic_error_init": ic_error_init,
                            "final_ic_error": float(final_ic_error),
                            "final_ic_div": final_ic_div,
                            "ic_opt": ic_opt,
                            "target": _target_k,
                            "ic_init": _ic_init_k,
                            "phys_kw": phys
                            if _is_sigma_sweep
                            else {**phys, sweep_key: val},
                            "ic_seed": _s,
                        }
                    )
                    if _is_primary and _hist:
                        _primary_hist = _hist
                    if _is_primary and _ic_err_hist:
                        _primary_ic_err_hist = _ic_err_hist

                if not _trial_results:
                    by_sweep[name][val] = None
                    _write_partial()
                    continue

                # ── Aggregate across seeds ────────────────────────────────────
                _all_fice = [r["final_ic_error"] for r in _trial_results]
                _first = _trial_results[0]
                _first_diag = _first.get("diag") or {}
                by_sweep[name][val] = {
                    "errors": _first["errors"],
                    "grad_norms": _first_diag.get("grad_norms"),
                    "grad_divs": _first_diag.get("grad_divs"),
                    "ic_divs": _first_diag.get("ic_divs"),
                    "ic_error_history": _primary_ic_err_hist,
                    "ic_error_init": float(
                        np.mean([r["ic_error_init"] for r in _trial_results])
                    ),
                    "final_ic_error": float(np.mean(_all_fice)),
                    "final_ic_error_std": float(np.std(_all_fice))
                    if _is_multi_seed
                    else None,
                    "final_ic_error_trials": _all_fice if _is_multi_seed else None,
                    "final_ic_div": float(
                        np.nanmean(
                            [
                                r["final_ic_div"]
                                for r in _trial_results
                                if r["final_ic_div"] is not None
                            ]
                        )
                    )
                    if any(r["final_ic_div"] is not None for r in _trial_results)
                    else None,
                    "converged": np.mean(
                        [
                            r["final_ic_error"] < failure_threshold
                            for r in _trial_results
                        ]
                    )
                    > 0.5,
                    "perturb_sigma": float(val) if _is_sigma_sweep else perturb_sigma,
                    "max_div_ic": max_div if ic_true.ndim == 4 else None,
                    "n_trials": len(_trial_results),
                    "final_loss": float(
                        np.mean(
                            [r["errors"][-1] for r in _trial_results if r.get("errors")]
                        )
                    ),
                    "final_loss_trials": [
                        r["errors"][-1] for r in _trial_results if r.get("errors")
                    ]
                    if _is_multi_seed
                    else None,
                }
                _write_partial()

                # ── Visualization snaps (use primary seed trial) ──────────────
                _prim_trial = next(
                    (r for r in _trial_results if r["ic_seed"] == _primary_seed),
                    _trial_results[0],
                )
                ic_opt = _prim_trial["ic_opt"]
                target = _prim_trial["target"]
                _ic_init = _prim_trial["ic_init"]
                _all_ic_opts[val] = ic_opt
                if val == rep_val:
                    ic_snaps[name] = ic_opt
                    ic_init_snaps[name] = _ic_init
                    if _primary_hist:
                        ic_histories[name] = np.asarray(_primary_hist)
                if _prim_trial["final_ic_error"] < failure_threshold:
                    _best_conv = {
                        "val": val,
                        "ic_opt": ic_opt,
                        "target": target,
                        "ic_init": _ic_init,
                        "phys_kw": _prim_trial["phys_kw"],
                    }

            # Collect final states from the hardest converged val; fall back to rep_val.
            _fv = _best_conv.get("val", rep_val)
            _fic = _best_conv.get("ic_opt", ic_snaps.get(name))
            _ftgt = _best_conv.get("target")
            _fphys = _best_conv.get("phys_kw", phys)
            if _fic is not None and _ftgt is not None:
                final_states_gt[name] = np.asarray(_ftgt)
                final_states_rep_val[name] = _fv
                try:
                    inp_rec = cfg.make_inputs(
                        name,
                        _fic,
                        domain_extent=cfg.domain_extent,
                        **_fphys,
                    )
                    final_states_rec[name] = np.asarray(
                        apply_tesseract(t, inp_rec)[cfg.output_key]
                    )
                except Exception:
                    pass

            # Build per-sigma stacks for the all-sigma grid plot.
            # For sigma sweep the target (GT final) is the same for all sigmas.
            _ic_stack, _fr_stack = [], []
            for _v in sweep_values:
                _io = _all_ic_opts.get(_v)
                if _io is not None:
                    _ic_stack.append(np.asarray(_io))
                    _kw = phys if _is_sigma_sweep else {**phys, sweep_key: _v}
                    try:
                        _inp = cfg.make_inputs(
                            name, _io, domain_extent=cfg.domain_extent, **_kw
                        )
                        _fr_stack.append(
                            np.asarray(apply_tesseract(t, _inp)[cfg.output_key])
                        )
                    except Exception:
                        _fr_stack.append(np.zeros_like(np.asarray(_io)))
                else:
                    _ic_stack.append(np.zeros_like(np.asarray(ic_true)))
                    _fr_stack.append(np.zeros_like(np.asarray(ic_true)))
            if _ic_stack:
                all_ic_snaps[name] = np.stack(_ic_stack)
                all_final_rec_snaps[name] = np.stack(_fr_stack)
            # Build per-sigma stack of forward rollouts from perturbed ICs.
            if _is_sigma_sweep:
                _fp_stack = []
                for _v in sweep_values:
                    _ip = _sigma_ics.get(_v)
                    if _ip is not None:
                        try:
                            _inp_p = cfg.make_inputs(
                                name, _ip, domain_extent=cfg.domain_extent, **phys
                            )
                            _fp_stack.append(
                                np.asarray(apply_tesseract(t, _inp_p)[cfg.output_key])
                            )
                        except Exception:
                            _fp_stack.append(np.zeros_like(np.asarray(ic_true)))
                    else:
                        _fp_stack.append(np.zeros_like(np.asarray(ic_true)))
                if _fp_stack:
                    all_final_perturbed_snaps[name] = np.stack(_fp_stack)
            _wall_times[name] = time.perf_counter() - _t0

        run_with_gpu_pool(
            _diff_solvers(cfg, "optimization", exp_key),
            tags,
            _recovery_long_work,
            gpu_ids=gpu_ids,
        )

        failure_values: dict = {}
        for name, s_results in by_sweep.items():
            fail_val = None
            for val in sweep_values:
                r = s_results.get(val)
                if r is None or not r["converged"]:
                    fail_val = val
                    break
            failure_values[name] = fail_val

        solver_names = list(ic_snaps.keys())
        _rep_ic_init = (
            ic_init_snaps.get(solver_names[0], ic_init)
            if ic_init_snaps and solver_names
            else ic_init
        )
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
                entry["final_perturbed_all:"] = np.asarray(
                    all_final_perturbed_snaps[sname]
                )
            per_solver_arrays[sname] = entry
        # Store the ground-truth final state (same for all sigmas in sigma sweep)
        _shared_final_gt = (
            np.asarray(final_states_gt[solver_names[0]])
            if solver_names and solver_names[0] in final_states_gt
            else None
        )
        shared: dict = {
            "rep_val": np.array([rep_val]),
            "sweep_values": np.array(sweep_values, dtype=float),
            "ic_true": np.asarray(ic_true),
            "ic_init": np.asarray(_rep_ic_init)
            if _rep_ic_init is not None
            else np.asarray(ic_true),
        }
        if _shared_final_gt is not None:
            shared["final_gt_shared"] = _shared_final_gt
        if _is_sigma_sweep and _sigma_ics:
            shared["ic_perturbed_all"] = np.stack(
                [np.asarray(_sigma_ics[_sv]) for _sv in sweep_values]
            )
        save_gradient_fields_npz(
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

        result = {
            "sweep_key": sweep_key,
            "by_sweep": by_sweep,
            "failure_values": failure_values,
            "params": run,
        }
        if not by_sweep:
            from mosaic.benchmarks.core.console import print_warn

            print_warn(
                f"{exp_key}: by_sweep is empty (all solvers excluded or skipped) — "
                "skipping result.json save to preserve existing data"
            )
        else:
            save_experiment(
                result, out_dir, cfg=cfg, harness_fn=harness_fn, wall_time_s=_wall_times
            )
        if n_runs > 1:
            all_results[ic_name] = result
        else:
            all_results = result

    return all_results


def run_recovery_constant_ic(
    cfg: ProblemConfig, tags: dict[str, str], **overrides
) -> dict:
    """IC recovery from zero initialisation (cold start), fixed steps=100."""
    return _run_recovery_long_impl(
        cfg, tags, "recovery_constant_ic", run_recovery_constant_ic, **overrides
    )


def run_recovery_constant_ic_bfgs(
    cfg: ProblemConfig, tags: dict[str, str], **overrides
) -> dict:
    """L-BFGS variant of run_recovery_constant_ic. Reads inverse_defaults["recovery_constant_ic_bfgs"]."""
    return _run_recovery_long_impl(
        cfg,
        tags,
        "recovery_constant_ic_bfgs",
        run_recovery_constant_ic_bfgs,
        _optim_fn=_run_lbfgs,
        **overrides,
    )


def run_recovery_constant_ic_bfgs_proj(
    cfg: ProblemConfig, tags: dict[str, str], **overrides
) -> dict:
    """L-BFGS variant with gradient Helmholtz-projected onto ∇·g = 0 each iteration.

    Reads inverse_defaults["recovery_constant_ic_bfgs_proj"]. Identical to
    run_recovery_constant_ic_bfgs except that the L-BFGS gradient is
    spectral-projected onto the divergence-free subspace before each quasi-Newton
    update, keeping the search direction compatible with the incompressibility
    constraint throughout optimisation.
    """
    return _run_recovery_long_impl(
        cfg,
        tags,
        "recovery_constant_ic_bfgs_proj",
        run_recovery_constant_ic_bfgs_proj,
        _optim_fn=_run_lbfgs,
        _project_grads=True,
        **overrides,
    )


# ── Topology optimisation ─────────────────────────────────────────────────────


def run_topopt(cfg: ProblemConfig, tags: dict[str, str], **overrides) -> dict:
    """Topology optimisation: minimise compliance subject to a volume fraction constraint.

    Runs Adam gradient descent with ρ clipped to [x_min, 1] and a soft volume
    penalty. Designed for static FEA problems (structural-mesh) where IC recovery
    is degenerate.

    Returns:
        {"by_solver": {solver: {"compliances", "vol_fracs", "final_compliance",
                                "final_vol_frac", "n_iters", "converged"}},
         "params": run}
        or {ic_name: <above>} when multiple runs are configured.
    """
    runs = cfg.inverse_defaults.get("topopt", [])
    if not runs:
        raise NotImplementedError(
            f"No 'topopt' inverse_defaults configured for '{cfg.name}'"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
        seed = ic_cfg.get("seed", 0)
        phys = run.get("physics", {})
        optim_cfg = run.get("optim", {})
        lr = optim_cfg.get("lr", 1e-2)
        max_iters = optim_cfg.get("max_iters", 200)
        patience = optim_cfg.get("patience", 30)
        v_frac = phys.get("v_frac")
        compliance_key = phys.get("compliance_key", "compliance")
        penalty_weight = phys.get("penalty_weight", 100.0)
        x_min = phys.get("x_min", 1e-3)
        snap_interval = int(phys.get("snap_interval", 0))
        ic_subdir = ic_name if n_runs > 1 else ""

        if v_frac is None:
            raise NotImplementedError(
                f"'topopt' requires physics.v_frac in inverse_defaults "
                f"(not configured for '{cfg.name}')"
            )

        rho_init = cfg.make_ic[ic_name](rho_0=v_frac, seed=seed, **phys)

        by_solver: dict = {}
        rho_snaps: dict = {}
        rho_histories: dict = {}
        _wall_times: dict[str, float] = {}
        gpu_ids = overrides.get("gpu_ids")

        def _topopt_work(name: str, t) -> None:
            _t0 = time.perf_counter()
            rho_opt = rho_init
            optimizer = optax.adam(lr)
            opt_state = optimizer.init(rho_opt)
            compliances, vol_fracs = [], []
            best_c, no_improve = jnp.inf, 0
            rho_history: list = []
            grad_norms: list[float] = []

            def loss_fn(rho, _t=t):
                inp = cfg.make_inputs(name, rho, **phys)
                compliance = apply_tesseract(_t, inp)[compliance_key]
                vol_penalty = penalty_weight * (jnp.mean(rho) - v_frac) ** 2
                return compliance + vol_penalty, compliance

            for i in range(max_iters):
                (_, compliance_val), g = jax.value_and_grad(loss_fn, has_aux=True)(
                    rho_opt
                )
                grad_norms.append(float(jnp.linalg.norm(g.ravel())))
                updates, opt_state = optimizer.update(g, opt_state)
                rho_opt = jnp.clip(optax.apply_updates(rho_opt, updates), x_min, 1.0)
                c = float(compliance_val)
                compliances.append(c)
                vol_fracs.append(float(jnp.mean(rho_opt)))
                if snap_interval > 0 and (i + 1) % snap_interval == 0:
                    rho_history.append(np.array(rho_opt))
                if c < best_c:
                    best_c, no_improve = c, 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break

            rho_snaps[name] = np.array(rho_opt)
            rho_histories[name] = rho_history
            by_solver[name] = {
                "compliances": compliances,
                "vol_fracs": vol_fracs,
                "final_compliance": compliances[-1],
                "final_vol_frac": vol_fracs[-1],
                "n_iters": len(compliances),
                "converged": no_improve >= patience,
                "grad_norms": grad_norms,
            }
            _wall_times[name] = time.perf_counter() - _t0

        run_with_gpu_pool(_topopt_solvers(cfg), tags, _topopt_work, gpu_ids=gpu_ids)

        out_dir = experiment_dir(
            results_dir(),
            cfg.name,
            _SUITE,
            f"topopt/{ic_subdir}" if ic_subdir else "topopt",
            suffix="_debug" if overrides.get("debug") else "",
        )
        solver_names = list(rho_snaps.keys())
        per_solver: dict[str, dict[str, np.ndarray]] = {}
        for sname in solver_names:
            entry: dict[str, np.ndarray] = {"rho_final:": np.asarray(rho_snaps[sname])}
            if rho_histories[sname]:
                entry["rho_history:"] = np.asarray(rho_histories[sname])
            per_solver[sname] = entry
        save_gradient_fields_npz(
            out_dir,
            solver_names,
            per_solver,
            shared_arrays={"rho_init": np.array(rho_init)},
            filename="topopt_fields.npz",
            prefixes=("rho_final", "rho_history"),
        )

        result = {"by_solver": by_solver, "params": run}
        save_experiment(
            result, out_dir, cfg=cfg, harness_fn=run_topopt, wall_time_s=_wall_times
        )
        if n_runs > 1:
            all_results[ic_name] = result
        else:
            all_results = result

    return all_results


def run_drag_opt(cfg: ProblemConfig, tags: dict[str, str], **overrides) -> dict:
    """Inflow profile optimisation: minimise drag on an embedded obstacle via Adam.

    Optimises the ``inflow_profile`` input field (1-D inlet velocity u_x(y)) to
    minimise the scalar ``drag`` output.  A flow-rate conservation penalty is
    added to prevent the optimiser from trivially reducing drag by zeroing the
    inflow:  L = drag + flow_penalty_weight * (mean(profile) - U_mean)².

    Expects ``inverse_defaults["drag_opt"]`` runs with:
        name: str               — used as result subdir when multiple runs present
        ic: {name, seed}        — IC generator returning 1-D profile, shape (N,)
        physics: {N, nu, dt, steps, domain_extent, U_mean, obstacle, ...}
        optim: {lr, max_iters, patience, flow_penalty_weight}
    """
    runs = cfg.inverse_defaults.get("drag_opt", [])
    if not runs:
        raise NotImplementedError(
            f"No 'drag_opt' inverse_defaults configured for '{cfg.name}'"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
        seed = ic_cfg.get("seed", 0)
        phys = run.get("physics", {})
        optim_cfg = run.get("optim", {})
        lr = optim_cfg.get("lr", 1e-3)
        max_iters = optim_cfg.get("max_iters", 150)
        patience = optim_cfg.get("patience", 30)
        flow_penalty_weight = optim_cfg.get("flow_penalty_weight", 50.0)
        # Snap every ``snap_interval`` iterations so the plotter can animate
        # the inflow-profile evolution; 0 disables capture.
        snap_interval = int(optim_cfg.get("snap_interval", 0))
        U_mean = float(phys.get("U_mean", 0.5))
        run_name = run.get("name", ic_name)
        ic_subdir = run_name if n_runs > 1 else ""
        gpu_ids = overrides.get("gpu_ids")

        profile_init = jnp.array(
            cfg.make_ic[ic_name](L=cfg.domain_extent, seed=seed, **phys)
        )

        by_solver: dict = {}
        profile_snaps: dict = {}
        profile_histories: dict = {}
        # Velocity fields per solver, shape (N, N, 1, 2).
        # flow_init_snaps: initial (unoptimised) flow per solver.
        # flow_snaps:      final (optimised) flow per solver.
        flow_init_snaps: dict = {}
        flow_snaps: dict = {}
        _wall_times: dict[str, float] = {}

        # Compute the output directory up front so the optimisation loop can
        # checkpoint partial results into ``result_partial.json``. Without this,
        # a killed / interrupted long run loses every iteration recorded so far.
        _dbg_partial = "_debug" if overrides.get("debug") else ""
        if ic_subdir:
            _partial_parent = experiment_dir(
                results_dir(), cfg.name, _SUITE, "drag_opt", suffix=_dbg_partial
            )
            partial_out_dir = _partial_parent / ic_subdir
            partial_out_dir.mkdir(parents=True, exist_ok=True)
        else:
            partial_out_dir = experiment_dir(
                results_dir(), cfg.name, _SUITE, "drag_opt", suffix=_dbg_partial
            )
        _partial_lock = threading.Lock()

        def _write_partial() -> None:
            """Snapshot the current by_solver dict to result_partial.json.

            Mirrors the recovery_constant_ic harness so an interrupted drag_opt
            (long PICT runs in particular) preserves per-iteration progress.
            """
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
            _t0 = time.perf_counter()
            profile = profile_init
            optimizer = optax.adam(lr)
            opt_state = optimizer.init(profile)
            drags, flow_rates = [], []
            best_drag, no_improve = jnp.inf, 0
            profile_history: list = []
            grad_norms: list[float] = []

            # Capture the *initial* (unoptimised) velocity field before training.
            try:
                _de0 = phys.get("domain_extent", cfg.domain_extent)
                _inp0 = cfg.make_inputs(
                    name,
                    profile_init,
                    domain_extent=_de0,
                    **{k: v for k, v in phys.items() if k != "domain_extent"},
                )
                _out0 = apply_tesseract(t, _inp0)
                _vel0 = _out0.get("result")
                if _vel0 is not None:
                    flow_init_snaps[name] = np.array(_vel0)
            except Exception:
                pass

            def loss_fn(p, _t=t):
                # domain_extent may already be inside phys (drag_opt sets it to 1.0);
                # avoid passing it twice by letting phys take precedence.
                _de = phys.get("domain_extent", cfg.domain_extent)
                inp = cfg.make_inputs(
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
                # Use abs(drag) so the loss works regardless of sign convention:
                # solvers that return positive drag (pict, su2) and negative drag
                # (xlb, phiflow) are both minimised correctly.
                return jnp.abs(jnp.squeeze(drag_val)) + flow_penalty, jnp.squeeze(
                    drag_val
                )

            non_finite_grad = False
            for i in range(max_iters):
                (_, drag_val), g = jax.value_and_grad(loss_fn, has_aux=True)(profile)
                # NaN/Inf gradient detection (Option B): if the VJP returns
                # non-finite gradients on the first step, the adjoint is broken
                # for this solver/experiment combination.  Break immediately so
                # the patience counter cannot tick up 30 times and record
                # converged=True with an all-NaN drag trajectory.
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
                profile = jnp.clip(
                    optax.apply_updates(profile, updates), 0.0, 3.0 * U_mean
                )
                d = float(drag_val)
                drags.append(d)
                flow_rates.append(float(jnp.mean(profile)))
                if snap_interval > 0 and (i + 1) % snap_interval == 0:
                    profile_history.append(np.asarray(profile))
                if abs(d) < best_drag:
                    best_drag, no_improve = abs(d), 0
                else:
                    no_improve += 1
                # Snapshot progress into by_solver so a partial-write captures
                # whatever's been done, even if the process is killed mid-loop.
                by_solver[name] = {
                    "drags": drags,
                    "flow_rates": flow_rates,
                    "initial_drag": drags[0] if drags else None,
                    "final_drag": drags[-1] if drags else None,
                    "drag_reduction_pct": (
                        100.0
                        * (abs(drags[0]) - abs(drags[-1]))
                        / (abs(drags[0]) + 1e-30)
                        if len(drags) >= 2
                        else 0.0
                    ),
                    "n_iters": len(drags),
                    "converged": False,  # finalised after the loop exits
                    "grad_norms": grad_norms,
                    "in_progress": True,
                }
                # Flush every 10 iterations to bound IO without losing more than
                # ~10 iters of work on an interrupt.
                if (i + 1) % 10 == 0:
                    _write_partial()
                if no_improve >= patience:
                    break

            profile_snaps[name] = np.array(profile)
            if profile_history:
                profile_histories[name] = np.asarray(profile_history)
            by_solver[name] = {
                "drags": drags,
                "flow_rates": flow_rates,
                "initial_drag": drags[0] if drags else None,
                "final_drag": drags[-1] if drags else None,
                "drag_reduction_pct": (
                    100.0 * (abs(drags[0]) - abs(drags[-1])) / (abs(drags[0]) + 1e-30)
                    if len(drags) >= 2
                    else 0.0
                ),
                "n_iters": len(drags),
                "converged": (not non_finite_grad) and (no_improve >= patience),
                "grad_norms": grad_norms,
                **({"error": "non-finite gradients"} if non_finite_grad else {}),
            }
            # Final partial flush so a kill between this point and result.json
            # save still preserves the converged-state record.
            _write_partial()
            # Capture the final velocity field (no gradient needed — plain call).
            try:
                _de = phys.get("domain_extent", cfg.domain_extent)
                _inp_final = cfg.make_inputs(
                    name,
                    profile,
                    domain_extent=_de,
                    **{k: v for k, v in phys.items() if k != "domain_extent"},
                )
                _out_final = apply_tesseract(t, _inp_final)
                _vel = _out_final.get("result")
                if _vel is not None:
                    flow_snaps[name] = np.array(_vel)
            except Exception:
                pass  # velocity snapshot is optional; do not abort the run
            _wall_times[name] = time.perf_counter() - _t0

        # drag_opt requires both inflow_profile support and obstacle drag output.
        # exclusion_lookup checks most-specific first:
        #   "recovery/drag_opt/<run_name>" > "drag_opt/<run_name>" >
        #   "optimization/drag_opt" > "drag_opt" > "optimization"
        # Pass the run-specific sub-key so per-run exclusions (e.g.
        # "recovery/drag_opt/re20") are honoured at solver-selection time.
        _drag_exp = f"drag_opt/{run_name}" if run_name else "drag_opt"
        drag_opt_solvers = _diff_solvers(cfg, "optimization", _drag_exp)
        run_with_gpu_pool(
            drag_opt_solvers,
            tags,
            _drag_opt_work,
            gpu_ids=gpu_ids,
        )

        _dbg = "_debug" if overrides.get("debug") else ""
        if ic_subdir:
            # Multi-run layout: drag_opt[_debug]/<ic_subdir>/
            _parent = experiment_dir(
                results_dir(), cfg.name, _SUITE, "drag_opt", suffix=_dbg
            )
            out_dir = _parent / ic_subdir
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = experiment_dir(
                results_dir(), cfg.name, _SUITE, "drag_opt", suffix=_dbg
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
                "run_drag_opt: by_solver is empty (all solvers excluded or skipped) — "
                "skipping result.json save to preserve existing data"
            )
            if n_runs > 1:
                all_results[run_name] = result
            else:
                all_results = result
            continue
        save_experiment(
            result, out_dir, cfg=cfg, harness_fn=run_drag_opt, wall_time_s=_wall_times
        )
        # Merge profiles.npz with any prior per-solver entries so a
        # single-solver rerun does not wipe peer solvers' profiles / histories.
        profiles_payload: dict[str, np.ndarray] = {
            "initial": np.array(profile_init),
        }
        profiles_path = out_dir / "profiles.npz"
        if profiles_path.exists():
            try:
                prior = np.load(profiles_path)
                for key in prior.files:
                    if key.startswith("final_") or key.startswith("profile_history_"):
                        profiles_payload[key] = prior[key]
            except Exception:
                pass
        for k, v in profile_snaps.items():
            profiles_payload[f"final_{k}"] = v
        for k, v in profile_histories.items():
            profiles_payload[f"profile_history_{k}"] = v
        np.savez(profiles_path, **profiles_payload)
        if flow_snaps or flow_init_snaps:
            _npz_fields: dict = {}
            # Merge with any prior entries so a single-solver rerun does not
            # wipe peer solvers' fields — mirrors profiles.npz merge logic.
            _ff_path = out_dir / "flow_fields.npz"
            if _ff_path.exists():
                try:
                    _prior_ff = np.load(_ff_path)
                    for _k in _prior_ff.files:
                        _npz_fields[_k] = _prior_ff[_k]
                except Exception:
                    pass
            # Save per-solver initial flow (keys flow_initial_{name})
            for _sn, _v in flow_init_snaps.items():
                _npz_fields[f"flow_initial_{_sn}"] = _v
            # Save one canonical initial-flow array using the first available solver
            if flow_init_snaps and "flow_initial" not in _npz_fields:
                _npz_fields["flow_initial"] = next(iter(flow_init_snaps.values()))
            # Save per-solver final flow (keys flow_final_{name})
            for _sn, _v in flow_snaps.items():
                _npz_fields[f"flow_final_{_sn}"] = _v
            np.savez(_ff_path, **_npz_fields)
        if n_runs > 1:
            all_results[run_name] = result
        else:
            all_results = result

    return all_results


def run_topopt_bfgs(cfg: ProblemConfig, tags: dict[str, str], **overrides) -> dict:
    """L-BFGS variant of run_topopt. Reads inverse_defaults["topopt_bfgs"]."""
    runs = cfg.inverse_defaults.get("topopt_bfgs", [])
    if not runs:
        raise NotImplementedError(
            f"No 'topopt_bfgs' inverse_defaults configured for '{cfg.name}'"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
        seed = ic_cfg.get("seed", 0)
        phys = run.get("physics", {})
        optim_cfg = run.get("optim", {})
        max_iters = optim_cfg.get("max_iters", 100)
        v_frac = phys.get("v_frac")
        compliance_key = phys.get("compliance_key", "compliance")
        penalty_weight = phys.get("penalty_weight", 100.0)
        x_min = phys.get("x_min", 1e-3)
        snap_interval = int(phys.get("snap_interval", 0))
        ic_subdir = ic_name if n_runs > 1 else ""

        if v_frac is None:
            raise NotImplementedError(
                f"'topopt_bfgs' requires physics.v_frac in inverse_defaults "
                f"(not configured for '{cfg.name}')"
            )

        rho_init = cfg.make_ic[ic_name](rho_0=v_frac, seed=seed, **phys)

        by_solver: dict = {}
        rho_snaps: dict = {}
        rho_histories: dict = {}
        _wall_times: dict[str, float] = {}
        gpu_ids = overrides.get("gpu_ids")

        def _topopt_bfgs_work(name: str, t) -> None:
            _t0 = time.perf_counter()
            rho_opt = rho_init
            rho_history: list = []

            def loss_fn_scalar(rho, _t=t):
                inp = cfg.make_inputs(name, rho, **phys)
                compliance = apply_tesseract(_t, inp)[compliance_key]
                vol_penalty = penalty_weight * (jnp.mean(rho) - v_frac) ** 2
                return compliance + vol_penalty

            def _log_iter(i, loss_val, _n=name):
                from mosaic.benchmarks.core.console import console

                console.print(
                    f"  [green]{_n}[/] topopt_bfgs iter {i}/{max_iters} loss={loss_val:.4g}"
                )

            rho_opt, losses, lbfgs_diag = _run_lbfgs(
                loss_fn_scalar,
                rho_opt,
                max_iters=max_iters,
                snap_interval=snap_interval,
                history=rho_history if snap_interval > 0 else None,
                log_fn=_log_iter,
                log_interval=10,
                clip_fn=lambda rho: jnp.clip(rho, x_min, 1.0),
            )

            try:
                inp_final = cfg.make_inputs(name, rho_opt, **phys)
                out_final = apply_tesseract(t, inp_final)
                final_compliance = float(out_final[compliance_key])
                final_vol_frac = float(jnp.mean(rho_opt))
            except Exception:
                final_compliance = losses[-1] if losses else float("nan")
                final_vol_frac = float(jnp.mean(rho_opt))

            rho_snaps[name] = np.array(rho_opt)
            rho_histories[name] = rho_history
            by_solver[name] = {
                "compliances": losses,
                "vol_fracs": [],
                "final_compliance": final_compliance,
                "final_vol_frac": final_vol_frac,
                "n_iters": len(losses),
                "converged": len(losses) < max_iters,
                "grad_norms": (lbfgs_diag or {}).get("grad_norms"),
            }
            _wall_times[name] = time.perf_counter() - _t0

        run_with_gpu_pool(
            _topopt_solvers(cfg), tags, _topopt_bfgs_work, gpu_ids=gpu_ids
        )

        out_dir = experiment_dir(
            results_dir(),
            cfg.name,
            _SUITE,
            f"topopt_bfgs/{ic_subdir}" if ic_subdir else "topopt_bfgs",
            suffix="_debug" if overrides.get("debug") else "",
        )
        solver_names = list(rho_snaps.keys())
        per_solver: dict[str, dict[str, np.ndarray]] = {}
        for sname in solver_names:
            entry: dict[str, np.ndarray] = {"rho_final:": np.asarray(rho_snaps[sname])}
            if rho_histories[sname]:
                entry["rho_history:"] = np.asarray(rho_histories[sname])
            per_solver[sname] = entry
        save_gradient_fields_npz(
            out_dir,
            solver_names,
            per_solver,
            shared_arrays={"rho_init": np.array(rho_init)},
            filename="topopt_fields.npz",
            prefixes=("rho_final", "rho_history"),
        )

        result = {"by_solver": by_solver, "params": run}
        save_experiment(
            result,
            out_dir,
            cfg=cfg,
            harness_fn=run_topopt_bfgs,
            wall_time_s=_wall_times,
        )
        if n_runs > 1:
            all_results[ic_name] = result
        else:
            all_results = result

    return all_results


def run_drag_opt_bfgs(cfg: ProblemConfig, tags: dict[str, str], **overrides) -> dict:
    """L-BFGS variant of run_drag_opt. Reads inverse_defaults["drag_opt_bfgs"]."""
    runs = cfg.inverse_defaults.get("drag_opt_bfgs", [])
    if not runs:
        raise NotImplementedError(
            f"No 'drag_opt_bfgs' inverse_defaults configured for '{cfg.name}'"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
        seed = ic_cfg.get("seed", 0)
        phys = run.get("physics", {})
        optim_cfg = run.get("optim", {})
        max_iters = optim_cfg.get("max_iters", 50)
        flow_penalty_weight = optim_cfg.get("flow_penalty_weight", 50.0)
        snap_interval = int(optim_cfg.get("snap_interval", 0))
        U_mean = float(phys.get("U_mean", 0.5))
        run_name = run.get("name", ic_name)
        ic_subdir = run_name if n_runs > 1 else ""
        gpu_ids = overrides.get("gpu_ids")

        profile_init = jnp.array(
            cfg.make_ic[ic_name](L=cfg.domain_extent, seed=seed, **phys)
        )

        by_solver: dict = {}
        profile_snaps: dict = {}
        profile_histories: dict = {}
        flow_init_snaps: dict = {}
        flow_snaps: dict = {}
        _wall_times: dict[str, float] = {}

        def _drag_opt_bfgs_work(name: str, t) -> None:
            _t0 = time.perf_counter()
            profile = profile_init
            profile_history: list = []

            try:
                _de0 = phys.get("domain_extent", cfg.domain_extent)
                _inp0 = cfg.make_inputs(
                    name,
                    profile_init,
                    domain_extent=_de0,
                    **{k: v for k, v in phys.items() if k != "domain_extent"},
                )
                _out0 = apply_tesseract(t, _inp0)
                _vel0 = _out0.get("result")
                if _vel0 is not None:
                    flow_init_snaps[name] = np.array(_vel0)
            except Exception:
                pass

            def loss_fn(p, _t=t):
                _de = phys.get("domain_extent", cfg.domain_extent)
                inp = cfg.make_inputs(
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
                profile,
                max_iters=max_iters,
                snap_interval=snap_interval,
                history=profile_history if snap_interval > 0 else None,
                log_fn=_log_iter,
                log_interval=10,
                clip_fn=lambda p: jnp.clip(p, 0.0, 3.0 * U_mean),
            )

            try:
                _de = phys.get("domain_extent", cfg.domain_extent)
                _inp_f = cfg.make_inputs(
                    name,
                    profile,
                    domain_extent=_de,
                    **{k: v for k, v in phys.items() if k != "domain_extent"},
                )
                _out_f = apply_tesseract(t, _inp_f)
                final_drag = float(
                    jnp.squeeze(_out_f.get("drag", jnp.array(float("nan"))))
                )
                _vel = _out_f.get("result")
                if _vel is not None:
                    flow_snaps[name] = np.array(_vel)
            except Exception:
                final_drag = losses[-1] if losses else float("nan")

            initial_drag = losses[0] if losses else None

            profile_snaps[name] = np.array(profile)
            if profile_history:
                profile_histories[name] = np.asarray(profile_history)
            by_solver[name] = {
                "drags": losses,
                "flow_rates": [],
                "initial_drag": initial_drag,
                "final_drag": final_drag,
                "drag_reduction_pct": (
                    100.0
                    * (abs(losses[0]) - abs(final_drag))
                    / (abs(losses[0]) + 1e-30)
                    if losses
                    else 0.0
                ),
                "n_iters": len(losses),
                "converged": len(losses) < max_iters,
                "grad_norms": (lbfgs_diag or {}).get("grad_norms"),
            }
            _wall_times[name] = time.perf_counter() - _t0

        _drag_exp = f"drag_opt_bfgs/{run_name}" if run_name else "drag_opt_bfgs"
        drag_opt_solvers = _diff_solvers(cfg, "optimization", _drag_exp)
        run_with_gpu_pool(drag_opt_solvers, tags, _drag_opt_bfgs_work, gpu_ids=gpu_ids)

        _dbg = "_debug" if overrides.get("debug") else ""
        if ic_subdir:
            _parent = experiment_dir(
                results_dir(), cfg.name, _SUITE, "drag_opt_bfgs", suffix=_dbg
            )
            out_dir = _parent / ic_subdir
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = experiment_dir(
                results_dir(), cfg.name, _SUITE, "drag_opt_bfgs", suffix=_dbg
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
                "run_drag_opt_bfgs: by_solver is empty (all solvers excluded or skipped) — "
                "skipping result.json save to preserve existing data"
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
            harness_fn=run_drag_opt_bfgs,
            wall_time_s=_wall_times,
        )
        profiles_payload: dict[str, np.ndarray] = {
            "initial": np.array(profile_init),
        }
        profiles_path = out_dir / "profiles.npz"
        if profiles_path.exists():
            try:
                prior = np.load(profiles_path)
                for key in prior.files:
                    if key.startswith("final_") or key.startswith("profile_history_"):
                        profiles_payload[key] = prior[key]
            except Exception:
                pass
        for k, v in profile_snaps.items():
            profiles_payload[f"final_{k}"] = v
        for k, v in profile_histories.items():
            profiles_payload[f"profile_history_{k}"] = v
        np.savez(profiles_path, **profiles_payload)
        if flow_snaps or flow_init_snaps:
            _npz_fields: dict = {}
            _ff_path = out_dir / "flow_fields.npz"
            if _ff_path.exists():
                try:
                    _prior_ff = np.load(_ff_path)
                    for _k in _prior_ff.files:
                        _npz_fields[_k] = _prior_ff[_k]
                except Exception:
                    pass
            for _sn, _v in flow_init_snaps.items():
                _npz_fields[f"flow_initial_{_sn}"] = _v
            if flow_init_snaps and "flow_initial" not in _npz_fields:
                _npz_fields["flow_initial"] = next(iter(flow_init_snaps.values()))
            for _sn, _v in flow_snaps.items():
                _npz_fields[f"flow_final_{_sn}"] = _v
            np.savez(_ff_path, **_npz_fields)
        if n_runs > 1:
            all_results[run_name] = result
        else:
            all_results = result

    return all_results


# ── Conductivity recovery (thermal-mesh) ─────────────────────────────────────


def run_conductivity_recovery(
    cfg: ProblemConfig,
    tags: dict[str, str],
    _exp_key: str = "conductivity_recovery",
    use_lbfgs: bool = False,
    **overrides,
) -> dict:
    """Conductivity-field recovery: recover rho from temperature observations.

    Optimises the SIMP density field (rho, clipped to [x_min, 1]) to minimise
    identification_error = ||T(rho) - T_target||^2 using Adam.  The target
    temperature is produced by forward-solving with a two-Gaussian ground-truth
    conductivity and zero volumetric source (Neumann BC only).

    Expects ``inverse_defaults[_exp_key]`` runs with:
        ic:     {name, seed}         — IC generator for initial rho (e.g. "uniform")
        physics: {nx, ny, nz, Lx, Ly, Lz, rho_0, Q_total, compliance_key,
                  penalty_weight, x_min, snap_interval, target_rho_from_two_gaussians}
        optim:  {lr, max_iters, patience}

    Returns:
        {"by_solver": {solver: {"errors", "final_error", "n_iters", "converged"}},
         "params": run}
        or {ic_name: <above>} when multiple runs are configured.
    """
    runs = cfg.inverse_defaults.get(_exp_key, [])
    if not runs:
        raise NotImplementedError(
            f"No '{_exp_key}' inverse_defaults configured for '{cfg.name}'"
        )
    n_runs = len(extract_runs(runs))
    all_results: dict = {}

    for run in iter_runs(runs, overrides):
        ic_cfg = run.get("ic", {})
        ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
        seed = ic_cfg.get("seed", 0)
        phys = run.get("physics", {})
        optim_cfg = run.get("optim", {})
        lr = optim_cfg.get("lr", 1e-2)
        max_iters = optim_cfg.get("max_iters", 500)
        patience = optim_cfg.get("patience", 50)
        compliance_key = phys.get("compliance_key", "identification_error")
        penalty_weight = float(phys.get("penalty_weight", 0.0))
        x_min = float(phys.get("x_min", 1e-3))
        snap_interval = int(phys.get("snap_interval", 0))
        ic_subdir = ic_name if n_runs > 1 else ""
        gpu_ids = overrides.get("gpu_ids")

        rho_init = jnp.array(cfg.make_ic[ic_name](seed=seed, **phys))

        # Ground-truth conductivity field: prefer "two_gaussians_rho", then "two_gaussians".
        truth_ic_name: str | None = None
        for candidate in ("two_gaussians_rho", "two_gaussians"):
            if candidate in cfg.make_ic:
                truth_ic_name = candidate
                break
        if truth_ic_name is not None:
            rho_truth = np.asarray(
                cfg.make_ic[truth_ic_name](seed=seed, **phys), dtype=np.float32
            )
        else:
            rho_truth = None

        by_solver: dict = {}
        rho_snaps: dict = {}
        rho_histories: dict = {}
        _wall_times: dict[str, float] = {}

        candidate_solvers = _diff_solvers(cfg, "optimization", _exp_key)

        def _conductivity_recovery_work(name: str, t) -> None:
            _t0 = time.perf_counter()
            rho_opt = rho_init
            errors = []
            rho_history: list = []

            _loss_phys = {k: v for k, v in phys.items() if k != "rho_0"}

            def loss_fn(rho, _t=t):
                inp = cfg.make_inputs(name, rho, **_loss_phys)
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

            def loss_fn_scalar(rho, _t=t):
                inp = cfg.make_inputs(name, rho, **_loss_phys)
                out = apply_tesseract(_t, inp)
                err = out.get(compliance_key)
                if err is None:
                    raise RuntimeError(
                        f"Solver '{name}' did not return '{compliance_key}'"
                    )
                penalty = (
                    penalty_weight * jnp.mean(rho**2) if penalty_weight > 0 else 0.0
                )
                return jnp.squeeze(err) + penalty

            if use_lbfgs:
                rho_opt, errors, _lbfgs_diag = _run_lbfgs(
                    loss_fn_scalar,
                    rho_opt,
                    max_iters=max_iters,
                    snap_interval=snap_interval,
                    history=rho_history if snap_interval > 0 else None,
                    log_interval=10,
                    clip_fn=lambda r: jnp.clip(r, x_min, 1.0),
                )
                no_improve = patience  # treat as converged for result reporting
                grad_norms_adam: list[float] = []
            else:
                _lbfgs_diag = None
                optimizer = optax.adam(lr)
                opt_state = optimizer.init(rho_opt)
                best_err, no_improve = jnp.inf, 0
                grad_norms_adam: list[float] = []

                for i in range(max_iters):
                    try:
                        (_, err_val), g = jax.value_and_grad(loss_fn, has_aux=True)(
                            rho_opt
                        )
                        grad_norms_adam.append(float(jnp.linalg.norm(g.ravel())))
                        updates, opt_state = optimizer.update(g, opt_state)
                        # Project rho back into [x_min, 1] after each step.
                        rho_opt = jnp.clip(
                            optax.apply_updates(rho_opt, updates), x_min, 1.0
                        )
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

                        print_warn(
                            f"{name} conductivity_recovery iter {i} failed: {exc}"
                        )
                        break

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
                "converged": len(errors) < max_iters
                if use_lbfgs
                else no_improve >= patience,
                "grad_norms": (_lbfgs_diag or {}).get("grad_norms")
                if use_lbfgs
                else grad_norms_adam,
            }
            _wall_times[name] = time.perf_counter() - _t0

        run_with_gpu_pool(
            candidate_solvers, tags, _conductivity_recovery_work, gpu_ids=gpu_ids
        )

        _dbg = "_debug" if overrides.get("debug") else ""
        out_dir = experiment_dir(
            results_dir(),
            cfg.name,
            _SUITE,
            f"{_exp_key}/{ic_subdir}" if ic_subdir else _exp_key,
            suffix=_dbg,
        )

        solver_names = list(rho_snaps.keys())
        npz_payload: dict[str, np.ndarray] = {"rho_init": np.array(rho_init)}
        if rho_truth is not None:
            npz_payload["rho_truth"] = rho_truth
        existing_path = out_dir / "rho_fields.npz"
        if existing_path.exists():
            try:
                prior = np.load(existing_path)
                for key in prior.files:
                    if key.startswith("rho_final_") or key.startswith("rho_history_"):
                        npz_payload[key] = prior[key]
                    elif key == "rho_truth" and "rho_truth" not in npz_payload:
                        npz_payload[key] = prior[key]
            except Exception:
                pass
        for sname in solver_names:
            npz_payload[f"rho_final_{sname}"] = rho_snaps[sname]
            if rho_histories[sname]:
                npz_payload[f"rho_history_{sname}"] = np.asarray(rho_histories[sname])

        np.savez(existing_path, **npz_payload)

        result = {"by_solver": by_solver, "params": run}
        save_experiment(
            result,
            out_dir,
            cfg=cfg,
            harness_fn=run_conductivity_recovery,
            wall_time_s=_wall_times,
        )
        if n_runs > 1:
            all_results[ic_name] = result
        else:
            all_results = result

    return all_results


def run_conductivity_recovery_bfgs(
    cfg: ProblemConfig, tags: dict[str, str], **overrides
) -> dict:
    """L-BFGS variant of run_conductivity_recovery."""
    return run_conductivity_recovery(
        cfg, tags, _exp_key="conductivity_recovery_bfgs", use_lbfgs=True, **overrides
    )


# ── run_all + __main__ ────────────────────────────────────────────────────────

_EXPERIMENTS = {
    "recovery_constant_ic": run_recovery_constant_ic,
    "recovery_constant_ic_bfgs": run_recovery_constant_ic_bfgs,
    "recovery_constant_ic_bfgs_proj": run_recovery_constant_ic_bfgs_proj,
    "topopt": run_topopt,
    "topopt_bfgs": run_topopt_bfgs,
    "conductivity_recovery": run_conductivity_recovery,
    "conductivity_recovery_bfgs": run_conductivity_recovery_bfgs,
    "drag_opt": run_drag_opt,
    "drag_opt_bfgs": run_drag_opt_bfgs,
    "drag_opt/re20": run_drag_opt,
    "drag_opt_bfgs/re20": run_drag_opt_bfgs,
}


def _plot_fns() -> dict:
    from mosaic.benchmarks.plots.optimization import (
        plot_conductivity_recovery,
        plot_drag_opt,
        plot_recovery,
        plot_topopt,
    )

    return {
        "recovery_constant_ic": lambda cfg, **kw: plot_recovery(
            cfg, exp_key="recovery_constant_ic", **kw
        ),
        "recovery_constant_ic_bfgs": lambda cfg, **kw: plot_recovery(
            cfg, exp_key="recovery_constant_ic_bfgs", **kw
        ),
        "recovery_constant_ic_bfgs_proj": lambda cfg, **kw: plot_recovery(
            cfg, exp_key="recovery_constant_ic_bfgs_proj", **kw
        ),
        "topopt": plot_topopt,
        "topopt_bfgs": lambda cfg, **kw: plot_topopt(cfg, exp_key="topopt_bfgs", **kw),
        "conductivity_recovery": plot_conductivity_recovery,
        "conductivity_recovery_bfgs": lambda cfg, **kw: plot_conductivity_recovery(
            cfg, exp_key="conductivity_recovery_bfgs", **kw
        ),
        "drag_opt": plot_drag_opt,
        "drag_opt_bfgs": lambda cfg, **kw: plot_drag_opt(
            cfg, exp_key="drag_opt_bfgs", **kw
        ),
    }


def run_all(
    cfg: ProblemConfig,
    tags: dict[str, str],
    experiments: list[str] | None = None,
    plots: bool = True,
) -> dict[str, dict]:
    """Run optimization experiments and optionally generate plots."""
    from mosaic.benchmarks.core.runner import run_suite

    return run_suite(
        cfg,
        tags,
        _EXPERIMENTS,
        to_run=experiments,
        plots=plots,
        plot_fns=_plot_fns() if plots else None,
        suite_name=_SUITE,
    )
