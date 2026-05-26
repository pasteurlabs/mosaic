"""Gradient evaluation suite: FD verification, parameter sweep, Jacobian SVD.

Only runs solvers where ``SolverSpec.differentiable`` is True.

Every harness in this module is a decorated *kernel*: a tiny science function
that handles one (solver, sweep-point). The framework
(:func:`mosaic.benchmarks.core.experiment.run_experiment`) owns IC
construction, solver iteration, sweep walking, memory polling, failure
classification, and result persistence. Configs reference the kernel
directly — no harness wrapper:

    problem.add_experiment("gradient/horizon_sweep", param_sweep, ...)

:func:`jacobian_svd` is also a kernel but pairs with a custom
:func:`_jacobian_svd_aggregate` (passed via the ``@kernel`` decorator)
that does the cross-solver SVD + optional loss-landscape pass after the
per-solver Jacobian phase.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.console import console
from mosaic.benchmarks.core.experiment import (
    KernelContext,
    kernel,
    random_direction,
)
from mosaic.benchmarks.core.io import save_field_snapshots_npz, try_load_npz
from mosaic.benchmarks.core.runner import run_with_gpu_pool

# JAX-traced closures capture this reference at trace time; using the
# tracer-aware wrapper ensures primitive binding sees the active trace.
from mosaic.benchmarks.core.tracer_apply import apply_tesseract

# ── Science primitives ───────────────────────────────────────────────────────


def _vjp_grad(t, inputs: dict, output_key: str, ic_key: str) -> jax.Array:
    """Gradient of sum(output**2) w.r.t. inputs[ic_key] via jax.grad.

    Uses kinetic-energy-like loss rather than sum(output) so that the gradient
    is non-trivial for divergence-free / momentum-conserving solvers.
    """

    def f(ic):
        out = apply_tesseract(t, {**inputs, ic_key: ic})[output_key]
        return jnp.sum(out**2)

    return jax.grad(f)(inputs[ic_key])


def _fd_cosine(fd_arr: np.ndarray, vjp_arr: np.ndarray) -> float:
    """Subspace cosine: angle between fd and vjp directional-derivative vectors
    across all random directions.  1 = perfectly aligned, 0 = orthogonal."""
    return float(
        np.dot(fd_arr, vjp_arr)
        / (np.linalg.norm(fd_arr) * np.linalg.norm(vjp_arr) + 1e-30)
    )


_DEFAULT_EPS = [1e0, 1e-1, 1e-2, 1e-3]
_FD_CHECK_DEFAULT_EPS = [5e0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]


def _fd_vjp_arrays(
    t,
    ctx: KernelContext,
    eps_values: list,
    n_dirs: int,
) -> tuple[jax.Array, float, jax.Array, np.ndarray, dict[float, np.ndarray]]:
    """Compute the raw FD/VJP arrays underpinning every FD-style kernel.

    Returns ``(base_ic, ic_scale, g, vjp_arr, fd_per_eps)``:

      * ``base_ic`` — IC array after make_inputs (some problems override the
        passed IC via a parameter like rho_0; we must perturb around this
        post-make_inputs array, not the raw caller-side IC).
      * ``ic_scale`` — RMS of base_ic, used to scale ε relative to the IC
        magnitude so that ``eps_values`` are problem-agnostic.
      * ``g`` — VJP gradient of ``sum(output**2)`` w.r.t. ``base_ic``.
      * ``vjp_arr`` — per-direction VJP directional derivative
        ``<g, v_k>`` for each of ``n_dirs`` random directions.
      * ``fd_per_eps`` — per-eps central-FD directional derivatives, same
        layout as ``vjp_arr``.
    """
    base_inputs = ctx.make_inputs(ctx.name, ctx.ic, **ctx.phys)
    base_ic = jnp.array(base_inputs[ctx.ic_key])
    ic_scale = float(jnp.sqrt(jnp.mean(base_ic**2) + 1e-30))
    keys = jax.random.split(jax.random.PRNGKey(ctx.seed), n_dirs)
    dirs = [random_direction(base_ic.shape, k) for k in keys]

    g = _vjp_grad(t, base_inputs, ctx.output_key, ctx.ic_key)
    vjp_arr = np.array([float(jnp.dot(g.ravel(), v.ravel())) for v in dirs])

    fd_per_eps: dict[float, np.ndarray] = {}
    for eps in eps_values:
        abs_eps = eps * ic_scale
        fd_per_eps[eps] = np.array(
            [
                float(
                    jnp.sum(
                        apply_tesseract(
                            t, {**base_inputs, ctx.ic_key: base_ic + abs_eps * v}
                        )[ctx.output_key]
                        ** 2
                        - apply_tesseract(
                            t, {**base_inputs, ctx.ic_key: base_ic - abs_eps * v}
                        )[ctx.output_key]
                        ** 2
                    )
                    / (2 * abs_eps)
                )
                for v in dirs
            ]
        )
    return base_ic, ic_scale, g, vjp_arr, fd_per_eps


# ── Kernels ──────────────────────────────────────────────────────────────────


@kernel(sweep_mode="none", catch_label="VJP failed")
def fd_check(t, ctx: KernelContext) -> dict:
    """One FD-verification point: VJP grad + central-FD over ``eps_values``.

    Records per-direction rel_error (not aggregated) so that downstream
    analysis can recompute mean/std/cosine without rerunning the benchmark.
    """
    fd_cfg = ctx.run.get("fd", {})
    eps_values = fd_cfg.get("eps_values", _FD_CHECK_DEFAULT_EPS)
    n_dirs = fd_cfg.get("n_dirs", 20)

    base_ic, ic_scale, g, vjp_arr, fd_per_eps = _fd_vjp_arrays(
        t, ctx, eps_values, n_dirs
    )
    eps_sweep: dict = {}
    for eps, fd_arr in fd_per_eps.items():
        denom = np.maximum(np.maximum(np.abs(fd_arr), np.abs(vjp_arr)), 1e-30)
        eps_sweep[eps] = {
            "rel_error": (np.abs(fd_arr - vjp_arr) / denom).tolist(),
            "cosine": _fd_cosine(fd_arr, vjp_arr),
            "fd_arr": fd_arr.tolist(),
        }
    return {
        "metrics": {
            "ic_scale": ic_scale,
            "vjp_arr": vjp_arr.tolist(),
            "eps_sweep": eps_sweep,
        },
        "snapshot": np.asarray(g),
        # Override the framework's raw-IC default with the post-make_inputs
        # IC so plots see the same array the kernel perturbed around (matters
        # for structural_mesh where make_inputs substitutes rho_0).
        "shared": {"ic": np.asarray(base_ic)},
    }


@kernel(sweep_mode="default", horizons_shared=True, catch_label="VJP failed")
def param_sweep(t, ctx: KernelContext) -> dict:
    """One sweep point for a parameter sweep: grad_norm + per-eps mean/std/cosine.

    ``sweep.key`` / ``sweep.values`` on the run dict pick the parameter
    (``nu``, ``kT``, ``sigma8``, ``N``, ``steps``, …) so the same kernel
    handles every variant — registering it under ``gradient/horizon_sweep``
    versus ``gradient/param_sweep`` differs only by the experiment key.
    """
    fd_cfg = ctx.run.get("fd", {})
    eps_values = fd_cfg.get("eps_values", _DEFAULT_EPS)
    n_dirs = fd_cfg.get("n_dirs", 15)

    _, ic_scale, g, vjp_arr, fd_per_eps = _fd_vjp_arrays(t, ctx, eps_values, n_dirs)
    eps_sweep: dict = {}
    for eps, fd_arr in fd_per_eps.items():
        denom = np.maximum(np.maximum(np.abs(fd_arr), np.abs(vjp_arr)), 1e-30)
        eps_sweep[eps] = {
            "rel_error_mean": float(np.mean(np.abs(fd_arr - vjp_arr) / denom)),
            "rel_error_std": float(np.std(np.abs(fd_arr - vjp_arr) / denom)),
            "cosine_mean": _fd_cosine(fd_arr, vjp_arr),
        }
    return {
        "metrics": {
            "grad_norm": float(jnp.linalg.norm(g)),
            "ic_scale": ic_scale,
            "eps_sweep": eps_sweep,
        },
        "snapshot": np.asarray(g),
    }


@kernel(
    sweep_mode="limits",
    warmup=True,
    horizons_shared=True,
    catch_label="VJP failed",
)
def horizon_sweep_limits(t, ctx: KernelContext) -> dict:
    """One VJP attempt for the rollout-limit sweep — no FD check.

    Raises on non-finite gradient so the framework's per-step try/except
    catches it as a regular failure (classified as ``nan`` downstream).

    Result layout::

        by_solver[name][steps_val] = {"status": "ok",
                                      "grad_norm": float,
                                      "wall_time_s": float,
                                      "vram_peak_mib": float | None,
                                      "ram_peak_mib": float | None}
                                   | {"status": "failed",
                                      "failure_type": "OOM"|"timeout"|"container_died"|"nan"|"error",
                                      "error": str,
                                      "wall_time_s": float, ...}
                                   | {"status": "skipped", "reason": str}

    Intended to be run with one GPU per solver so OOM reflects a single GPU
    budget::

        mosaic gradient ns-3d-grid --experiments horizon_sweep_limits --gpu-ids 0 1 2 3
    """
    inputs = ctx.make_inputs(ctx.name, ctx.ic, **ctx.phys)
    g = _vjp_grad(t, inputs, ctx.output_key, ctx.ic_key)
    if not jnp.all(jnp.isfinite(g)):
        raise ValueError("VJP returned non-finite gradient (NaN/Inf)")
    g_np = np.array(g).ravel()
    return {
        "metrics": {
            "grad_norm": float(jnp.linalg.norm(g)),
            "grad_mean": float(g_np.mean()),
            "grad_std": float(g_np.std()),
            "grad_min": float(g_np.min()),
            "grad_max": float(g_np.max()),
        },
        "snapshot": np.asarray(g),
    }


# ── Jacobian SVD ─────────────────────────────────────────────────────────────


def _jacobian_svd_aggregate(  # noqa: PLR0913 — aggregate signature mirrors the kernel's extras
    by_solver,
    *,
    run,
    cfg,
    tags,
    out_dir,
    selected,
    gpu_ids,
    snapshots,
    shared_extras,
    ic,
    sweep_values,
    sweep_key,
    snapshot_filename,
    snapshot_prefixes,
    horizons_shared,
) -> dict:
    """Cross-solver SVD + optional loss-landscape pass.

    Inputs (via ``snapshots``): ``{name: {"grad:": grad_arr, "jac:": J_mat}}``
    from the per-solver kernel pass. ``by_solver`` is unused — all metrics
    are derived from the snapshots and the SVD.
    """
    del by_solver  # unused — metrics are computed here from snapshots
    jacobian_cfg = run.get("jacobian", {})
    n_alphas = jacobian_cfg.get("n_alphas", 41)
    alpha_range = jacobian_cfg.get("alpha_range", 0.3)
    k_svd = jacobian_cfg.get("k_svd", None)
    phys = run.get("physics", {})
    ic_cfg = run.get("ic", {})
    ic_name = ic_cfg.get("name")
    seed = ic_cfg.get("seed", 0)
    output_key = run.get("output_key", cfg.output_key)
    ic_key = run.get("ic_key", cfg.ic_key)

    # ── Unpack per-solver kernel output ───────────────────────────────────
    jacobians: dict = {}
    grad_snaps: dict = {}
    for name, suf_map in snapshots.items():
        for k, arr in suf_map.items():
            if k.startswith("jac:"):
                jacobians[name] = np.asarray(arr)
            elif k.startswith("grad:"):
                grad_snaps[name] = np.asarray(arr)

    # ── Merge with existing Jacobians from NPZ (partial-run resumption) ──
    # If a prior partial run saved Jacobians, load them so aggregate stats
    # (combined SVD, cross_cosine) use the full solver set rather than just
    # the current subset. Per-solver entries computed this run take precedence.
    _npz = try_load_npz(out_dir / snapshot_filename)
    _old_names = [str(n) for n in _npz.get("solver_names", np.array([]))]
    for _j, _old_name in enumerate(_old_names):
        _jac_key = f"jac_{_j}"
        _grad_key = f"grad_{_j}"
        if _old_name not in jacobians and _jac_key in _npz:
            _J = _npz[_jac_key]
            jacobians[_old_name] = _J
            if _grad_key in _npz:
                grad_snaps[_old_name] = _npz[_grad_key]
            elif _J.ndim == 2:
                # Frobenius row-norm as a stand-in when only the matrix is
                # cached — sign is not guaranteed but the value is only
                # consumed by grad_norms reporting, not by the SVD.
                grad_snaps[_old_name] = np.linalg.norm(_J, axis=0)
            console.print(
                f"  [dim]jacobian_svd: merged existing Jacobian for {_old_name} from NPZ[/]"
            )

    if not jacobians:
        raise RuntimeError("No differentiable solvers returned Jacobians")

    solver_names = list(jacobians.keys())
    G_stack = np.vstack([jacobians[n] for n in solver_names])

    # ── SVD ───────────────────────────────────────────────────────────────
    _U, S, Vt = np.linalg.svd(G_stack, full_matrices=False)
    k_report = k_svd if k_svd is not None else len(S)
    S = S[:k_report]
    Vt = Vt[:k_report]

    S_norm = (S / (S[0] + 1e-30)).tolist()
    cond = float(S[0] / (S[-1] + 1e-30))
    eff_rank = float(S.sum() ** 2 / ((S**2).sum() + 1e-30))
    expl_var = (np.cumsum(S**2) / (float((S**2).sum()) + 1e-30)).tolist()

    # Per-solver singular value spectra (SVD of each solver's Jacobian separately).
    # Reveals spectral structure per solver family: projection methods tend to
    # have a steep singular-value drop (low effective rank) while LBM Jacobians
    # have a flatter spectrum (higher effective rank).
    per_solver_spectra: dict[str, list[float]] = {}
    per_solver_cond: dict[str, float] = {}
    per_solver_eff_rank: dict[str, float] = {}
    per_solver_grad_norm: dict[str, float] = {}
    for name in solver_names:
        _, Si, _ = np.linalg.svd(jacobians[name], full_matrices=False)
        k_i = k_svd if k_svd is not None else len(Si)
        Si = Si[:k_i]
        per_solver_spectra[name] = (Si / (Si[0] + 1e-30)).tolist()
        per_solver_cond[name] = float(Si[0] / (Si[-1] + 1e-30))
        per_solver_eff_rank[name] = float(Si.sum() ** 2 / ((Si**2).sum() + 1e-30))
        per_solver_grad_norm[name] = float(Si[0])

    # Cross-cosine: Frobenius inner product between per-solver Jacobians (normalised)
    J_flat = {n: jacobians[n].ravel() for n in solver_names}
    J_norms = {n: float(np.linalg.norm(v) + 1e-30) for n, v in J_flat.items()}
    cross_cos = [
        [
            float(np.dot(J_flat[a], J_flat[b]) / (J_norms[a] * J_norms[b]))
            for b in solver_names
        ]
        for a in solver_names
    ]
    grad_norms = {n: float(np.linalg.norm(grad_snaps[n])) for n in solver_names}

    # Top singular direction — shaped like the IC.
    # The grad-snapshot has the IC shape; use any solver's to reconstruct.
    base_ic_arr = np.asarray(grad_snaps[solver_names[0]])
    ic_scale = float(np.sqrt(np.mean(base_ic_arr**2) + 1e-30))
    d_top = Vt[0].reshape(base_ic_arr.shape).astype(np.float32)

    # ── Pass 2: loss landscape along d_top (skip when n_alphas == 0) ──────
    alphas = (
        np.linspace(-alpha_range, alpha_range, n_alphas).tolist()
        if n_alphas > 0
        else []
    )
    landscape_by_solver: dict = {}

    if alphas:
        # Reconstruct the per-solver IC from make_ic — the original IC array
        # is shared across solvers (make_inputs may mutate it, but the
        # landscape pass needs the same base each solver was perturbed
        # around).
        base_ic_jax = jnp.array(ic)
        d_top_jax = jnp.array(d_top)
        domain_extent = cfg.domain_extent

        def _landscape_work(name: str, t) -> None:
            color = cfg.solver(name).color
            base_inputs = cfg.make_inputs(
                name, base_ic_jax, domain_extent=domain_extent, **phys
            )
            base_ic_solver = jnp.array(base_inputs[ic_key])
            losses = [
                float(
                    jnp.sum(
                        apply_tesseract(
                            t,
                            {
                                **base_inputs,
                                ic_key: base_ic_solver
                                + float(a) * ic_scale * d_top_jax,
                            },
                        )[output_key]
                        ** 2
                    )
                )
                for a in alphas
            ]
            landscape_by_solver[name] = losses
            console.print(f"  [{color}]{name}[/] landscape done")

        # Only run landscape for solvers that produced a Jacobian.
        landscape_solvers = [n for n in selected if n in jacobians]
        run_with_gpu_pool(landscape_solvers, tags, _landscape_work, gpu_ids=gpu_ids)

    # ── Save NPZ ──────────────────────────────────────────────────────────
    # Per-solver payload: grad_{j} (1-D gradient) and jac_{j} (full Jacobian
    # matrix). The ``jac`` prefix enables future partial runs to reload
    # existing Jacobians and recompute aggregate statistics without
    # re-running all solvers (see merge block above).
    _per_solver_npz: dict[str, dict[str, np.ndarray]] = {
        name: {
            "grad:": np.asarray(grad_snaps[name]),
            "jac:": np.asarray(jacobians[name]),
        }
        for name in solver_names
    }
    save_field_snapshots_npz(
        out_dir,
        solver_names,
        _per_solver_npz,
        shared_arrays={
            "singular_values": np.asarray(S),
            "singular_vectors": np.asarray(Vt),
            "ic": np.array(ic),
            **shared_extras,
        },
        filename=snapshot_filename,
        prefixes=snapshot_prefixes,
    )

    del ic_name, seed  # unused; accepted for signature parity
    del horizons_shared, sweep_values, sweep_key  # not applicable to non-sweep
    return {
        "solver_names": solver_names,
        "singular_values": S_norm,
        "singular_values_raw": S.tolist(),
        "condition_number": cond,
        "effective_rank": eff_rank,
        "explained_variance": expl_var,
        "per_solver_spectra": per_solver_spectra,
        "per_solver_cond": per_solver_cond,
        "per_solver_eff_rank": per_solver_eff_rank,
        "per_solver_grad_norm": per_solver_grad_norm,
        "cross_cosine": cross_cos,
        "grad_norms": grad_norms,
        "landscape": {"alphas": alphas, "by_solver": landscape_by_solver},
        "params": run,
    }


@kernel(
    sweep_mode="none",
    aggregate_fn=_jacobian_svd_aggregate,
    catch_label="VJP failed",
    snapshot_filename="jacobian_svd.npz",
    snapshot_prefixes=("grad", "jac"),
)
def jacobian_svd(t, ctx: KernelContext) -> dict:
    """Full per-solver Jacobian via sequential VJP — one column per output element.

    ``jax.jacrev`` would use ``vmap`` which tesseract doesn't support, so we
    loop. Tractable only for small N (D_out scales with the output grid).
    The cross-solver SVD, per-solver spectra, cross-cosine, and (optional)
    loss-landscape pass live in :func:`_jacobian_svd_aggregate`.
    """
    color = ctx.cfg.solver(ctx.name).color
    base_inputs = ctx.make_inputs(ctx.name, ctx.ic, **ctx.phys)
    base_ic = jnp.array(base_inputs[ctx.ic_key])

    def fwd(ic_arr):
        return apply_tesseract(t, {**base_inputs, ctx.ic_key: ic_arr})[ctx.output_key]

    out, vjp_fn = jax.vjp(fwd, base_ic)
    out_arr = np.array(out)
    D_out = int(out_arr.size)
    log_every = max(1, D_out // 8)
    console.print(f"  [{color}]{ctx.name}[/] Jacobian {D_out} rows starting")
    J_rows = []
    for i in range(D_out):
        e_i = jnp.zeros(D_out).at[i].set(1.0)
        (row,) = vjp_fn(e_i.reshape(out_arr.shape))
        J_rows.append(np.array(row).ravel())
        if (i + 1) % log_every == 0:
            console.print(
                f"  [{color}]{ctx.name}[/] Jacobian {i + 1}/{D_out} rows done"
            )
    J_mat = np.stack(J_rows)

    # Gradient of sum(output²) = J^T @ (2*output): for field plots.
    grad = (J_mat.T @ (2.0 * out_arr.ravel())).reshape(base_ic.shape)
    return {
        "metrics": {},  # populated entirely by the aggregate pass
        "snapshots": {"grad": np.asarray(grad), "jac": J_mat},
    }
