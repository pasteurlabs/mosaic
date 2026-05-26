# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic optimizer primitives — :func:`_run_optim` (Adam) and :func:`_run_lbfgs`.

Problem-specific optimisation harnesses (drag-opt, IC recovery, SIMP
topology optimisation, conductivity recovery) call these directly. The
``has_aux=True`` mode lets per-iter auxiliary values (compliance vs. loss,
volume fraction, error vs. target, …) flow through the same primitive so
each problem only writes its loss function — not its own loop.

Both primitives share the same call shape:

    final_x, losses, diag = _run_optim(loss_fn, init_x, lr, max_iters, patience, ...)
    final_x, losses, diag = _run_lbfgs(loss_fn, init_x, lr, max_iters, patience, ...)

Trailing ``lr``/``patience`` are ignored by L-BFGS so the two are
signature-compatible at call sites.

When ``has_aux=True``, ``loss_fn(x)`` must return ``(scalar_loss, aux_dict)``
where ``aux_dict`` maps a label to a JAX scalar. The primitive records each
label's per-iter trace into ``aux_history`` (the caller-supplied dict, keys
materialised on first sight) so callers can recover compliance/vol_frac/
error-vs-target trajectories without writing their own loop.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax


def _append_aux(aux_history: dict | None, aux: Any) -> None:
    """Append every (label, float) in ``aux`` to ``aux_history[label]``.

    ``aux`` must be a dict of JAX scalars (the convention for
    ``has_aux=True``). Missing labels are created on first sight so
    callers don't need to pre-populate the dict.
    """
    if aux_history is None or not isinstance(aux, dict):
        return
    for label, val in aux.items():
        aux_history.setdefault(label, []).append(float(val))


def _run_optim(
    loss_fn: Any,
    init_x: Any,
    lr: float,
    max_iters: int,
    patience: int,
    *,
    has_aux: bool = False,
    aux_history: dict | None = None,
    clip_fn: Any = None,
    snap_interval: int = 0,
    history: list | None = None,
    snap_error_fn: Any = None,
    error_history: list | None = None,
    log_fn: Any = None,
    log_interval: int = 20,
    record_diagnostics: bool = False,
    div_fn: Any = None,
    grad_proj_fn: Any = None,
) -> tuple:
    """Adam with patience-based early stopping.

    When ``has_aux=True``, ``loss_fn(x) → (scalar_loss, aux_dict)`` and each
    aux label's per-iter trace lands in ``aux_history[label]`` (caller
    provides the dict, primitive populates it).

    ``clip_fn`` (e.g. ``lambda x: jnp.clip(x, x_min, 1.0)``) is applied to
    ``x`` after each Adam update so the iterate stays in the feasible set
    — matches the projection contract used by :func:`_run_lbfgs`.

    Returns ``(final_x, losses, diag)`` where ``losses`` is the per-iter
    scalar-loss list and ``diag`` is a dict with optional diagnostic
    time-series (``None`` when ``record_diagnostics=False``):

      * ``grad_norms``: per-iter ``‖∇L‖₂``
      * ``grad_divs``:  per-iter ``max|∇·g|`` (only when ``div_fn`` set)
      * ``ic_divs``:    per-iter ``max|∇·u|`` (only when ``div_fn`` set)
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(init_x)
    x = init_x
    losses, best, no_improve = [], jnp.inf, 0
    diag: dict = {"grad_norms": []}
    if record_diagnostics:
        diag["grad_divs"] = [] if div_fn else None
        diag["ic_divs"] = [] if div_fn else None
    value_and_grad = jax.value_and_grad(loss_fn, has_aux=has_aux)
    for i in range(max_iters):
        if has_aux:
            (loss, aux), g = value_and_grad(x)
            _append_aux(aux_history, aux)
        else:
            loss, g = value_and_grad(x)
        diag["grad_norms"].append(float(jnp.linalg.norm(g.ravel())))
        if record_diagnostics and div_fn is not None:
            diag["grad_divs"].append(div_fn(np.asarray(g)))
            diag["ic_divs"].append(div_fn(np.asarray(x)))
        updates, opt_state = optimizer.update(g, opt_state)
        x = optax.apply_updates(x, updates)
        if clip_fn is not None:
            x = clip_fn(x)
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
    loss_fn: Any,
    init_x: Any,
    lr: Any = None,
    max_iters: int = 100,
    patience: Any = None,
    *,
    has_aux: bool = False,
    aux_history: dict | None = None,
    clip_fn: Any = None,
    record_diagnostics: bool = False,
    div_fn: Any = None,
    snap_interval: int = 0,
    history: list | None = None,
    snap_error_fn: Any = None,
    error_history: list | None = None,
    log_fn: Any = None,
    log_interval: int = 10,
    grad_proj_fn: Any = None,
) -> tuple:
    """L-BFGS with zoom line-search.

    When ``has_aux=True``, ``loss_fn(x) → (scalar_loss, aux_dict)``; we
    wrap to a scalar for optax's line-search, but reuse
    ``jax.value_and_grad(loss_fn, has_aux=True)`` so each iteration only
    computes one forward pass (the value/aux pair shares trace work). Aux
    values land in ``aux_history`` exactly as in :func:`_run_optim`.

    ``clip_fn`` projects ``x`` back into the feasible set after each
    update (e.g. ``jnp.clip(x, x_min, 1.0)`` for density fields).
    ``grad_proj_fn``, when provided, is called on the gradient (numpy)
    before the L-BFGS update — used by velocity-field optimisation to
    project onto the divergence-free subspace (Helmholtz).

    The ``lr``, ``patience``, ``record_diagnostics``, and ``div_fn``
    parameters are accepted but ignored, so call sites that pass them
    work without modification.

    Returns ``(final_x, losses, diag)`` where ``diag`` is
    ``{"grad_norms": [...]}``. Shape matches :func:`_run_optim`.
    """
    if has_aux:

        def _scalar_loss(x: Any) -> Any:
            return loss_fn(x)[0]

        vg = jax.value_and_grad(loss_fn, has_aux=True)
    else:
        _scalar_loss = loss_fn
        vg = jax.value_and_grad(loss_fn)

    import os
    import resource

    _mosaic_lbfgs_diag = os.environ.get("MOSAIC_LBFGS_DIAG", "0") == "1"

    def _probe(label: str, i: int) -> None:
        if not _mosaic_lbfgs_diag:
            return
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            live = len(jax.live_arrays())
        except Exception:
            live = -1
        print(
            f"[lbfgs-diag] iter={i:3d} {label:10s} RSS={rss_kb / 1024:.1f}MiB live_arrays={live}"
        )

    solver = optax.lbfgs()

    # Wrap one full L-BFGS step in a single jit. This is critical: the
    # default linesearch (``scale_by_zoom_linesearch``) builds a fresh
    # ``functools.partial`` body for its ``jax.lax.while_loop`` every
    # call, and the while_loop trace cache keys off body identity. Driving
    # the loop in Python therefore re-traces the linesearch every iter,
    # accumulating compiled XLA modules until the host runs out of memory.
    # JIT-ing the whole step gives the linesearch a stable identity inside
    # one cached trace — recompilation happens once per (x.shape, dtype).
    def _step_body(x: Any, opt_state: Any) -> tuple:
        if has_aux:
            (value, aux), grad = vg(x)
        else:
            value, grad = vg(x)
            aux = None
        if grad_proj_fn is not None:
            grad = grad_proj_fn(grad)
        grad_norm = jnp.linalg.norm(grad.ravel())
        updates, opt_state = solver.update(
            grad, opt_state, x, value=value, grad=grad, value_fn=_scalar_loss
        )
        x = optax.apply_updates(x, updates)
        if clip_fn is not None:
            x = clip_fn(x)
        return x, opt_state, value, grad_norm, aux

    step = jax.jit(_step_body)

    opt_state = solver.init(init_x)
    x = init_x
    losses: list[float] = []
    grad_norms: list[float] = []
    _probe("init", -1)
    for i in range(max_iters):
        x, opt_state, value, grad_norm, aux = step(x, opt_state)
        if has_aux:
            _append_aux(aux_history, aux)
        grad_norms.append(float(grad_norm))
        loss_val = float(value)
        losses.append(loss_val)
        _probe("after-step", i)
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
