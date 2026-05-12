"""Generic optimizer primitives — :func:`_run_optim` (Adam) and :func:`_run_lbfgs`.

Problem-specific optimization harnesses (drag-opt, IC recovery, SIMP
topology optimisation, conductivity recovery) live in each problem's
own ``optimization.py``. This module only holds the two cross-problem
primitives those harnesses share.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax


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
