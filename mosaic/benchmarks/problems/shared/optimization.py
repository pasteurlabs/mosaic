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

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.optimize


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


def _check_finite_loss(loss_val: float, i: int) -> None:
    """Raise once the loss goes NaN/Inf: the optimiser cannot recover from it."""
    if not math.isfinite(loss_val):
        raise FloatingPointError(
            f"optimization loss is not finite ({loss_val}) at iteration {i}"
        )


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
    min_rel_improvement: float = 0.0,
) -> tuple:
    """Adam with patience-based early stopping.

    Stops after ``patience`` iterations without an improvement, where an
    improvement is a loss below the best so far by more than
    ``min_rel_improvement`` times its magnitude. The best only moves on such
    an improvement, so small gains accumulate against it and a slow, steady
    descent still resets the count.

    When ``has_aux=True``, ``loss_fn(x) → (scalar_loss, aux_dict)`` and each
    aux label's per-iter trace lands in ``aux_history[label]`` (caller
    provides the dict, primitive populates it).

    ``clip_fn`` (e.g. ``lambda x: jnp.clip(x, x_min, 1.0)``) is applied to
    ``x`` after each Adam update so the iterate stays in the feasible set.

    Returns ``(final_x, losses, diag)`` where ``losses`` is the per-iter
    scalar-loss list and ``diag`` is a dict with optional diagnostic
    time-series (``None`` when ``record_diagnostics=False``):

      * ``grad_norms``: per-iter ``‖∇L‖₂``
      * ``grad_divs``:  per-iter ``max|∇·g|`` (only when ``div_fn`` set)
      * ``ic_divs``:    per-iter ``max|∇·u|`` (only when ``div_fn`` set)
      * ``converged``:  whether the patience-based early stop fired

    Raises ``FloatingPointError`` as soon as the loss is NaN/Inf.
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
        _check_finite_loss(loss_val, i)
        losses.append(loss_val)
        if i == 0 or loss_val < best - min_rel_improvement * abs(best):
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
    diag["converged"] = no_improve >= patience
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
    bounds: tuple[float, float] | None = None,
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
    """SciPy's L-BFGS-B on a single array ``init_x``.

    When ``has_aux=True``, ``loss_fn(x)`` returns ``(scalar_loss, aux_dict)``
    and the aux values land in ``aux_history`` as in :func:`_run_optim`.
    ``bounds=(lo, hi)`` bounds every entry of ``x``. ``grad_proj_fn`` is
    applied to every gradient L-BFGS-B sees, e.g. to project velocity fields
    onto the divergence-free subspace.

    L-BFGS-B iterates in float64 and ``loss_fn`` receives ``x`` in
    ``init_x``'s dtype. It stops after ``max_iters`` iterations, when the
    gradient's 2-norm falls below 1e-7, or when its line search fails from an
    empty history. SciPy's own gradient test is off: it bounds the largest
    entry, which on a large ``x`` fires orders of magnitude earlier.

    A trial point with a non-finite loss or gradient is reported as the loss
    at the current iterate plus its magnitude, which the line search always
    rejects. Reporting ``inf`` would make L-BFGS-B stop and claim convergence.
    The reported value scales with the loss: one far above it makes the line
    search shrink the step until ``x`` no longer changes at ``loss_fn``'s
    precision, and the run then ends as converged. Evaluations are
    cached by ``x`` at ``loss_fn``'s precision, so a restart from an accepted
    iterate reuses its value without asking the solver again.

    ``lr``, ``patience``, ``record_diagnostics`` and ``div_fn`` are accepted
    and ignored, so call sites can pass the same arguments as to
    :func:`_run_optim`.

    Returns ``(final_x, losses, diag)``. ``losses`` and ``diag["grad_norms"]``
    cover ``init_x`` and every iterate, ending at ``final_x``.
    ``diag["converged"]`` is false when the run hit ``max_iters`` or the line
    search failed. Raises ``FloatingPointError`` when the loss at ``init_x``
    is not finite.
    """
    shape, dtype = init_x.shape, init_x.dtype
    vg = jax.value_and_grad(loss_fn, has_aux=has_aux)

    @jax.jit
    def value_and_grad(x: Any) -> tuple:
        if has_aux:
            (value, aux), grad = vg(x)
        else:
            (value, grad), aux = vg(x), None
        if grad_proj_fn is not None:
            grad = grad_proj_fn(grad)
        return value, grad, aux

    def to_x(z: np.ndarray) -> np.ndarray:
        return z.reshape(shape).astype(dtype)

    # Evaluations since the last accepted iterate, plus that iterate.
    evals: dict[bytes, tuple] = {}

    def evaluate(z: np.ndarray) -> tuple:
        key = to_x(z).tobytes()
        if key not in evals:
            value, grad, aux = value_and_grad(jnp.asarray(to_x(z)))
            value = float(value)
            grad = np.asarray(grad, dtype=np.float64).ravel()
            finite = math.isfinite(value) and np.all(np.isfinite(grad))
            evals[key] = (value, grad, aux, finite)
        return evals[key]

    def objective(z: np.ndarray) -> tuple:
        value, grad, _, finite = evaluate(z)
        if not finite:
            return losses[-1] + abs(losses[-1]), np.zeros_like(grad)
        return value, grad

    losses: list[float] = []
    grad_norms: list[float] = []
    last_z: list[np.ndarray] = []

    def record(z: np.ndarray) -> None:
        entry = evaluate(z)
        evals.clear()
        evals[to_x(z).tobytes()] = entry
        value, grad, aux, _ = entry
        i = len(losses)
        _check_finite_loss(value, i)
        if has_aux:
            _append_aux(aux_history, aux)
        losses.append(value)
        grad_norms.append(float(np.linalg.norm(grad)))
        last_z[:] = [z.copy()]
        if i == 0:
            return
        if snap_interval > 0 and i % snap_interval == 0:
            x = to_x(z)
            if history is not None:
                history.append(x)
            if snap_error_fn is not None and error_history is not None:
                error_history.append(snap_error_fn(x))
        if log_fn is not None and i % log_interval == 0:
            log_fn(i, value)

    z0 = np.asarray(init_x, dtype=np.float64).ravel()
    record(z0)

    def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
        record(intermediate_result.x)
        if grad_norms[-1] < 1e-7:
            raise StopIteration

    res = scipy.optimize.minimize(
        objective,
        z0,
        jac=True,
        method="L-BFGS-B",
        bounds=None if bounds is None else scipy.optimize.Bounds(*bounds),
        callback=callback,
        options={"maxiter": max_iters, "gtol": 0.0, "ftol": 0.0},
    )
    # L-BFGS-B does not call back on the iterate at which it converges.
    if not np.array_equal(res.x, last_z[0]):
        record(res.x)
    converged = res.success or grad_norms[-1] < 1e-7
    diag = {"grad_norms": grad_norms, "converged": bool(converged)}
    return jnp.asarray(to_x(res.x)), losses, diag
