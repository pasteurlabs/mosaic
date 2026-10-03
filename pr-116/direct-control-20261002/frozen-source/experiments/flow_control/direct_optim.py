"""Direct action optimization with completion-time, coarse-only checkpoints.

Wall time is the primary comparison. An AD query requests one forward rollout
and one VJP; a derivative-free query requests one forward rollout. These are
API-level counts, not counts of backend-internal adjoint replay. Method-local
initialization and first-call compilation are charged. The caller's common
preparation and already-running solver service are outside this engine.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any


class _BudgetStop(Exception):
    """Internal control flow; a budget limit is not an optimizer failure."""


def optimize_actions(
    loss: Callable,
    initial_controls: Any,
    config: Any,
    method: str,
    optimizer_settings: Mapping[str, Any] | None = None,
    budgets: Sequence[float] = (30, 60, 120),
    seed: int = 0,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Optimize bounded controls, retaining every coarse query and deadline.

    ``loss`` maps physical controls to a JAX scalar. ``config.control_bound``
    defines their bound. Methods are ``adam``, ``spsa``, and ``powell``. Settings
    support ``lr``, SPSA ``epsilon``/``directions``, Powell ``xtol``/``ftol``,
    ``max_queries`` (default 512), and ``query_budgets`` (32,64,128).

    All methods evaluate the exact supplied initial controls first, then use
    tanh-bounded latents with a common inverse clipping of 1e-6. A snapshot is
    selected from successful, completed *coarse* queries only. Overshooting
    calls remain in the ledger but cannot improve an earlier checkpoint.
    Unreached query checkpoints are explicitly unavailable. Early convergence
    leaves a valid best-so-far wall-time snapshot without inventing more work.
    Returned snapshot controls are NumPy arrays; JSON callers must serialize
    them or save them separately. Ordinary runtime failures return partial
    evidence; KeyboardInterrupt/SystemExit propagate.
    """
    start = clock()
    import hashlib

    import jax
    import jax.numpy as jnp
    import numpy as np

    settings = dict(optimizer_settings or {})
    method = str(method).lower()
    deadlines = sorted({float(b) for b in budgets})
    query_budgets = sorted(
        {int(q) for q in settings.get("query_budgets", (32, 64, 128))}
    )
    max_queries = int(settings.get("max_queries", 512))
    bound = float(config.control_bound)
    if method not in {"adam", "spsa", "powell"}:
        raise ValueError(f"unknown method: {method}")
    if not deadlines or any(not np.isfinite(b) or b <= 0 for b in deadlines):
        raise ValueError("wall budgets must be finite and positive")
    if max_queries < 1 or any(q < 1 for q in query_budgets):
        raise ValueError("query budgets must be positive")
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("control bound must be finite and positive")
    initial = np.asarray(initial_controls, dtype=np.float32)
    if (
        initial.size == 0
        or not np.all(np.isfinite(initial))
        or np.any(np.abs(initial) > bound)
    ):
        raise ValueError("initial controls must be nonempty, finite and within bounds")
    lr = float(settings.get("lr", 0.01))
    epsilon = float(settings.get("epsilon", 0.01))
    directions = int(settings.get("directions", 1))
    if (
        lr <= 0
        or not np.isfinite(lr)
        or epsilon <= 0
        or not np.isfinite(epsilon)
        or directions < 1
    ):
        raise ValueError(
            "learning rate, perturbation and direction count must be positive"
        )
    events: list[dict[str, Any]] = []
    candidates: list[tuple[dict[str, Any], Any]] = []
    failure = None
    stop_reason = "not_started"
    optimizer_updates = 0
    first_ad_call_s = None
    initialization_s = None
    realized_initial = None
    max_time = max(deadlines)

    def elapsed():
        return max(0.0, float(clock() - start))

    def control_hash(controls: Any):
        return hashlib.sha256(
            np.ascontiguousarray(controls, dtype=np.float32).tobytes()
        ).hexdigest()

    def physical(latent: Any):
        return bound * jnp.tanh(latent)

    def latent_loss(latent: Any):
        return loss(physical(latent))

    value_grad = jax.value_and_grad(latent_loss)

    def query(controls: Any, *, latent: Any = None):
        nonlocal first_ad_call_s, stop_reason
        # Hash/copy and device synchronization are method overhead, charged before
        # deciding whether another indivisible oracle call may start.
        saved = np.asarray(jax.device_get(controls), dtype=np.float32).copy()
        digest = control_hash(saved)
        began = elapsed()
        if began >= max_time:
            stop_reason = "wall_budget"
            raise _BudgetStop
        if len(events) >= max_queries:
            stop_reason = "query_limit"
            raise _BudgetStop
        event = {
            "query": len(events) + 1,
            "started_s": began,
            "completed_s": None,
            "kind": "value_grad" if latent is not None else "forward",
            "forward_count": 1,
            "vjp_count": int(latent is not None),
            "controls_sha256": digest,
            "loss": None,
            "error": None,
            "successful": False,
        }
        events.append(event)
        try:
            if latent is None:
                value = float(jax.block_until_ready(loss(controls)))
                gradient = None
            else:
                value, gradient = jax.block_until_ready(value_grad(latent))
                value = float(value)
                if not np.all(np.isfinite(np.asarray(gradient))):
                    raise FloatingPointError("nonfinite objective gradient")
            if not np.isfinite(value):
                raise FloatingPointError("nonfinite objective value")
            event["loss"] = value
            event["successful"] = True
            return value, gradient
        except Exception as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event["completed_s"] = elapsed()
            event["overshoot_s"] = max(0.0, event["completed_s"] - max_time)
            if latent is not None and first_ad_call_s is None:
                first_ad_call_s = event["completed_s"] - began
            if event["successful"]:
                candidates.append((event, saved))

    try:
        latent = jnp.arctanh(
            jnp.clip(jnp.asarray(initial) / bound, -1 + 1e-6, 1 - 1e-6)
        )
        jax.block_until_ready(latent)
        realized_initial = np.asarray(physical(latent)).copy()
        initialization_s = elapsed()
        query(jnp.asarray(initial))
        if method in {"adam", "spsa"}:
            import optax

            optimizer = optax.adam(lr)
            opt_state = optimizer.init(latent)
            rng = np.random.default_rng(seed)
            while True:
                if method == "adam":
                    _, gradient = query(physical(latent), latent=latent)
                else:
                    gradient = jnp.zeros_like(latent)
                    for _ in range(directions):
                        delta = jnp.asarray(
                            rng.choice([-1.0, 1.0], size=initial.shape),
                            dtype=latent.dtype,
                        )
                        plus, _ = query(physical(latent + epsilon * delta))
                        minus, _ = query(physical(latent - epsilon * delta))
                        gradient = gradient + ((plus - minus) / (2 * epsilon)) * delta
                    gradient = gradient / directions
                if not np.all(np.isfinite(np.asarray(gradient))):
                    raise FloatingPointError("nonfinite optimizer gradient")
                updates, opt_state = optimizer.update(gradient, opt_state, latent)
                latent = optax.apply_updates(latent, updates)
                jax.block_until_ready(latent)
                optimizer_updates += 1
        else:
            from scipy.optimize import minimize

            shape = initial.shape

            def objective(flat: Any):
                value, _ = query(
                    physical(jnp.asarray(flat.reshape(shape), dtype=jnp.float32))
                )
                return value

            result = minimize(
                objective,
                np.asarray(latent, dtype=np.float64).ravel(),
                method="Powell",
                bounds=[(-8.0, 8.0)] * initial.size,
                options={
                    "xtol": float(settings.get("xtol", 1e-3)),
                    "ftol": float(settings.get("ftol", 1e-5)),
                    "maxfev": max_queries,
                },
            )
            optimizer_updates = int(result.nit)
            stop_reason = "converged" if result.success else "powell_stopped"
            if not result.success and result.status == 1:
                stop_reason = "query_limit"
            elif not result.success:
                failure = str(result.message)
    except _BudgetStop:
        pass
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        stop_reason = "failure"
    finished_s = elapsed()
    snapshots = []
    for kind, limits in [("wall_time_s", deadlines), ("queries", query_budgets)]:
        for limit in limits:
            eligible = [
                (event, controls)
                for event, controls in candidates
                if event["completed_s"]
                <= (limit if kind == "wall_time_s" else max_time)
                and (kind == "wall_time_s" or event["query"] <= limit)
            ]
            reached = (
                finished_s >= limit
                if kind == "wall_time_s"
                else any(
                    e["query"] >= limit and e["completed_s"] <= max_time for e in events
                )
            )
            available = bool(eligible) and (kind == "wall_time_s" or reached)
            best = (
                min(eligible, key=lambda item: item[0]["loss"]) if available else None
            )
            snapshots.append(
                {
                    "budget_kind": kind,
                    "budget": limit,
                    "available": available,
                    "reached": reached,
                    "best_loss": best[0]["loss"] if best else None,
                    "controls": best[1].copy() if best else None,
                    "controls_sha256": best[0]["controls_sha256"] if best else None,
                    "selected_query": best[0]["query"] if best else None,
                    "selected_completed_s": best[0]["completed_s"] if best else None,
                }
            )
    return {
        "method": method,
        "optimizer_settings": settings,
        "resolved_settings": {
            "lr": lr,
            "epsilon": epsilon,
            "directions": directions,
            "max_queries": max_queries,
            "query_budgets": query_budgets,
            "xtol": float(settings.get("xtol", 1e-3)),
            "ftol": float(settings.get("ftol", 1e-5)),
            "powell_latent_bound": 8.0,
            "inverse_tanh_clip": 1e-6,
        },
        "seed": int(seed),
        "snapshots": snapshots,
        "query_events": events,
        "failure": failure,
        "completed": failure is None,
        "stop_reason": stop_reason,
        "initial_controls_sha256": control_hash(initial),
        "realized_initial_controls_sha256": control_hash(realized_initial)
        if realized_initial is not None
        else None,
        "initial_roundtrip_max_absolute": float(
            np.max(np.abs(realized_initial - initial))
        )
        if realized_initial is not None
        else None,
        "accounting": {
            "wall_time_s": finished_s,
            "initialization_s": initialization_s,
            "initial_objective_s": events[0]["completed_s"] - events[0]["started_s"]
            if events
            else None,
            "first_ad_call_including_compilation_s": first_ad_call_s,
            "forward_count": len(events),
            "vjp_count": sum(e["vjp_count"] for e in events),
            "successful_queries": sum(e["successful"] for e in events),
            "failed_queries": sum(not e["successful"] for e in events),
            "optimizer_updates": optimizer_updates,
            "overshoot_s": max(0.0, finished_s - max_time),
            "oracle_wall_time_s": sum(
                e["completed_s"] - e["started_s"] for e in events
            ),
            "note": (
                "Method-local cold initialization/first-call costs charged; common preparation "
                "and service startup external. Forward/VJP counts are requests, including "
                "failures, not backend solve counts. Oracle time is included in wall time. "
                "Query snapshots also obey the maximum wall deadline."
            ),
        },
    }
