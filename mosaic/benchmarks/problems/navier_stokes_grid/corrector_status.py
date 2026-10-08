"""Execution checks for the bounded corrector benchmark, without a win criterion."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any

from mosaic.benchmarks.core.status_checks import CheckOutcome, OptimizationSummary


def _records(metrics: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if (
        "reference_kind" in metrics
        or metrics.get("status") == "failed"
        or metrics.get("failure_type")
    ):
        yield metrics
    else:
        for value in metrics.values():
            if isinstance(value, dict):
                yield from _records(value)


def corrector_execution(updates: int) -> Callable[[OptimizationSummary], CheckOutcome]:
    """Require admitted references, actual training and finite verified gradients."""

    def check(summary: OptimizationSummary) -> CheckOutcome:
        records = list(_records(summary.metrics))
        if not records:
            return "anomaly", "missing corrector execution metrics"
        for row in records:
            if row.get("status") == "failed" or row.get("failure_type"):
                return "anomaly", "corrector execution failed"
            for key in (
                "eligible_for_corrector_training",
                "reference_convergence_passed",
                "valid_for_vjp_ranking",
                "completed",
                "supervised_completed",
            ):
                if row.get(key) is not True:
                    return "anomaly", f"corrector {key} did not pass"
            if row.get("reference_kind") != "solver_self_refined":
                return "anomaly", "corrector requires same-solver references"
            seeds = row.get("model_seeds", [])
            if not seeds or len(set(seeds)) != len(seeds):
                return "anomaly", "missing or duplicate corrector model seeds"
            for key in (
                "total_optimizer_updates",
                "stop_gradient_total_optimizer_updates",
                "supervised_total_optimizer_updates",
            ):
                if row.get(key) != updates * len(seeds):
                    return "anomaly", f"corrector {key} differs from configured budget"
            checks = row.get("end_to_end_fd_checks_by_seed", [])
            if len(checks) != len(seeds) or any(not values for values in checks):
                return "anomaly", "missing corrector gradient checks"
            for values in checks:
                for diagnostic in values:
                    error = diagnostic.get("relative_error")
                    if (
                        not isinstance(error, int | float)
                        or not math.isfinite(error)
                        or error >= 0.05
                    ):
                        return "anomaly", "corrector model gradient check failed"
            for key in (
                "final_grad_norm",
                "stop_gradient_final_grad_norm",
                "mean_rollout_error",
                "stop_gradient_mean_rollout_error",
                "supervised_mean_rollout_error",
                "final_train_loss",
                "stop_gradient_final_train_loss",
            ):
                value = row.get(key)
                if not isinstance(value, int | float) or not math.isfinite(value):
                    return "anomaly", f"nonfinite or missing corrector {key}"
        return None

    return check
