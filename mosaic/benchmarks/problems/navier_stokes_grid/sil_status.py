# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Status check of the Kolmogorov solver-in-the-loop benchmark.

The check gates feasibility only: whether a solver can be trained through and
its comparison is sound. There is deliberately no criterion on which arm
wins; a run passes even when the supervised arm beats solver-in-the-loop
training.
"""

from __future__ import annotations

from collections.abc import Callable

from mosaic.benchmarks.core.status_checks import CheckOutcome, OptimizationSummary

# Each gate and the reason given when it fails.
_GATES = (
    ("audit_passed", "the reference audit failed"),
    ("closure_passed", "chained solver calls do not close"),
    ("completed", "a training run stopped before its update budget"),
    ("fd_passed", "the finite-difference check of the training gradient failed"),
    ("finite", "a loss, gradient norm or rollout error is not finite"),
    ("selection_valid", "a selected checkpoint is not a valid one"),
)


def feasible() -> Callable[[OptimizationSummary], CheckOutcome]:
    """Anomaly unless every gate of the comparison holds."""

    def _check(s: OptimizationSummary) -> CheckOutcome:
        m = s.metrics
        for key, reason in _GATES:
            if m.get(key) is not True:
                detail = (
                    m.get("reason")
                    if key in ("audit_passed", "closure_passed")
                    else None
                )
                return ("anomaly", f"{reason}: {detail}" if detail else reason)
        return None

    return _check
