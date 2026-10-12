# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution checks; winning against supervision is deliberately not a CI gate."""

import math
from collections.abc import Iterator
from typing import Any

from mosaic.benchmarks.core.status_checks import CheckOutcome, OptimizationSummary


def _records(metrics: dict) -> Iterator[dict]:
    if "gradient_check" in metrics or metrics.get("status") == "failed":
        yield metrics
    else:
        for value in metrics.values():
            if isinstance(value, dict):
                yield from _records(value)


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
    )


def corrector_execution(summary: OptimizationSummary) -> CheckOutcome:
    """Require real, finite training in every arm and trustworthy references/VJPs."""
    rows = list(_records(summary.metrics))
    if not rows:
        return "anomaly", "missing buoyancy corrector metrics"
    for row in rows:
        if row.get("status") != "ok":
            return "anomaly", "buoyancy corrector execution failed"
        for value, label, limit in (
            (
                row.get("gradient_check", {}).get("relative_error"),
                "gradient check",
                0.05,
            ),
            (
                row.get("gradient_check_nonzero", {}).get("relative_error"),
                "nonzero-corrector gradient check",
                0.05,
            ),
            (
                row.get("reference", {}).get("time_refinement_error"),
                "reference refinement",
                0.05,
            ),
            (row.get("recurrence_error"), "restart equivalence", 1e-4),
            (
                row.get("reference", {}).get("audit_gap_ratio"),
                "reference/coarse-gap ratio",
                0.25,
            ),
        ):
            if not _finite(value) or not 0 <= value <= limit:
                return "anomaly", f"buoyancy {label} failed"
        updates = row.get("updates", 0)
        reference = row.get("reference", {})
        audits = reference.get("splits", {})
        if set(audits) != {"train", "validation", "test"}:
            return "anomaly", "missing buoyancy target-split reference audit"
        for audit in audits.values():
            if any(
                not _finite(audit.get(key)) or audit[key] < 0
                for key in ("time_refinement_error", "coarse_gap", "audit_gap_ratio")
            ):
                return "anomaly", "nonfinite buoyancy reference split audit"
        for key in ("time_refinement_error", "audit_gap_ratio"):
            if reference.get(key) != max(a[key] for a in audits.values()):
                return (
                    "anomaly",
                    "buoyancy reference summary must include worst target split",
                )
        frames = row.get("eval_frames", 0)
        if (
            not isinstance(updates, int)
            or updates < 1
            or not isinstance(frames, int)
            or frames < 1
        ):
            return "anomaly", "invalid buoyancy training/evaluation budget"
        seeds = row.get("seeds", {})
        if (
            set(seeds) != {"train", "validation", "test"}
            or len(set(seeds.values())) != 3
        ):
            return "anomaly", "buoyancy train/validation/test seeds must be distinct"
        learning_rates = row.get("learning_rates", [])
        if not learning_rates or not all(
            _finite(lr) and lr > 0 for lr in learning_rates
        ):
            return "anomaly", "missing buoyancy learning-rate candidates"
        for arm in (
            "reference",
            "uncorrected",
            "supervised",
            "solver_in_loop",
            "stop_gradient",
        ):
            divergence = row.get("divergence_rms", {}).get(arm, [])
            if len(divergence) != frames + 1 or not all(
                _finite(v) and v >= 0 for v in divergence
            ):
                return "anomaly", f"incomplete or nonfinite {arm} divergence"
        for arm in ("uncorrected", "supervised", "solver_in_loop", "stop_gradient"):
            result = row.get(arm, {})
            for key in ("loss", "velocity_mse", "scalar_mse"):
                values = result.get(key, [])
                if len(values) != frames + 1 or not all(
                    _finite(v) and v >= 0 for v in values
                ):
                    return "anomaly", f"incomplete or nonfinite {arm} rollout"
            if arm == "uncorrected":
                continue
            candidates = result.get("candidates", [])
            if len(candidates) != len(learning_rates):
                return "anomaly", f"incomplete {arm} candidate search"
            selected = result.get("selected_candidate")
            if not isinstance(selected, int) or not 0 <= selected < len(candidates):
                return "anomaly", f"missing {arm} validation selection"
            for key in ("preprocessing_seconds", "total_training_seconds"):
                if not _finite(result.get(key)) or result[key] < 0:
                    return "anomaly", f"invalid {arm} training cost"
            for lr, candidate in zip(learning_rates, candidates, strict=True):
                if candidate.get("learning_rate") != lr:
                    return "anomaly", f"unmatched {arm} learning-rate candidate"
                if (
                    not _finite(candidate.get("training_seconds"))
                    or candidate["training_seconds"] < 0
                ):
                    return "anomaly", f"invalid {arm} candidate training cost"
                history = candidate.get("training_loss", [])
                if (
                    candidate.get("completed_updates") != updates
                    or len(history) != updates
                ):
                    return "anomaly", f"incomplete {arm} optimizer updates"
                if not all(_finite(v) for v in history) or not _finite(
                    candidate.get("validation_loss")
                ):
                    return "anomaly", f"nonfinite {arm} training/validation loss"
            if candidates[selected]["validation_loss"] != min(
                c["validation_loss"] for c in candidates
            ):
                return "anomaly", f"{arm} selection did not minimize validation loss"
    return None
