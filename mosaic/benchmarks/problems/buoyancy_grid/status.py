# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution checks for solver-in-the-loop training and its reference trajectories."""

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
    """Require complete, finite SOL training and trustworthy references/VJPs."""
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
        cases = row.get("training_cases", [{"seed": row.get("seeds", {}).get("train")}])
        if (
            not isinstance(cases, list)
            or not cases
            or any(
                not isinstance(case, dict) or not isinstance(case.get("seed"), int)
                for case in cases
            )
        ):
            return "anomaly", "invalid training-case descriptors"
        for case in cases:
            if set(case) - {
                "seed",
                "position_jitter",
                "width_jitter",
                "amplitude_jitter",
            } or any(
                not _finite(value) or value < 0
                for key, value in case.items()
                if key != "seed"
            ):
                return "anomaly", "invalid training-case IC jitter"
        audits = reference.get("splits", {})
        expected_splits = {
            "train",
            "validation",
            "test",
            *(f"train_{i}" for i in range(1, len(cases))),
        }
        if set(audits) != expected_splits:
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
        interval = row.get("validation_interval")
        if not isinstance(interval, int) or interval < 1:
            return "anomaly", "invalid validation checkpoint interval"
        schedule = sorted({0, updates, *range(interval, updates + 1, interval)})
        train_frames, horizon = row.get("train_frames"), row.get("unroll")
        if (
            not isinstance(train_frames, int)
            or not isinstance(horizon, int)
            or not 1 <= horizon <= train_frames
        ):
            return "anomaly", "invalid training trajectory/window length"
        count = train_frames + 1 - horizon
        starts = row.get("window_starts", [])
        if (
            not isinstance(row.get("window_seed"), int)
            or len(starts) != updates
            or any(
                not isinstance(start, int) or not 0 <= start < count for start in starts
            )
        ):
            return "anomaly", "invalid shuffled training windows"
        case_ids = row.get("window_cases", [0] * updates)
        if len(case_ids) != updates or any(
            not isinstance(case, int) or not 0 <= case < len(cases) for case in case_ids
        ):
            return "anomaly", "invalid training-case schedule"
        if any(
            len(set(case_ids[i : i + len(cases)])) != len(case_ids[i : i + len(cases)])
            for i in range(0, updates, len(cases))
        ):
            return "anomaly", "training-case schedule is not balanced per round"
        for case in range(len(cases)):
            case_starts = [
                start
                for selected_case, start in zip(case_ids, starts, strict=True)
                if selected_case == case
            ]
            if any(
                len(set(case_starts[i : i + count])) != len(case_starts[i : i + count])
                for i in range(0, len(case_starts), count)
            ):
                return (
                    "anomaly",
                    "training case repeats a window before covering its permutation",
                )
        if row.get(
            "training_case_counts", [case_ids.count(i) for i in range(len(cases))]
        ) != [case_ids.count(i) for i in range(len(cases))]:
            return "anomaly", "incorrect training-case update counts"
        if row.get("normalization_case", 0) != 0:
            return "anomaly", "normalization must use the unchanged first training case"
        seeds = row.get("seeds", {})
        if (
            set(seeds) != {"train", "validation", "test"}
            or len(set(seeds.values())) != 3
        ):
            return "anomaly", "buoyancy train/validation/test seeds must be distinct"
        all_seeds = [case["seed"] for case in cases] + [
            seeds["validation"],
            seeds["test"],
        ]
        if seeds["train"] != cases[0]["seed"] or len(set(all_seeds)) != len(all_seeds):
            return (
                "anomaly",
                "training cases must have distinct seeds disjoint from held-out seeds",
            )
        learning_rates = row.get("learning_rates", [])
        if not learning_rates or not all(
            _finite(lr) and lr > 0 for lr in learning_rates
        ):
            return "anomaly", "missing buoyancy learning-rate candidates"
        for arm in (
            "reference",
            "uncorrected",
            "solver_in_loop",
        ):
            divergence = row.get("divergence_rms", {}).get(arm, [])
            if len(divergence) != frames + 1 or not all(
                _finite(v) and v >= 0 for v in divergence
            ):
                return "anomaly", f"incomplete or nonfinite {arm} divergence"
        for arm in ("uncorrected", "solver_in_loop"):
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
            for key in (
                "total_training_seconds",
                "total_validation_seconds",
                "total_seconds",
            ):
                if not _finite(result.get(key)) or result[key] < 0:
                    return "anomaly", f"invalid {arm} measured cost"
            for lr, candidate in zip(learning_rates, candidates, strict=True):
                if candidate.get("learning_rate") != lr:
                    return "anomaly", f"unmatched {arm} learning-rate candidate"
                for key in ("training_seconds", "validation_seconds"):
                    if not _finite(candidate.get(key)) or candidate[key] < 0:
                        return "anomaly", f"invalid {arm} candidate measured cost"
                checkpoints = candidate.get("validation_checkpoints", [])
                if [point.get("update") for point in checkpoints] != schedule or any(
                    not _finite(point.get("loss")) or point["loss"] < 0
                    for point in checkpoints
                ):
                    return (
                        "anomaly",
                        f"incomplete or nonfinite {arm} validation checkpoints",
                    )
                best = min(
                    checkpoints, key=lambda point: (point["loss"], point["update"])
                )
                if (
                    candidate.get("selected_update") != best["update"]
                    or candidate.get("validation_loss") != best["loss"]
                ):
                    return (
                        "anomaly",
                        f"{arm} did not select its earliest best validation checkpoint",
                    )
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
            expected = min(
                range(len(candidates)),
                key=lambda i: (
                    candidates[i]["validation_loss"],
                    candidates[i]["selected_update"],
                    i,
                ),
            )
            if (
                selected != expected
                or result.get("selected_update")
                != candidates[selected]["selected_update"]
            ):
                return (
                    "anomaly",
                    f"{arm} selection did not minimize validation loss with earliest ties",
                )
            train_cost = sum(c["training_seconds"] for c in candidates)
            validation_cost = sum(c["validation_seconds"] for c in candidates)
            if any(
                not math.isclose(result[key], value, rel_tol=1e-9, abs_tol=1e-9)
                for key, value in (
                    ("total_training_seconds", train_cost),
                    ("total_validation_seconds", validation_cost),
                    ("total_seconds", train_cost + validation_cost),
                )
            ):
                return "anomaly", f"{arm} measured cost omits training or validation"
    return None
