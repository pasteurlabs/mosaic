# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the timing harness.

Covers:
- ``classify_failure``: maps exception name + message to one of five labels.
- ``TimedResult.as_record``: success and failure record shapes.
- ``run_timed_trials``: warmup discard, trial timing, wall-limit early exit,
  ``capture_value`` flag, and exception classification on warmup vs. trial.
"""

from __future__ import annotations

import time

import pytest

from mosaic.benchmarks.core.harness import (
    TimedResult,
    classify_failure,
    run_timed_trials,
)

# ── classify_failure ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "exc_name, exc_msg, expected",
    [
        # ContainerDied is matched by exception NAME, not message
        ("ContainerDied", "anything", "container_died"),
        # OOM: matched by substrings in the message
        ("RuntimeError", "RESOURCE_EXHAUSTED: out of memory on device", "OOM"),
        ("RuntimeError", "CUDA_ERROR_OUT_OF_MEMORY", "OOM"),
        ("Exception", "Operation crashed: out of memory.", "OOM"),
        # Timeout: matched by exception name
        ("TimeoutError", "deadline exceeded", "timeout"),
        ("WatchdogTimeout", "", "timeout"),
        ("WatchdogError", "", "timeout"),
        ("ReadTimeout", "", "timeout"),
        ("ConnectTimeout", "", "timeout"),
        # Timeout: any exception NAME containing "timeout" (case-insensitive)
        ("MyCustomTimeoutThing", "", "timeout"),
        # NaN: matched by substring
        ("RuntimeError", "result contained nan values", "nan"),
        ("RuntimeError", "loss is not finite", "nan"),
        # Default fallback
        ("ValueError", "bad shape", "error"),
        ("RuntimeError", "", "error"),
    ],
)
def test_classify_failure_categories(exc_name, exc_msg, expected):
    assert classify_failure(exc_name, exc_msg) == expected


def test_classify_failure_oom_beats_timeout_when_both_keywords_present():
    """OOM is checked before timeout in the cascade, so 'timeout' in the
    message shouldn't promote an OOM error to timeout — but a timeout-named
    exception with 'out of memory' in the message *will* classify as OOM
    because OOM is checked first. Document the actual order.
    """
    # OOM-named substring beats other message content because the OOM branch
    # runs before the timeout branch.
    assert classify_failure("RuntimeError", "out of memory; timeout fallback") == "OOM"


def test_classify_failure_container_died_short_circuits():
    """ContainerDied is matched by name and ignores message content."""
    assert classify_failure("ContainerDied", "out of memory") == "container_died"


# ── TimedResult.as_record ─────────────────────────────────────────────────────


def test_as_record_success_multi_trial():
    """Successful result with multiple trials reports mean and non-zero std."""
    r = TimedResult(
        times=[0.1, 0.2, 0.3],
        mem={"vram_peak_mib": 1024.0, "ram_peak_mib": 500.0},
        wall_limit_hit=False,
        first_elapsed=0.1,
        last_value=None,
        failure=None,
    )
    rec = r.as_record(n_trials=3)
    assert rec["mean"] == pytest.approx(0.2, abs=1e-6)
    assert rec["std"] > 0
    assert rec["vram_peak_mib"] == 1024.0
    assert rec["ram_peak_mib"] == 500.0
    assert rec["n_trials"] == 3
    assert "status" not in rec


def test_as_record_success_single_trial_std_zero():
    """Single-trial success reports std == 0 (no spread to measure)."""
    r = TimedResult(
        times=[0.5],
        mem={},
        wall_limit_hit=False,
        first_elapsed=0.5,
        last_value=None,
        failure=None,
    )
    rec = r.as_record()
    assert rec["mean"] == pytest.approx(0.5)
    assert rec["std"] == 0.0


def test_as_record_failure_propagates_failure_dict_and_mem():
    """On failure, as_record includes status='failed' plus the failure dict
    and the memory dict — useful for downstream display.
    """
    r = TimedResult(
        times=[],
        mem={"vram_peak_mib": 256.0},
        wall_limit_hit=False,
        first_elapsed=None,
        last_value=None,
        failure={
            "failure_type": "OOM",
            "exc_type": "RuntimeError",
            "exc_msg": "boom",
            "traceback": "...",
        },
    )
    rec = r.as_record(custom_key="extra")
    assert rec["status"] == "failed"
    assert rec["failure_type"] == "OOM"
    assert rec["exc_type"] == "RuntimeError"
    assert rec["exc_msg"] == "boom"
    assert rec["vram_peak_mib"] == 256.0
    assert rec["custom_key"] == "extra"
    # On failure, no mean/std is reported.
    assert "mean" not in rec
    assert "std" not in rec


# ── run_timed_trials ──────────────────────────────────────────────────────────


def test_run_timed_trials_happy_path():
    """n_trials timed calls + one discarded warmup."""
    calls = []

    def fn():
        calls.append(time.perf_counter())
        return 42

    result = run_timed_trials(fn, n_trials=3, wall_limit_s=10.0)

    assert result.succeeded
    assert len(result.times) == 3
    # One warmup + three trials = 4 total calls
    assert len(calls) == 4
    assert result.first_elapsed is not None
    assert result.first_elapsed >= 0
    assert result.wall_limit_hit is False
    # capture_value defaults to False — last_value should be None even on success
    assert result.last_value is None


def test_run_timed_trials_capture_value_returns_last_trial_value():
    """capture_value=True saves the last trial's return value (not warmup's)."""
    values = iter(
        [
            "warmup",  # discarded
            "trial-1",
            "trial-2",
            "trial-3",
        ]
    )

    def fn():
        return next(values)

    result = run_timed_trials(fn, n_trials=3, wall_limit_s=10.0, capture_value=True)
    assert result.succeeded
    assert result.last_value == "trial-3"


def test_run_timed_trials_warmup_failure_leaves_times_empty():
    """If warmup raises, the trial loop never runs and times is empty."""

    def fn():
        raise RuntimeError("boom during warmup")

    result = run_timed_trials(fn, n_trials=3, wall_limit_s=10.0)

    assert not result.succeeded
    assert result.times == []
    assert result.first_elapsed is None
    assert result.failure["exc_type"] == "RuntimeError"
    assert result.failure["failure_type"] == "error"
    assert "boom during warmup" in result.failure["exc_msg"]
    assert "Traceback" in result.failure["traceback"]


def test_run_timed_trials_trial_failure_keeps_earlier_times():
    """If trial 2 raises, trial 1's time is still recorded."""
    state = {"i": 0}

    def fn():
        state["i"] += 1
        # Sequence: 1=warmup, 2=trial-0 (ok), 3=trial-1 (raise)
        if state["i"] == 3:
            raise RuntimeError("CUDA_ERROR_OUT_OF_MEMORY at trial 2")
        return None

    result = run_timed_trials(fn, n_trials=3, wall_limit_s=10.0)

    assert not result.succeeded
    assert len(result.times) == 1  # only the first trial completed
    assert result.first_elapsed is not None
    assert result.failure["failure_type"] == "OOM"


def test_run_timed_trials_wall_limit_short_circuits():
    """If the first trial exceeds wall_limit_s, the loop stops immediately."""

    def fn():
        time.sleep(0.02)

    result = run_timed_trials(fn, n_trials=5, wall_limit_s=0.001)

    assert result.succeeded
    assert result.wall_limit_hit is True
    # Even though n_trials=5 was requested, only the first trial ran
    assert len(result.times) == 1


def test_run_timed_trials_wall_limit_not_triggered_under_limit():
    """If the first trial is fast enough, the loop runs all n_trials."""

    def fn():
        pass

    result = run_timed_trials(fn, n_trials=4, wall_limit_s=10.0)
    assert result.succeeded
    assert result.wall_limit_hit is False
    assert len(result.times) == 4


def test_run_timed_trials_records_first_elapsed_only_on_first_trial():
    """first_elapsed reflects trial 0, not trial n-1."""
    durations = iter([0.0, 0.05, 0.0, 0.0])  # warmup, trial-0, trial-1, trial-2

    def fn():
        time.sleep(next(durations))

    result = run_timed_trials(fn, n_trials=3, wall_limit_s=10.0)

    assert result.succeeded
    assert result.first_elapsed == result.times[0]
    # Trial 0 sleeps 50ms, trials 1-2 sleep 0 — first_elapsed should be the slowest
    assert result.first_elapsed >= 0.04


def test_run_timed_trials_timeout_classification():
    """A TimeoutError raised mid-trial is classified as 'timeout'."""

    def fn():
        raise TimeoutError("request exceeded deadline")

    result = run_timed_trials(fn, n_trials=2, wall_limit_s=10.0)
    assert not result.succeeded
    assert result.failure["failure_type"] == "timeout"


def test_run_timed_trials_container_died_classification():
    """An exception named ContainerDied is classified as 'container_died'."""

    class ContainerDied(Exception):
        pass

    def fn():
        raise ContainerDied("solver crashed")

    result = run_timed_trials(fn, n_trials=2, wall_limit_s=10.0)
    assert not result.succeeded
    assert result.failure["failure_type"] == "container_died"
