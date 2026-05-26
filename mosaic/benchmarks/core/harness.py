# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable timing harness for benchmark suites.

Centralises the warmup + repeated-trial pattern that appears in every
timing suite (cost, gradient, optimization) together with peak-VRAM/RAM
sampling and exception classification.

The harness is **solver- and problem-agnostic**: callers pass a thunk and
get back a :class:`TimedResult` describing what happened. All
problem-specific input construction and result post-processing stays in
the suite.

Typical use::

    result = run_timed_trials(
        lambda: t.apply(inputs),
        n_trials=3,
        wall_limit_s=1000.0,
        gpu_id=gpu_id,
        image_tag=image_tag,
    )
    if result.failure is not None:
        ...  # record failure, stop sweep
    record = result.as_record()  # {"mean", "std", "vram_peak_mib", ...}
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from mosaic.benchmarks.core.hardware import ResourceSampler


def classify_failure(exc_name: str, exc_str: str) -> str:
    """Map an exception type+message to a short failure-type label.

    Returns one of: ``container_died``, ``OOM``, ``timeout``, ``nan``, ``error``.
    """
    s = exc_str.lower()
    if exc_name == "ContainerDied":
        return "container_died"
    if (
        "resource_exhausted" in s
        or "out of memory" in s
        or "cuda_error_out_of_memory" in s
    ):
        return "OOM"
    if (
        exc_name
        in (
            "WatchdogTimeout",
            "WatchdogError",
            "TimeoutError",
            "ReadTimeout",
            "ConnectTimeout",
        )
        or "timeout" in exc_name.lower()
    ):
        return "timeout"
    if "nan" in s or "not finite" in s:
        return "nan"
    return "error"


@dataclass
class TimedResult:
    """Outcome of :func:`run_timed_trials`.

    Attributes:
        times: Per-trial elapsed seconds. Empty if the warmup raised before
            any trial ran.
        mem: ``{"vram_peak_mib", "ram_peak_mib"}`` from the resource sampler.
        wall_limit_hit: ``True`` iff the first trial exceeded ``wall_limit_s``
            and the loop stopped early.
        first_elapsed: Elapsed seconds for the first trial, or ``None`` if no
            trial completed.
        last_value: Last return value of ``fn`` from inside the trial loop
            (warmup return value is discarded). Only populated when
            ``capture_value=True``; otherwise ``None``.
        failure: ``None`` on success; otherwise
            ``{"failure_type", "exc_type", "exc_msg", "traceback"}``.
    """

    times: list[float]
    mem: dict
    wall_limit_hit: bool
    first_elapsed: float | None
    last_value: Any | None
    failure: dict | None

    @property
    def succeeded(self) -> bool:
        """True if the timed run produced no failure."""
        return self.failure is None

    def as_record(self, **extra: object) -> dict:
        """Return a dict suitable for inserting into a suite result.

        On success: ``{"mean", "std", "trials_s", **mem, **extra}``.
        On failure: ``{"status": "failed", "trials_s", **failure, **mem, **extra}``.

        ``trials_s`` is the raw list of per-trial elapsed seconds (in trial
        order). It's emitted unconditionally so downstream tooling can
        recompute distributional statistics (min, median, percentiles, …)
        without re-running the benchmark. On warmup-time failure the list
        is empty; on mid-trial failure it contains every successful trial
        before the exception.
        """
        trials = [float(t) for t in self.times]
        if self.failure is not None:
            return {
                "status": "failed",
                "trials_s": trials,
                **self.failure,
                **self.mem,
                **extra,
            }
        return {
            "mean": float(jnp.mean(jnp.array(self.times))),
            "std": float(jnp.std(jnp.array(self.times)))
            if len(self.times) > 1
            else 0.0,
            "trials_s": trials,
            **self.mem,
            **extra,
        }


def run_timed_trials(
    fn: Callable[[], Any],
    *,
    n_trials: int,
    wall_limit_s: float,
    gpu_id: int | None = None,
    image_tag: str | None = None,
    capture_value: bool = False,
) -> TimedResult:
    """Run ``fn`` n_trials times under a :class:`ResourceSampler`.

    Behaviour:
      * One unreported warmup ``fn()`` call is performed first to absorb
        per-solver JIT compilation, first-touch CUDA kernel caching, and
        scan-unroll tracing.
      * ``n_trials`` timed calls follow, each bracketed by ``perf_counter``.
      * If the first timed trial exceeds ``wall_limit_s``, the loop stops
        immediately and ``wall_limit_hit`` is set on the result.
      * If ``fn`` raises (during warmup or any trial), ``failure`` is
        populated via :func:`classify_failure` and timing stops where it
        was. The sampler context unwinds normally so peak memory is still
        reported.
      * ``capture_value=True`` stores the last successful ``fn()`` return
        value from inside the trial loop (useful for grabbing a gradient
        for ``grad_norm`` / snapshotting).
    """
    sampler = ResourceSampler(gpu_id=gpu_id, image_tag=image_tag)
    times: list[float] = []
    last_value: Any = None
    wall_limit_hit = False
    first_elapsed: float | None = None
    failure: dict | None = None
    try:
        with sampler:
            fn()  # warmup; return value intentionally discarded
            for i in range(n_trials):
                t0 = time.perf_counter()
                v = fn()
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                if i == 0:
                    first_elapsed = elapsed
                if capture_value:
                    last_value = v
                if i == 0 and elapsed > wall_limit_s:
                    wall_limit_hit = True
                    break
    except Exception as exc:
        failure = {
            "failure_type": classify_failure(type(exc).__name__, str(exc)),
            "exc_type": type(exc).__name__,
            "exc_msg": str(exc),
            "traceback": traceback.format_exc(),
        }
    mem = sampler.summary
    return TimedResult(
        times=times,
        mem=mem,
        wall_limit_hit=wall_limit_hit,
        first_elapsed=first_elapsed,
        last_value=last_value,
        failure=failure,
    )
