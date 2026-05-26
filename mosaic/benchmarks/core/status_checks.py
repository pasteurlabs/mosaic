"""Explicit, extensible status checks for per-experiment classification.

A status check is a callable ``(summary) -> tuple[str, str] | None`` where
``summary`` is a per-suite dataclass carrying the metrics the check needs.
The return is either ``None`` (check doesn't fire / doesn't apply) or
``(status, reason)`` with ``status == "anomaly"``. The classifier walks
the check list in order; the first ``anomaly`` wins.

Built-in checks for each suite are factory functions that capture their
threshold and return a callable matching the suite's summary type.
User-defined checks can be plain functions with the same signature — the
classifier doesn't care whether a check came from a factory or was
hand-written, only that its signature accepts the summary. Add a new check
by writing it next to the ``.add_experiment(status_check=[...])`` call —
no core-code edits needed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# A check returns (status, reason) on anomaly, or None on pass / not-applicable.
CheckOutcome = tuple[str, str] | None


# ── Per-suite summary dataclasses ────────────────────────────────────────────
# Each suite's classifier builds one of these per solver, then walks the
# experiment's check list passing the summary in. The fields below are the
# stable contract that user-defined checks can rely on.


@dataclass
class ForwardSummary:
    """Per-solver summary handed to forward-suite checks.

    ``errs_by_pval`` and ``peer_medians_by_pval`` are aligned: the keys are
    sweep-parameter values (e.g. ``"N"`` values, ``"nu"`` values). Entries
    only appear for sweep points where the solver produced a valid result.
    """

    errs_by_pval: dict[Any, float] = field(default_factory=dict)
    peer_medians_by_pval: dict[Any, float] = field(default_factory=dict)
    n_valid_points: int = 0


@dataclass
class CostSummary:
    """Per-solver summary handed to cost-suite checks."""

    solver_median_time: float | None = None
    peer_median_time: float | None = None


@dataclass
class FdCheckSummary:
    """Per-solver summary handed to gradient/fd_check checks.

    ``best_cosine``: max cosine across ε values for this solver.
    ``best_rel_err``: min (across ε) of the median-across-FD-directions rel error.
    ``peer_rel_err_median``: peer median of ``best_rel_err`` across solvers; ``None``
    when fewer than 3 peers reported a valid value.
    """

    best_cosine: float | None = None
    best_rel_err: float | None = None
    peer_rel_err_median: float | None = None


@dataclass
class OptimizationSummary:
    """Per-solver summary handed to optimization-suite checks.

    ``final_initial_ratio``: ``loss_final / loss_initial`` for the
    worst-case (highest-initial-loss) trajectory. A solver that didn't
    reduce loss has ratio ≥ 1.

    ``peer_final_loss_by_sweep``: per-sweep-value ratio of this solver's
    final loss to the best (minimum) final loss across all peers at the
    same sweep value. Empty when the experiment isn't a numeric sweep.
    """

    final_initial_ratio: float | None = None
    peer_final_loss_by_sweep: dict[Any, float] = field(default_factory=dict)


# ── Built-in check factories ─────────────────────────────────────────────────


def median_k(k: float) -> Callable[[ForwardSummary], CheckOutcome]:
    """Anomaly if the solver's error exceeds ``k × peer-median`` on at least
    half of the sweep points where it produced valid results."""

    def _check(s: ForwardSummary) -> CheckOutcome:
        bad: list[tuple[Any, float, float]] = []
        for pval, err in s.errs_by_pval.items():
            med = s.peer_medians_by_pval.get(pval, 0.0)
            if med > 0 and err > k * med:
                bad.append((pval, err, med))
        if not bad or len(bad) < max(1, s.n_valid_points // 2):
            return None
        worst = max(bad, key=lambda t: t[1] / max(t[2], 1e-300))
        ratio = worst[1] / max(worst[2], 1e-300)
        return (
            "anomaly",
            f"error {worst[1]:.3g} at sweep={worst[0]} is {ratio:.1f}× peer median ({worst[2]:.3g}); "
            f"threshold k={k}",
        )

    return _check


def max_error(threshold: float) -> Callable[[ForwardSummary], CheckOutcome]:
    """Anomaly if any sweep point's error exceeds ``threshold`` (absolute)."""

    def _check(s: ForwardSummary) -> CheckOutcome:
        bad = [(p, e) for p, e in s.errs_by_pval.items() if e > threshold]
        if not bad:
            return None
        worst = max(bad, key=lambda t: t[1])
        return (
            "anomaly",
            f"error {worst[1]:.3g} at sweep={worst[0]} > max_error={threshold}",
        )

    return _check


def max_peer_k(k: float) -> Callable[[CostSummary], CheckOutcome]:
    """Cost suite: anomaly if median wall-clock time > ``k × peer median``."""

    def _check(s: CostSummary) -> CheckOutcome:
        if s.solver_median_time is None or s.peer_median_time is None:
            return None
        if s.solver_median_time <= k * s.peer_median_time:
            return None
        ratio = s.solver_median_time / s.peer_median_time
        return (
            "anomaly",
            f"median time {s.solver_median_time:.1f}s is {ratio:.0f}× peer median "
            f"({s.peer_median_time:.2f}s); threshold k={k}",
        )

    return _check


def min_cosine(threshold: float) -> Callable[[FdCheckSummary], CheckOutcome]:
    """Gradient fd_check: anomaly if best-ε FD cosine < ``threshold``."""

    def _check(s: FdCheckSummary) -> CheckOutcome:
        if s.best_cosine is None or s.best_cosine >= threshold:
            return None
        return ("anom", f"best FD cosine {s.best_cosine:.4f} < {threshold}")

    return _check


def max_rel_err(threshold: float) -> Callable[[FdCheckSummary], CheckOutcome]:
    """Gradient fd_check: anomaly if best-ε median rel_error > ``threshold``."""

    def _check(s: FdCheckSummary) -> CheckOutcome:
        if s.best_rel_err is None or s.best_rel_err <= threshold:
            return None
        return (
            "anomaly",
            f"best-ε median FD rel_err {s.best_rel_err:.2e} > max_rel_err={threshold:.0e}",
        )

    return _check


def rel_err_peer_outlier(k: float) -> Callable[[FdCheckSummary], CheckOutcome]:
    """Gradient fd_check: anomaly if best-ε rel_err > ``k × peer median``
    (requires ≥3 valid peers; otherwise the check is skipped)."""

    def _check(s: FdCheckSummary) -> CheckOutcome:
        if s.best_rel_err is None or s.peer_rel_err_median is None:
            return None
        if s.peer_rel_err_median <= 0:
            return None
        if s.best_rel_err <= k * s.peer_rel_err_median:
            return None
        ratio = s.best_rel_err / s.peer_rel_err_median
        return (
            "anomaly",
            f"best-ε rel_err {s.best_rel_err:.2e} is {ratio:.1f}× peer median "
            f"({s.peer_rel_err_median:.2e}); threshold k={k}",
        )

    return _check


def max_final_ratio(threshold: float) -> Callable[[OptimizationSummary], CheckOutcome]:
    """Optimization suite: anomaly if final/initial loss ratio > ``threshold``
    (i.e. the optimiser didn't reduce loss by at least ``1 - threshold``)."""

    def _check(s: OptimizationSummary) -> CheckOutcome:
        if s.final_initial_ratio is None or s.final_initial_ratio <= threshold:
            return None
        return (
            "anomaly",
            f"final/initial = {s.final_initial_ratio:.2f} (> {threshold})",
        )

    return _check


def peer_final_loss_k(k: float) -> Callable[[OptimizationSummary], CheckOutcome]:
    """Optimization suite (sweeps only): anomaly if at any sweep value this
    solver's final loss is more than ``k×`` the best peer final loss."""

    def _check(s: OptimizationSummary) -> CheckOutcome:
        if not s.peer_final_loss_by_sweep:
            return None
        worst = max(s.peer_final_loss_by_sweep.items(), key=lambda kv: kv[1])
        sweep_k, ratio = worst
        if ratio <= k:
            return None
        return (
            "anomaly",
            f"final_loss at sweep={sweep_k} is {ratio:.1f}× best peer (threshold {k}×)",
        )

    return _check


def normalize(checks: list[Callable] | Callable | None) -> list[Callable]:
    """Coerce ``None`` / single callable to the canonical list-of-callables form."""
    if not checks:
        return []
    if callable(checks):
        return [checks]
    if isinstance(checks, list):
        return list(checks)
    raise TypeError(
        f"status_check must be a list of callables; got {type(checks).__name__}"
    )
