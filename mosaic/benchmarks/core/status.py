# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment-completion status discovery for the `mosaic status` CLI.

Walks ``<results_dir>/<problem>/<suite>/<experiment>/`` on disk, parses
each ``result.json``, and classifies every (experiment × solver) cell as one
of:

    "ok"        – solver produced valid data
    "anomaly"   – valid but suspiciously bad (outlier vs. peers, misaligned
                  gradient, or diverged optimisation)
    "failed"    – solver was attempted but its entry is empty / invalid / NaN
    "not_run"   – no result file, or solver absent from the parsed result
    "excluded"  – solver is excluded for this (suite, experiment) via
                  ``SolverSpec.exclusions``. Exclusion keys may be suite-level
                  (``"gradient"``), experiment-level (``"drag_opt"``), or
                  fully-qualified (``"recovery/drag_opt"``).

Cell status is computed from whichever of the three canonical result layouts
is present: ``by_solver[solver]``, ``by_param[value][solver]`` (forward suite),
or ``by_N[solver][N]`` / ``by_steps`` (cost suite). When a failure carries a
human-readable reason (``error`` field, ``"status": "error"``), the reason is
surfaced alongside the cell.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    EXCL_PERMANENT,
    Exclusion,
    ExclusionCategory,
    Problem,
)
from .io import (
    harness_fn_hash,
    load_json,
    results_dir,
    tesseract_content_hash,
)
from .utils import exclusion_lookup

# Suites visited by the status command. "ics" produces no per-solver results.
SUITES: tuple[str, ...] = ("forward", "cost", "gradient", "optimization")

# Cell status states.
OK = "ok"
ANOMALY = "anomaly"
FAILED = "failed"
NOT_RUN = "not_run"
EXCLUDED = "excluded"

# Exclusion categories carried on EXCLUDED cells live on
# :class:`mosaic.benchmarks.core.config.ExclusionCategory`. Only
# ``CATEGORICAL`` is *permanent* (out of the score denominator); everything
# else is "work to do" at the neutral weight. ``Cell.category`` stores the
# raw string value (str-Enum), so existing comparisons against ``"categorical"``
# / ``"explained"`` etc. continue to work unchanged.

# Permanent categories as a set of raw strings (for ``cell.category in …``
# checks where the cell's category is a plain string).
EXCL_PERMANENT_VALUES: frozenset[str] = frozenset(c.value for c in EXCL_PERMANENT)


# ── weighted campaign-health score ──────────────────────────────────────────
#
# A single scalar in [0.0, +1.0] summarising campaign state.  Each
# non-categorical cell contributes its weight; categorical exclusions are
# excluded from both numerator and denominator.
#
# Staleness (the ``*`` marker) does not affect the weight: it's a "re-run
# recommended" hint for humans, not a health signal, and folding it into the
# score made the headline number move on the incremental PR-preview path for
# reasons a PR author couldn't see or fix (baseline cells carried over from
# ``main`` hash-mismatch the current tree and get flagged even when the PR
# didn't touch them). Releases always re-run the full suite from scratch, so
# nothing is ever legitimately stale there. Keeping stale weights equal to
# their fresh counterparts preserves the invariant "no status change ⇒ no
# score change" that reviewers expect. The ``*`` glyph is still rendered.
SCORE_WEIGHTS: dict[str, float] = {
    "ok": 1.00,
    "anom": 0.53,
    "missing": 0.33,
    "excl": 0.33,  # all non-categorical exclusions
    "fail": 0.00,
    # "perm" (EXCLUDED + categorical) is excluded from the denominator.
}


def cell_weight_key(cell: Cell) -> str | None:
    """Return the SCORE_WEIGHTS key for a cell, or None if categorical.

    Categorical (permanent) exclusions return None — the caller should skip
    them entirely (no numerator contribution, not counted in denominator).

    Staleness is intentionally ignored: a stale cell scores identically to
    its fresh counterpart (see :data:`SCORE_WEIGHTS`).
    """
    if cell.status == OK:
        return "ok"
    if cell.status == ANOMALY:
        return "anom"
    if cell.status == FAILED:
        return "fail"
    if cell.status == NOT_RUN:
        return "missing"
    if cell.status == EXCLUDED:
        if cell.category in EXCL_PERMANENT:
            return None
        return "excl"
    return None


def compute_score(cells: list[Cell]) -> tuple[float | None, int]:
    """Weighted campaign-health score over a list of cells.

    Returns ``(score, n_contributing)``. ``score`` is ``None`` when no cell
    contributes (all categorical / empty input) — callers should treat that
    as "no signal" rather than as 0.0, which is a real data point meaning
    "all work-to-do, no progress".

    Range: ``[0.0, +1.0]`` — fail=0, neutral (missing/todo/…)=0.33, ok=1.0.
    """
    total = 0.0
    n = 0
    for cell in cells:
        key = cell_weight_key(cell)
        if key is None:
            continue
        total += SCORE_WEIGHTS.get(key, 0.0)
        n += 1
    if n == 0:
        return None, 0
    return total / n, n


def _lookup_check(cfg: Problem, suite: str, experiment: str) -> list:
    """Return the merged list of check callables for (suite, experiment).

    Sources are accumulated in order of increasing specificity; suite-level
    defaults come first so per-experiment checks can override them by
    short-circuiting (the classifier walks the list and stops on the first
    ``anom``, so more-specific entries should appear *later* — placing them
    at the tail ensures they're consulted last and thus have the final say
    only when earlier checks pass).

    Sources:
      1. ``cfg.status_checks[suite]`` — suite-level defaults
      2. ``cfg.status_checks[<suite>/<experiment>]`` /
         ``cfg.status_checks[<suite>/<leading>]`` — per-experiment / per-IC
         overrides from the Problem-level dict
      3. ``cfg.experiments[full].params["status_check"]`` — inline overrides
         set on the ``.add_experiment(..., status_check=[...])`` call

    Each source is a list of callables (or a single callable); both shapes
    are normalised via :func:`status_checks.normalize`.
    """
    from .status_checks import normalize

    checks = cfg.status_checks
    merged: list = []
    merged.extend(normalize(checks.get(suite)))
    # experiment labels may include an IC sub-dir (e.g. "agreement/tgv");
    # match both the full label and the leading token.
    for key in (f"{suite}/{experiment}", f"{suite}/{experiment.split('/', 1)[0]}"):
        merged.extend(normalize(checks.get(key)))
        exp = cfg.experiments.get(key)
        if exp is not None:
            inline = (
                exp.params.get("status_check") if isinstance(exp.params, dict) else None
            )
            merged.extend(normalize(inline))
    return merged


@dataclass
class Cell:
    """A single (experiment × solver) status cell."""

    status: str
    reason: str = ""
    # Only populated when status == EXCLUDED. One of the EXCL_* constants.
    category: str = ""
    # True when the result that produced this cell predates the current
    # tesseract/harness source — a re-run is needed. Rendered as a trailing
    # `*` on the cell's glyph (e.g. `ok*`, `anom*`), never replaces the
    # underlying status.
    stale: bool = False
    # Resource frontier for swept experiments: the first sweep value at which
    # the solver hit a *resource ceiling* (OOM / timeout). ``None`` when the
    # solver ran to completion at every value. A ceiling is NOT a failure — a
    # solver that OOMs only at the largest sizes is still OK — but the frontier
    # is carried into the snapshot so ``diff_snapshots`` can tell when the
    # ceiling *moves* (OOM starting earlier than baseline = regression). This
    # is the platform-dependent case that can't be encoded as an exclusion.
    resource_frontier: str = ""
    # Numeric scalars for this cell, keyed by a stable metric name (see
    # ``METRIC_SPECS``). Diffed across snapshots by ``_diff_metrics`` so a cell
    # that stays ``ok`` still surfaces a numeric regression / improvement — the
    # status ladder alone hides most of what optimisation PRs actually move.
    # Empty when the suite has no comparable scalar or the solver didn't run.
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentRow:
    """One row in a :class:`ProblemStatus` table (one suite × experiment pair)."""

    suite: str
    experiment: str  # may be "<name>" or "<name>/<ic_name>" for IC-sub-dirs
    result_path: Path | None  # None when result.json is missing
    cells: dict[str, Cell] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Return the fully-qualified ``<suite>/<experiment>`` label."""
        return f"{self.suite}/{self.experiment}"


@dataclass
class ProblemStatus:
    """Status summary for a single benchmark problem."""

    problem: str
    solvers: list[str]
    rows: list[ExperimentRow]


# ── result-file parsing ──────────────────────────────────────────────────────


def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def _has_any_finite(obj: Any) -> bool:
    """Return True if obj contains at least one finite numeric value."""
    if isinstance(obj, bool):
        return False
    if isinstance(obj, int | float):
        return not _is_nan(obj) and math.isfinite(obj)
    if isinstance(obj, dict):
        return any(_has_any_finite(v) for v in obj.values())
    if isinstance(obj, list | tuple):
        return any(_has_any_finite(v) for v in obj)
    return False


def _reason_from_entry(entry: Any) -> str:
    """Extract a short human-readable failure reason from a result entry."""
    if isinstance(entry, dict):
        for key in ("error", "message", "reason"):
            v = entry.get(key)
            if isinstance(v, str) and v:
                return v.strip().splitlines()[0][:160]
    return ""


def _run_checks(checks: list, summary: Any) -> tuple[str, str] | None:
    """Walk a check list against a summary; return first ``anom`` or ``None``.

    Checks whose signature doesn't match the summary type (e.g. a gradient
    check passed to a forward classifier) raise ``TypeError`` /
    ``AttributeError`` when reading absent fields — we swallow those so a
    suite-level check list can mix entries that apply to different suites
    without crashing the classifier.
    """
    for check in checks:
        try:
            result = check(summary)
        except (AttributeError, TypeError):
            continue
        if result and result[0] == ANOMALY:
            return result
    return None


def _median(values: list[float]) -> float:
    vs = sorted(values)
    mid = len(vs) // 2
    return vs[mid] if len(vs) % 2 else 0.5 * (vs[mid - 1] + vs[mid])


def _refine_cost(data: dict, cells: dict[str, Cell], checks: list) -> None:
    """Walk ``checks`` against per-solver :class:`CostSummary` instances.

    Solver-medians and peer-median are computed once across OK solvers (need
    ≥2 to form a meaningful peer comparison), then each check function is
    given a CostSummary per solver. First anomaly wins.
    """
    from .status_checks import CostSummary

    if not checks:
        return
    key = "by_N" if "by_N" in data else "by_steps" if "by_steps" in data else None
    if not key:
        return
    top = data[key]
    solver_medians: dict[str, float] = {}
    for solver, cell in cells.items():
        if cell.status != OK:
            continue
        vals = top.get(solver, {})
        times = [
            v["mean"]
            for v in vals.values()
            if isinstance(v, dict) and math.isfinite(v.get("mean", float("nan")))
        ]
        if times:
            solver_medians[solver] = _median(times)
    if len(solver_medians) < 2:
        return
    peer_median = _median(list(solver_medians.values()))
    if peer_median <= 0:
        return
    for solver, med in solver_medians.items():
        summary = CostSummary(solver_median_time=med, peer_median_time=peer_median)
        verdict = _run_checks(checks, summary)
        if verdict:
            cells[solver] = Cell(*verdict)


def _refine_fd_check(data: dict, cells: dict[str, Cell], checks: list) -> None:
    """Anomaly checks for fd_check / source_fd_check.

    For each solver we compute ``best_rel`` = the minimum across ε of the
    *median-across-directions* rel_error. "Median across directions" is more
    honest than min-across-directions: min cherry-picks the one lucky FD
    direction that happened to align with the reverse-mode gradient, and
    hides systematic backward-magnitude error that affects most directions.

    Reads from *checks*:
      - ``min_cosine``        anomaly if best-ε cosine < this (direction)
      - ``max_rel_err``       anomaly if best-ε median rel_error > this
                              absolute threshold
      - ``rel_err_peer_k``    anomaly if best-ε median rel_error > K × peer
                              median of that same metric (relative outlier;
                              requires ≥3 valid peers)

    Absent keys skip that check.
    """
    from .status_checks import FdCheckSummary

    if not checks:
        return

    by_solver = data.get("by_solver", {})
    if not isinstance(by_solver, dict):
        return

    stats_per_solver: dict[str, tuple[float | None, float | None]] = {}
    for solver, entry in by_solver.items():
        sweep = entry.get("eps_sweep") if isinstance(entry, dict) else None
        if not isinstance(sweep, dict):
            stats_per_solver[solver] = (None, None)
            continue
        best_cos = None
        best_rel = None
        for st in sweep.values():
            if not isinstance(st, dict):
                continue
            c = st.get("cosine")
            if isinstance(c, int | float) and math.isfinite(c):
                best_cos = c if best_cos is None else max(best_cos, c)
            vals = [
                r
                for r in (st.get("rel_error") or [])
                if isinstance(r, int | float) and math.isfinite(r)
            ]
            if not vals:
                continue
            med = _median(vals)
            best_rel = med if best_rel is None else min(best_rel, med)
        stats_per_solver[solver] = (best_cos, best_rel)

    peer_rels = sorted(v for _, v in stats_per_solver.values() if v is not None)
    peer_median = _median(peer_rels) if len(peer_rels) >= 3 else None

    for solver, (best_cos, best_rel) in stats_per_solver.items():
        if solver not in cells or cells[solver].status != OK:
            continue
        summary = FdCheckSummary(
            best_cosine=best_cos,
            best_rel_err=best_rel,
            peer_rel_err_median=peer_median,
        )
        verdict = _run_checks(checks, summary)
        if verdict:
            cells[solver] = Cell(*verdict)


def _is_sweep_key(k: Any) -> bool:
    """True for keys that look like numeric sweep values (int/float, or a string that parses as float)."""
    if isinstance(k, int | float):
        return True
    if isinstance(k, str):
        try:
            float(k)
            return True
        except (ValueError, TypeError):
            return False
    return False


def _sweep_sub_entry(entry: dict, sweep_k: str) -> Any:
    """Look up a sweep sub-entry by string key, falling back to a float key when the string parses as a plain number."""
    sub = entry.get(sweep_k)
    if sub is not None:
        return sub
    if sweep_k.replace(".", "").lstrip("-").isdigit():
        return entry.get(float(sweep_k))
    return entry.get(sweep_k)


def _collect_sweep_keys(top: dict) -> set[str]:
    """Return the union of numeric-style keys across all solver entries."""
    keys: set[str] = set()
    for entry in top.values():
        if not isinstance(entry, dict):
            continue
        for k in entry:
            if _is_sweep_key(k):
                keys.add(str(k))
    return keys


def _peer_finals_at(top: dict, sweep_k: str) -> dict[str, float]:
    """Gather non-trivial, finite final_loss values across all solvers at sweep value *sweep_k*.

    Trivial points (initial_loss <= 0) are skipped.
    """
    peer_finals: dict[str, float] = {}
    for solver, entry in top.items():
        if not isinstance(entry, dict):
            continue
        sub = _sweep_sub_entry(entry, sweep_k)
        if not isinstance(sub, dict):
            continue
        fl = sub.get("final_loss")
        il = sub.get("initial_loss", 0.0)
        if (
            isinstance(fl, int | float)
            and math.isfinite(fl)
            and fl >= 0
            and isinstance(il, int | float)
            and float(il) > 0
        ):
            peer_finals[solver] = float(fl)
    return peer_finals


def _worst_case_trajectory(entry: dict) -> list[float] | None:
    """Pick the worst-case (highest initial loss) trajectory from a numeric-sweep dict.

    Falls back to a direct trajectory lookup when the entry isn't numeric-keyed.

    Without this, ``_find_trajectory`` would return the all-zero trajectory
    of a trivial sweep value first and the caller would bail on
    initial<=0 — masking non-trivial test points entirely.
    """
    numeric_subs = {
        k: v for k, v in entry.items() if _is_sweep_key(k) and isinstance(v, dict)
    }
    if not numeric_subs:
        return _find_trajectory(entry)
    best_init = -1.0
    series: list[float] | None = None
    for sub_v in numeric_subs.values():
        cand = _find_trajectory(sub_v)
        if not cand:
            continue
        cand_init = abs(cand[0])
        if cand_init > best_init:
            best_init = cand_init
            series = cand
    return series


def _refine_recovery(data: dict, cells: dict[str, Cell], checks: list) -> None:
    """Walk ``checks`` against per-solver :class:`OptimizationSummary` instances.

    Builds two metrics per solver: ``final_initial_ratio`` (from the
    worst-case trajectory) and ``peer_final_loss_by_sweep`` (per-sweep
    ratio to the best peer final loss). Then iterates each solver's check
    list — first anomaly wins.

    Note that peer values include categorically-excluded solvers since
    their loss values still represent self-consistent optimisations.
    """
    from .status_checks import OptimizationSummary

    if not checks:
        return
    top = data.get("by_solver") or data.get("by_sweep") or {}
    if not isinstance(top, dict):
        return

    # Per-sweep peer-min final losses (used for peer_final_loss_k checks).
    peer_min_by_sweep: dict[Any, float] = {}
    sweep_keys = _collect_sweep_keys(top)
    for sweep_k in sweep_keys:
        peer_finals = _peer_finals_at(top, sweep_k)
        if len(peer_finals) >= 2:
            best = min(peer_finals.values())
            if best > 0:
                peer_min_by_sweep[sweep_k] = best

    for solver in list(cells):
        if cells[solver].status != OK:
            continue
        entry = top.get(solver)
        if not isinstance(entry, dict):
            continue
        # final_initial_ratio from the worst-case trajectory.
        series = _worst_case_trajectory(entry)
        ratio: float | None = None
        if series:
            initial = abs(series[0])
            final = abs(series[-1])
            if initial > 0 and math.isfinite(final):
                ratio = final / initial
        # Per-sweep ratios to peer-min final.
        per_sweep: dict[Any, float] = {}
        for sweep_k, peer_min in peer_min_by_sweep.items():
            sub = _sweep_sub_entry(entry, sweep_k)
            if not isinstance(sub, dict):
                continue
            fl = sub.get("final_loss")
            if isinstance(fl, int | float) and math.isfinite(fl):
                per_sweep[sweep_k] = float(fl) / peer_min
        summary = OptimizationSummary(
            final_initial_ratio=ratio,
            peer_final_loss_by_sweep=per_sweep,
        )
        verdict = _run_checks(checks, summary)
        if verdict:
            cells[solver] = Cell(*verdict)


def _find_trajectory(entry: Any) -> list[float] | None:
    """Return the first list of floats named "errors"/"drags"/"loss" in entry."""
    if not isinstance(entry, dict):
        return None
    for key in ("errors", "drags", "loss", "losses"):
        val = entry.get(key)
        if (
            isinstance(val, list)
            and len(val) >= 2
            and all(isinstance(v, int | float) for v in val)
        ):
            return [float(v) for v in val]
    # Nested (e.g. by_sweep[solver][sigma_val][errors]).
    for v in entry.values():
        nested = _find_trajectory(v)
        if nested:
            return nested
    return None


def _classify_from_v1(data: dict, solvers: list[str], checks: list) -> dict[str, Cell]:
    """Classify from schema_version=1 flat results list."""
    cells: dict[str, Cell] = {}
    results = data.get("results", [])
    if not results:
        return {s: Cell(NOT_RUN) for s in solvers}

    # Group by solver
    per_solver: dict[str, list[dict]] = {}
    for entry in results:
        solver = entry.get("solver", "")
        per_solver.setdefault(solver, []).append(entry)

    for solver in solvers:
        entries = per_solver.get(solver, [])
        if not entries:
            cells[solver] = Cell(NOT_RUN)
            continue

        # Walk each sweep point and classify it into one of these signals:
        #   crash          — ``status: "failed"`` with a failure_type that is a
        #                    real bug (error / nan / container_died, or an
        #                    unlabelled failure). A crash at ANY size fails the
        #                    whole cell — a larger value succeeding doesn't
        #                    excuse it (e.g. PR #132's non-power-of-two compile
        #                    error). The failure record carries finite mem
        #                    fields, so ``_has_any_finite`` alone would mis-read
        #                    it as OK — we must consult ``status`` explicitly.
        #   resource ceiling — ``status: "failed"`` with failure_type OOM /
        #                    timeout. This is a platform-dependent scaling
        #                    limit, expected at large sizes and not encodable as
        #                    an exclusion. It does NOT fail the cell; we only
        #                    record the frontier (first value that hit it) so
        #                    the diff can tell when the ceiling moves.
        #   valid          — real, finite result data.
        #   empty / skipped — ``{}`` (never ran) or a deliberately-skipped
        #                    point; benign "didn't run", ignored.
        #   valid: False   — the forward/gradient suites' invalid sentinel.
        has_valid = False
        has_any = False
        fail_reason = ""
        valid_false_reason = ""
        frontier: str | None = None
        _RESOURCE_CEILING = ("OOM", "timeout")
        for entry in entries:
            metrics = entry.get("metrics")
            if metrics is None:
                continue
            # A "skipped" point (a value the framework deliberately didn't run
            # after an earlier ceiling / wall-limit hit) is a benign "didn't
            # run" — treat it like an absent/empty cell.
            if isinstance(metrics, dict) and metrics.get("status") == "skipped":
                continue
            has_any = True
            if not isinstance(metrics, dict):
                continue
            if metrics.get("status") == "failed":
                if metrics.get("failure_type") in _RESOURCE_CEILING:
                    # Resource ceiling — not a failure. Record the first value
                    # at which it bit, in on-disk sweep order.
                    if frontier is None:
                        frontier = str(entry.get("sweep_value") or "")
                elif not fail_reason:
                    fail_reason = _fail_reason_from_metrics(metrics)
            elif metrics.get("valid") is True:
                has_valid = True
            elif metrics.get("valid") is False:
                if not valid_false_reason:
                    valid_false_reason = _reason_from_entry(metrics) or ""
            elif _has_any_finite(metrics):
                has_valid = True

        frontier = frontier or ""
        if fail_reason:
            cells[solver] = Cell(FAILED, fail_reason, resource_frontier=frontier)
        elif not has_any:
            cells[solver] = Cell(NOT_RUN)
        elif has_valid:
            cells[solver] = Cell(OK, resource_frontier=frontier)
        else:
            # No valid data and no real crash. If a resource ceiling was the
            # only outcome (OOM at every attempted size), the solver can't run
            # this experiment on this platform at all — surface it as FAILED so
            # it isn't a silent all-empty, but keep the frontier for the diff.
            if frontier:
                cells[solver] = Cell(
                    FAILED,
                    f"resource ceiling at all sizes (from {frontier})",
                    resource_frontier=frontier,
                )
            else:
                cells[solver] = Cell(FAILED, valid_false_reason or "all entries failed")

    return cells


def _fail_reason_from_metrics(metrics: dict) -> str:
    """Build a short human-readable reason from a ``status: "failed"`` record.

    Cost/timing failures carry ``failure_type`` + ``exc_type`` + ``exc_msg``
    (see :meth:`TimedResult.as_record`); prefer those over the generic
    ``error``/``message``/``reason`` fields the other suites use.
    """
    exc_type = metrics.get("exc_type")
    exc_msg = metrics.get("exc_msg") or metrics.get("error")
    failure_type = metrics.get("failure_type")
    if exc_type and exc_msg:
        head = str(exc_msg).strip().splitlines()[0][:160]
        prefix = f"{failure_type}: " if failure_type else ""
        return f"{prefix}{exc_type}: {head}"
    if failure_type:
        return str(failure_type)
    return _reason_from_entry(metrics) or "failed"


def _classify_result(data: dict, solvers: list[str], checks: list) -> dict[str, Cell]:
    """Classify each solver from a schema_version=1 result."""
    cells = _classify_from_v1(data, solvers, checks)

    # Solvers whose Tesseract container failed to start (or whose work raised)
    # before any result was recorded are absent from the data layout above and
    # would otherwise classify as NOT_RUN. The harness records them in
    # ``_solver_failures``; promote those to FAILED so broken containers are
    # surfaced rather than indistinguishable from "wasn't selected".
    failures = data.get("_solver_failures") or {}
    if isinstance(failures, dict):
        for solver, reason in failures.items():
            if solver in cells and cells[solver].status == NOT_RUN:
                cells[solver] = Cell(FAILED, str(reason) or "solver failed")
    return cells


# ── filesystem enumeration ───────────────────────────────────────────────────


def _iter_experiment_dirs(suite_dir: Path):
    """Yield (experiment_label, result_path) under ``suite_dir``.

    Skips ``*_debug`` experiments. When a direct ``result.json`` is absent,
    descends one level to pick up IC-sub-dir layouts like
    ``agreement/tgv/result.json`` and returns labels ``agreement/tgv``.
    """
    if not suite_dir.is_dir():
        return
    for exp_dir in sorted(suite_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name.endswith("_debug"):
            continue
        direct = exp_dir / "result.json"
        if direct.exists():
            yield exp_dir.name, direct
            continue
        any_sub = False
        for sub in sorted(exp_dir.iterdir()):
            if not sub.is_dir() or sub.name.endswith("_debug"):
                continue
            sub_result = sub / "result.json"
            if sub_result.exists():
                any_sub = True
                yield f"{exp_dir.name}/{sub.name}", sub_result
        if not any_sub:
            yield exp_dir.name, None


def _results_dir(cfg: Problem) -> Path:
    return results_dir() / cfg.name


def _resolve_harness_hash(qualname: str, cache: dict[str, str | None]) -> str | None:
    """Resolve ``module.qualname`` and hash via the AST-normalised harness_fn_hash.

    Must match the writer in ``save_experiment``. Returns ``None`` on any
    failure; results are memoised in *cache*.
    """
    if qualname in cache:
        return cache[qualname]
    try:
        module_path, _, attr_path = qualname.rpartition(".")
        # Handle nested qualnames (e.g. Class.method).
        if "." in attr_path or not module_path:
            # Walk the qualname against importable prefixes.
            parts = qualname.split(".")
            for i in range(len(parts) - 1, 0, -1):
                with contextlib.suppress(ImportError):
                    mod = importlib.import_module(".".join(parts[:i]))
                    with contextlib.suppress(AttributeError, OSError, TypeError):
                        target = functools.reduce(getattr, parts[i:], mod)
                        h = harness_fn_hash(target) or None
                        cache[qualname] = h
                        return h
            cache[qualname] = None
            return None
        mod = importlib.import_module(module_path)
        target = getattr(mod, attr_path)
        h = harness_fn_hash(target) or None
    except Exception:
        h = None
    cache[qualname] = h
    return h


def _resolve_tesseract_hash(cfg: Problem, solver: str, cache: dict[str, str]) -> str:
    """Hash the on-disk tesseract directory for *solver*; memoised in *cache*."""
    if solver in cache:
        return cache[solver]
    try:
        spec = cfg.solver(solver)
    except KeyError:
        cache[solver] = ""
        return ""
    tess_dir = cfg.tesseract_dir / spec.dir
    h = tesseract_content_hash(tess_dir) if tess_dir.is_dir() else ""
    cache[solver] = h
    return h


def _v1_to_legacy_view(data: dict) -> dict:
    """Build a legacy-shaped view from v1 results for anomaly refinement functions.

    Returns a dict with ``by_solver``, ``by_param``, or ``by_N``/``by_steps``
    depending on the experiment's sweep structure, so existing refinement
    functions can work without modification.
    """
    if data.get("schema_version") != 1:
        return data
    results = data.get("results", [])
    sweep_info = data.get("sweep")
    extras = data.get("extras", {})

    # Group by solver
    by_solver: dict = {}
    for entry in results:
        solver = entry.get("solver", "")
        sv = entry.get("sweep_value")
        metrics = entry.get("metrics")
        if sv is None:
            by_solver[solver] = metrics or {}
        else:
            by_solver.setdefault(solver, {})[sv] = metrics or {}

    view: dict = {"by_solver": by_solver, "params": data.get("params", {})}

    # Reconstruct sweep-specific keys for refinement functions
    if sweep_info and isinstance(sweep_info, dict):
        sweep_key = sweep_info.get("key", "")
        if sweep_key in ("N", "n_elements"):
            view["by_N"] = by_solver
        elif sweep_key == "steps":
            view["by_steps"] = by_solver

    # Forward extras into the view
    view.update(extras)
    return view


def _refine_for_suite(
    suite: str, exp_label: str, data: dict, cells: dict[str, Cell], checks: list
) -> None:
    """Dispatch suite-specific anomaly refinements (no-op without thresholds)."""
    view = _v1_to_legacy_view(data) if data.get("schema_version") == 1 else data
    if suite == "cost":
        _refine_cost(view, cells, checks)
    elif suite == "gradient" and exp_label.split("/")[0] in (
        "fd_check",
        "source_fd_check",
    ):
        _refine_fd_check(view, cells, checks)
    elif suite == "optimization":
        _refine_recovery(view, cells, checks)


# ── per-cell numeric metric collection ───────────────────────────────────────
#
# The status ladder (ok / anom / fail / …) is coarse: a solver that stays ``ok``
# while getting 2× slower or doubling its gradient error produces no transition.
# We therefore carry a small dict of numeric scalars per cell into the snapshot
# so ``diff_snapshots`` can surface a same-status numeric regression /
# improvement (see ``_diff_metrics``). These collectors run *unconditionally*
# from ``_build_row`` — independent of whether an anomaly check is configured —
# and compute the same scalars the ``_refine_*`` functions already use.


# Metric name → (relative-change threshold to surface, "lower"|"higher" is
# better, confidence). Wall-clock ("cost_time_s") is noisy across GitHub-hosted
# runners (shared-host contention) so it gets a wide threshold and an
# "indicative" confidence; the deterministic correctness metrics are compared
# firmly. NOTE: this firm/indicative split assumes every benchmark run uses one
# GH-hosted hardware class — revisit if that ever stops holding.
METRIC_SPECS: dict[str, tuple[float, str, str]] = {
    "rel_err": (0.15, "lower", "firm"),
    "cosine": (0.02, "higher", "firm"),
    "final_ratio": (0.15, "lower", "firm"),
    "final_loss": (0.15, "lower", "firm"),
    "fwd_error": (0.15, "lower", "firm"),
    "cost_time_s": (0.35, "lower", "indicative"),
}


def _store_metric(cell: Cell, key: str, value: float | None) -> None:
    """Record a finite metric scalar on a cell; skip None / NaN / inf."""
    if isinstance(value, int | float) and math.isfinite(value):
        cell.metrics[key] = float(value)


def _collect_cost_metrics(data: dict, cells: dict[str, Cell]) -> None:
    """Store each OK solver's median wall-clock time as ``cost_time_s``."""
    key = "by_N" if "by_N" in data else "by_steps" if "by_steps" in data else None
    top = data.get(key, {}) if key else data.get("by_solver", {})
    if not isinstance(top, dict):
        return
    for solver, cell in cells.items():
        if cell.status != OK:
            continue
        vals = top.get(solver, {})
        if not isinstance(vals, dict):
            continue
        times = [
            v["mean"]
            for v in vals.values()
            if isinstance(v, dict) and math.isfinite(v.get("mean", float("nan")))
        ]
        if times:
            _store_metric(cell, "cost_time_s", _median(times))


def _collect_fd_metrics(data: dict, cells: dict[str, Cell]) -> None:
    """Store best-ε ``rel_err`` and ``cosine`` per OK solver (mirrors ``_refine_fd_check``)."""
    by_solver = data.get("by_solver", {})
    if not isinstance(by_solver, dict):
        return
    for solver, cell in cells.items():
        if cell.status != OK:
            continue
        entry = by_solver.get(solver)
        sweep = entry.get("eps_sweep") if isinstance(entry, dict) else None
        if not isinstance(sweep, dict):
            continue
        best_cos = None
        best_rel = None
        for st in sweep.values():
            if not isinstance(st, dict):
                continue
            c = st.get("cosine")
            if isinstance(c, int | float) and math.isfinite(c):
                best_cos = c if best_cos is None else max(best_cos, c)
            vals = [
                r
                for r in (st.get("rel_error") or [])
                if isinstance(r, int | float) and math.isfinite(r)
            ]
            if vals:
                med = _median(vals)
                best_rel = med if best_rel is None else min(best_rel, med)
        _store_metric(cell, "rel_err", best_rel)
        _store_metric(cell, "cosine", best_cos)


def _collect_recovery_metrics(data: dict, cells: dict[str, Cell]) -> None:
    """Store the worst-case ``final_ratio`` (and ``final_loss`` when present) per OK solver."""
    top = data.get("by_solver") or data.get("by_sweep") or {}
    if not isinstance(top, dict):
        return
    for solver, cell in cells.items():
        if cell.status != OK:
            continue
        entry = top.get(solver)
        if not isinstance(entry, dict):
            continue
        series = _worst_case_trajectory(entry)
        if series:
            initial = abs(series[0])
            final = abs(series[-1])
            if initial > 0 and math.isfinite(final):
                _store_metric(cell, "final_ratio", final / initial)
        # ``final_loss`` is only written by some domains (e.g. ns-3d IC
        # recovery); store it when directly present, else fall back to the
        # trajectory endpoint so the metric exists for every optimisation cell.
        fl = entry.get("final_loss")
        if isinstance(fl, int | float) and math.isfinite(fl):
            _store_metric(cell, "final_loss", fl)
        elif series and math.isfinite(abs(series[-1])):
            _store_metric(cell, "final_loss", abs(series[-1]))


def _collect_forward_metrics(data: dict, cells: dict[str, Cell]) -> None:
    """Store each OK solver's median forward ``error`` as ``fwd_error``.

    Forward results are ``by_solver[solver] = {error, valid}`` (non-swept) or
    ``by_solver[solver][pval] = {error, valid}`` (swept). Only valid points
    contribute; the per-solver scalar is the median across valid sweep points.
    """
    by_solver = data.get("by_solver", {})
    if not isinstance(by_solver, dict):
        return
    for solver, cell in cells.items():
        if cell.status != OK:
            continue
        entry = by_solver.get(solver)
        if not isinstance(entry, dict):
            continue
        errs: list[float] = []
        # Swept: values are per-pval dicts. Non-swept: entry is the metrics dict.
        subs = (
            list(entry.values())
            if entry and all(isinstance(v, dict) for v in entry.values())
            else [entry]
        )
        for sub in subs:
            if not isinstance(sub, dict) or sub.get("valid") is False:
                continue
            e = sub.get("error")
            if isinstance(e, int | float) and math.isfinite(e):
                errs.append(float(e))
        if errs:
            _store_metric(cell, "fwd_error", _median(errs))


def _collect_metrics_for_suite(
    suite: str, exp_label: str, data: dict, cells: dict[str, Cell]
) -> None:
    """Populate ``cell.metrics`` with the suite's comparable numeric scalars.

    Runs unconditionally (no threshold needed) so a same-status numeric shift
    is diffable even for problems that configure no anomaly checks.
    """
    view = _v1_to_legacy_view(data) if data.get("schema_version") == 1 else data
    if suite == "cost":
        _collect_cost_metrics(view, cells)
    elif suite == "gradient" and exp_label.split("/")[0] in (
        "fd_check",
        "source_fd_check",
    ):
        _collect_fd_metrics(view, cells)
    elif suite == "optimization":
        _collect_recovery_metrics(view, cells)
    elif suite == "forward":
        _collect_forward_metrics(view, cells)


def _row_harness_stale(data: dict, harness_hash_cache: dict[str, str | None]) -> bool:
    """Whether the result's stored harness hash matches the current source.

    Missing-or-empty stored hash → stale.
    """
    # v1: provenance.harness_hash / provenance.harness_fn
    prov = data.get("provenance", {}) if data.get("schema_version") == 1 else {}
    stored_harness_hash = prov.get("harness_hash") or data.get("harness_hash")
    stored_harness_fn = prov.get("harness_fn") or data.get("harness_fn")
    if not stored_harness_hash or not stored_harness_fn:
        return True
    current = _resolve_harness_hash(stored_harness_fn, harness_hash_cache)
    return current is None or current != stored_harness_hash


def _apply_staleness(
    cfg: Problem,
    data: dict,
    cells: dict[str, Cell],
    solvers: list[str],
    tesseract_hash_cache: dict[str, str],
    harness_hash_cache: dict[str, str | None],
) -> None:
    """Mark row-level (harness) and cell-level (tesseract) staleness on cells.

    Row-level: if the stored harness hash differs from the current on-disk
    source (or no hash was stored), every non-excluded cell gets ``*``.
    Cell-level: mismatch or missing tesseract hash flags that solver alone
    even if the row as a whole isn't stale.
    """
    row_stale = _row_harness_stale(data, harness_hash_cache)
    # v1: provenance.tesseract_hashes
    prov = data.get("provenance", {}) if data.get("schema_version") == 1 else {}
    stored_tess = prov.get("tesseract_hashes") or data.get("tesseract_hashes") or {}
    if not isinstance(stored_tess, dict):
        stored_tess = {}
    for solver in solvers:
        cell = cells.get(solver)
        if cell is None or cell.status in (NOT_RUN, EXCLUDED):
            continue
        if row_stale:
            cell.stale = True
            continue
        stored = stored_tess.get(solver)
        if not stored:
            cell.stale = True
            continue
        current = _resolve_tesseract_hash(cfg, solver, tesseract_hash_cache)
        if current and stored != current:
            cell.stale = True


def _apply_exclusions(
    cfg: Problem, suite: str, exp_label: str, cells: dict[str, Cell]
) -> None:
    """Mark excluded solvers (overrides whatever the result file said).

    Reads from ``cfg.exclusions[name]`` (canonical store). Uses the shared
    ``exclusion_lookup`` helper so the status display and the runtime
    ``active_solvers`` filter can't drift on which key takes precedence.
    Most-specific wins: ``"{suite}/{exp}[/sub]" > "{exp}[/sub]" >
    "{suite}/{exp_head}" > "{exp_head}" > "{suite}"``. Entries with
    ``Exclusion.category == "anomaly_explained"`` are skipped here — they're
    handled by :func:`_apply_explained_anomalies` below.
    """
    for spec in cfg.solvers:
        name = spec.name
        match = exclusion_lookup(cfg.exclusions.get(spec.key, {}), suite, exp_label)
        if match is None:
            continue
        _key, value = match
        if getattr(value, "category", None) == "anomaly_explained":
            continue
        cells[name] = _build_excluded_cell(value)


def _apply_explained_anomalies(
    cfg: Problem, suite: str, exp_label: str, cells: dict[str, Cell]
) -> None:
    """Mark explained-anomaly solvers.

    These override OK cells only — the solver runs and produces finite
    results, but underperforms peers for documented method-intrinsic reasons.
    FAILED and EXCLUDED cells are never downgraded by this pass.

    Reads ``cfg.exclusions[name]`` filtered to entries with
    ``Exclusion.category == "anomaly_explained"``.
    """
    for spec in cfg.solvers:
        name = spec.name
        match = exclusion_lookup(cfg.exclusions.get(spec.key, {}), suite, exp_label)
        if match is None:
            continue
        _key, value = match
        if getattr(value, "category", None) != "anomaly_explained":
            continue
        cell = cells.get(name)
        if cell is None or cell.status in (FAILED, EXCLUDED):
            continue
        if cell.status == OK:
            cells[name] = _build_explained_anomaly_cell(value)
        elif cell.status == ANOMALY and cell.category != "explained":
            cells[name] = Cell(
                ANOMALY, cell.reason, category="explained", stale=cell.stale
            )


def _suite_filter(cfg: Problem, suite: str) -> set[str]:
    """Return the set of allowed experiment-head names for *suite*.

    Returns an empty set when no filter applies (every experiment is admitted).

    Walks ``cfg.experiments`` and returns the *first* path segment after the
    suite prefix for every entry that has a non-empty ``params`` payload —
    "configured experiments." Auto-sweep fan-out registers keys like
    ``forward/agreement/tgv``; we collapse those to ``agreement`` so the
    caller's ``exp_label.split("/")[0]`` membership test matches.

    Entries without params are registered in the suite catalog but not
    configured for this problem, so they're filtered out of the status
    display.
    """
    prefix = f"{suite}/"
    return {
        k[len(prefix) :].split("/", 1)[0]
        for k, exp in cfg.experiments.items()
        if k.startswith(prefix) and exp.params
    }


def _build_row(
    cfg: Problem,
    suite: str,
    exp_label: str,
    result_path: Path | None,
    solvers: list[str],
    tesseract_hash_cache: dict[str, str],
    harness_hash_cache: dict[str, str | None],
) -> ExperimentRow:
    """Construct one ExperimentRow with classified, refined, and stamped cells."""
    row = ExperimentRow(suite=suite, experiment=exp_label, result_path=result_path)
    if result_path is None:
        row.cells = {s: Cell(NOT_RUN) for s in solvers}
        _apply_exclusions(cfg, suite, exp_label, row.cells)
        _apply_explained_anomalies(cfg, suite, exp_label, row.cells)
        return row
    try:
        data = load_json(result_path)
    except Exception as exc:
        row.cells = {s: Cell(FAILED, f"unreadable result.json: {exc}") for s in solvers}
        # Even when the file is corrupted, exclusions still take precedence —
        # a solver that's categorically excluded shouldn't be reported as
        # ``fail`` just because the JSON for solvers that DID run is broken.
        _apply_exclusions(cfg, suite, exp_label, row.cells)
        _apply_explained_anomalies(cfg, suite, exp_label, row.cells)
        return row
    checks = _lookup_check(cfg, suite, exp_label)
    row.cells = _classify_result(data, solvers, checks)
    # Capture numeric scalars *before* anomaly refinement downgrades any OK
    # cell — the scalar is worth diffing whether or not it trips a threshold.
    _collect_metrics_for_suite(suite, exp_label, data, row.cells)
    _refine_for_suite(suite, exp_label, data, row.cells, checks)
    _apply_staleness(
        cfg, data, row.cells, solvers, tesseract_hash_cache, harness_hash_cache
    )
    _apply_exclusions(cfg, suite, exp_label, row.cells)
    _apply_explained_anomalies(cfg, suite, exp_label, row.cells)
    return row


def collect_status(cfg: Problem, suites: list[str] | None = None) -> ProblemStatus:
    """Build a ProblemStatus for one problem by walking its results/ tree."""
    suites = list(suites) if suites else list(SUITES)
    solvers = list(cfg.solver_names)
    root = _results_dir(cfg)
    # Caches shared across rows: hashing is O(files) per tesseract and
    # O(source-size) per harness fn — both stable within one status call.
    tesseract_hash_cache: dict[str, str] = {}
    harness_hash_cache: dict[str, str | None] = {}

    rows: list[ExperimentRow] = []
    seen: set[tuple[str, str]] = set()
    for suite in suites:
        suite_dir = root / suite
        allowed = _suite_filter(cfg, suite)
        for exp_label, result_path in _iter_experiment_dirs(suite_dir):
            if allowed and exp_label.split("/")[0] not in allowed:
                continue
            row = _build_row(
                cfg,
                suite,
                exp_label,
                result_path,
                solvers,
                tesseract_hash_cache,
                harness_hash_cache,
            )
            rows.append(row)
            seen.add((suite, exp_label))

    # Inject rows for registered experiments that have no on-disk result yet,
    # so an experiment is visible in the status table as soon as it's added
    # to the Problem — its cells render as ``missing`` (the NOT_RUN status).
    # ``ics/`` registrations are IC visualisations, not benchmark experiments,
    # so they're filtered out here.
    for full_key in sorted(cfg.experiments):
        suite, _, exp_label = full_key.partition("/")
        if not exp_label or suite == "ics" or suite not in suites:
            continue
        if (suite, exp_label) in seen:
            continue
        allowed = _suite_filter(cfg, suite)
        if allowed and exp_label.split("/")[0] not in allowed:
            continue
        row = _build_row(
            cfg,
            suite,
            exp_label,
            None,
            solvers,
            tesseract_hash_cache,
            harness_hash_cache,
        )
        rows.append(row)
        seen.add((suite, exp_label))

    return ProblemStatus(problem=cfg.name, solvers=solvers, rows=rows)


def _build_excluded_cell(value: Exclusion) -> Cell:
    """Construct an EXCLUDED cell from an :class:`Exclusion`.

    The cell's ``category`` is the raw string value of the enum member
    (e.g. ``"categorical"``), so existing comparisons against string
    literals continue to work.
    """
    return Cell(EXCLUDED, value.reason, category=value.category.value)


def _build_explained_anomaly_cell(value: Exclusion) -> Cell:
    """Construct an ANOMALY cell from an explained-anomaly :class:`Exclusion`.

    The solver runs and produces finite output but underperforms peers for
    documented method-intrinsic reasons (e.g. LBM compressibility floor,
    staggered MAC grid interpolation error). These appear in the status table
    as anomalies — not excluded — so they stay in the score denominator and
    solver weaknesses remain visible.

    ``category="explained"`` marks the cell as a pre-documented anomaly,
    distinguishing it from threshold-triggered anomalies without re-inspecting
    ``result.json``.
    """
    return Cell(ANOMALY, value.reason, category="explained")


# ── JSON / markdown / diff rendering ─────────────────────────────────────────
#
# These helpers are consumed by the `mosaic status --format {md,json}` CLI
# and by `mosaic status-diff` so a CI bot can post a PR comment comparing
# two snapshots of the campaign.


def status_to_dict(st: ProblemStatus) -> dict:
    """Convert a ProblemStatus into a JSON-serialisable dict.

    Includes the weighted ``score`` (and its denominator ``score_n``) so
    downstream consumers (CI bots, dashboards) get the same canonical
    metric without re-implementing the weighting.
    """
    t = tally(st)
    return {
        "problem": st.problem,
        "solvers": list(st.solvers),
        "score": t["score"],
        "score_n": t["score_n"],
        "rows": [
            {
                "suite": r.suite,
                "experiment": r.experiment,
                "label": r.label,
                "cells": {
                    s: {
                        "status": c.status,
                        "reason": c.reason,
                        "category": c.category,
                        "stale": c.stale,
                        "resource_frontier": c.resource_frontier,
                        "metrics": c.metrics,
                    }
                    for s, c in r.cells.items()
                },
            }
            for r in st.rows
        ],
    }


def snapshot_to_dict(statuses: list[ProblemStatus]) -> dict:
    """Bundle multiple problem snapshots into one dict for serialisation.

    Overall score is a weighted mean over per-problem scores, weighted by
    each problem's contributing-cell count (``score_n``).
    """
    num = 0.0
    den = 0
    per_problem: dict[str, dict] = {}
    for st in statuses:
        d = status_to_dict(st)
        per_problem[st.problem] = d
        if d["score"] is not None:
            num += d["score"] * d["score_n"]
            den += d["score_n"]
    overall = (num / den) if den else None
    return {
        "problems": per_problem,
        "score": overall,
        "score_n": den,
    }


def tally(st: ProblemStatus) -> dict[str, int]:
    """Return per-state counts for *st*, split excluded/stale counts, and the health score.

    ``excl_perm`` counts categorical (permanent) exclusions that don't count
    toward the %-ok denominator; ``excl_work`` counts every other exclusion
    category (work-to-do) which does. ``stale`` is the total number of cells
    with a ``*`` annotation (any underlying status). ``stale_ok`` is the
    subset of ``stale`` where the underlying status is OK — those cells do
    NOT count as fresh ok and therefore don't contribute to the numerator.
    The %-ok numerator is fresh-ok (OK cells that are not stale).

    ``score`` is the weighted sum of per-cell weights (``SCORE_WEIGHTS``)
    over non-categorical cells, divided by their count. Value ``None`` when
    there are no contributing cells (empty problem / all-categorical) —
    this is distinct from ``0.0``, which is a real data point meaning
    "all work-to-do, zero net progress". ``score_n`` is the denominator
    (non-categorical cell count) so callers can aggregate across problems
    with a proper weighted mean.
    """
    counts = {OK: 0, ANOMALY: 0, FAILED: 0, NOT_RUN: 0, EXCLUDED: 0}
    excl_perm = excl_work = stale = stale_ok = 0
    all_cells: list[Cell] = []
    for row in st.rows:
        for cell in row.cells.values():
            all_cells.append(cell)
            counts[cell.status] = counts.get(cell.status, 0) + 1
            is_stale = getattr(cell, "stale", False)
            if is_stale:
                stale += 1
                if cell.status == OK:
                    stale_ok += 1
            if cell.status == EXCLUDED:
                if cell.category in EXCL_PERMANENT:
                    excl_perm += 1
                else:
                    excl_work += 1
    fresh_ok = counts[OK] - stale_ok
    counts["fresh_ok"] = fresh_ok
    counts["excl_perm"] = excl_perm
    counts["excl_work"] = excl_work
    counts["stale"] = stale
    counts["stale_ok"] = stale_ok
    # Denominator: fresh-ok + every other work-to-do bucket + stale-ok.
    counts["total"] = (
        fresh_ok
        + counts[ANOMALY]
        + counts[FAILED]
        + counts[NOT_RUN]
        + excl_work
        + stale_ok
    )
    counts["pct_ok"] = 100.0 * fresh_ok / counts["total"] if counts["total"] else 0.0
    score, score_n = compute_score(all_cells)
    counts["score"] = score
    counts["score_n"] = score_n
    return counts


# Markdown glyphs — emoji render cleanly in GFM comments and avoid the
# monospace-width surprises that longer text labels produce inside tables.
_MD_GLYPHS = {
    OK: "✅",
    ANOMALY: "🟠",
    FAILED: "❌",
    NOT_RUN: "·",
    EXCLUDED: "⚪",
}

# Per-category glyph for EXCLUDED cells.
_MD_EXCL_GLYPHS = {
    ExclusionCategory.CATEGORICAL.value: "🚫",
}


def md_cell_glyph(cell: Cell) -> str:
    """Pick the markdown glyph for a cell, resolving the exclusion category.

    Appends ``*`` to the glyph when ``cell.stale`` is set (excluded cells
    never go stale — nothing to re-run).
    """
    if cell.status == EXCLUDED:
        glyph = _MD_EXCL_GLYPHS.get(cell.category, _MD_GLYPHS[EXCLUDED])
    else:
        glyph = _MD_GLYPHS.get(cell.status, "?")
    if getattr(cell, "stale", False) and cell.status != EXCLUDED:
        return glyph + "\\*"
    return glyph


_MD_LEGEND = (
    "**Legend** · "
    "✅ ok · "
    "🟠 anom · "
    "❌ fail · "
    "· missing · "
    "🚫 excluded (permanent — out of score denominator) · "
    "⚪ excluded (work-to-do) · "
    "**\\*** stale — result predates current benchmark run"
)

_MD_EXPLAINER = (
    "Each solver is run against every experiment in the suite. "
    "**ok** = produced valid results; "
    "**fail** = crashed or returned invalid data; "
    "**anom** = ran successfully but tripped an automated quality check "
    "(e.g. poor gradient accuracy, outlier wall-clock time, or diverged optimisation). "
    "Thresholds are defined per-problem in the problem config."
)


def format_score(score: float | None) -> str:
    """Plain-text score formatter: ``"0.62"`` / ``"—"``.

    ``None`` renders as a dash (no signal — all-categorical / empty).
    """
    if score is None:
        return "—"
    return f"{score:.2f}"


# ── unified weight → colour/emoji mapping ─────────────────────────────────────
#
# One canonical ladder drives every coloured element — cell labels, the
# per-problem score header, the overall summary score, the progress bar
# fill, and markdown cell glyphs. Callers pass a weight ``w ∈ [0.0, +1.0]``
# (from ``SCORE_WEIGHTS``) or ``None`` for "no signal" and get back a rich
# ansi colour or a GFM emoji from the same ladder.
#
# Health-signal continuous palette: 11 RGB control points at t = 0, 0.1, …,
# 1.0; linearly interpolated between stops. Designed to read as a traffic-light
# ramp: red (fail) → orange → yellow → green → bright green (ok).
#
# Ansi (hex via rich). Score weights annotated where they land; intermediate
# stops are ramp positions used for gradient fill / the status progress bar
# (e.g. its stale-ok segment sits at 0.67), not score weights themselves:
#   w = 0.00 → red           #cc0d0d   fail
#   w = 0.17 → red-orange    #e03a0b
#   w = 0.33 → orange        #f78005   missing / neutral / excl (work)
#   w = 0.53 → yellow        #f5d80f   anom
#   w = 0.67 → yellow-green  #73d114
#   w = 1.00 → bright green  #00e659   ok
#   None     → dim
#
# Markdown emoji (4 buckets — rough at-a-glance signal only):
#   w ≥ 0.65 → 🟢 · w ≥ 0.30 → 🟡 · w ≥ 0.15 → 🟠 · w < 0.15 → 🔴 · None → —

# Health-signal RGB control points at t = 0.0, 0.1, …, 1.0.
_HEALTH_LUT: tuple[tuple[float, float, float], ...] = (
    (0.800, 0.050, 0.050),  # t=0.0  red
    (0.870, 0.180, 0.040),  # t=0.1
    (0.930, 0.330, 0.030),  # t=0.2
    (0.970, 0.500, 0.020),  # t=0.3  orange
    (0.980, 0.670, 0.010),  # t=0.4
    (0.970, 0.850, 0.050),  # t=0.5  yellow
    (0.750, 0.870, 0.060),  # t=0.6
    (0.450, 0.820, 0.080),  # t=0.7  green
    (0.180, 0.780, 0.100),  # t=0.8
    (0.040, 0.850, 0.200),  # t=0.9
    (0.000, 0.900, 0.350),  # t=1.0  bright green
)


def weight_color(w: float | None) -> str:
    """Return a rich-markup hex colour for a weight ``w ∈ [0.0, +1.0]``.

    Health-signal palette: red (w=0) → orange → yellow → bright green (w=1).
    ``None`` → ``dim``.
    """
    if w is None:
        return "dim"
    w = max(0.0, min(1.0, w))
    n = len(_HEALTH_LUT) - 1
    pos = w * n
    lo = min(int(pos), n - 1)
    alpha = pos - lo
    r = _HEALTH_LUT[lo][0] + alpha * (_HEALTH_LUT[lo + 1][0] - _HEALTH_LUT[lo][0])
    g = _HEALTH_LUT[lo][1] + alpha * (_HEALTH_LUT[lo + 1][1] - _HEALTH_LUT[lo][1])
    b = _HEALTH_LUT[lo][2] + alpha * (_HEALTH_LUT[lo + 1][2] - _HEALTH_LUT[lo][2])
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def weight_emoji(w: float | None) -> str:
    """Return a GFM emoji for a weight — markdown analogue of ``weight_color``.

    Four buckets; ``None`` renders an em-dash.
      🟢 w ≥ 0.65  (ok*)   · 🟡 w ≥ 0.30 (neutral/anom)
      🟠 w ≥ 0.15  (bug/fail*) · 🔴 w < 0.15 (fail)
    """
    if w is None:
        return "—"
    if w >= 0.65:
        return "🟢"
    if w >= 0.30:
        return "🟡"
    if w >= 0.15:
        return "🟠"
    return "🔴"


def cell_weight(cell: Cell) -> float | None:
    """Return the SCORE_WEIGHTS value for a cell (None for categorical).

    Thin wrapper around ``cell_weight_key`` that looks the key up in the
    weight table. Categorical exclusions return ``None`` — the caller
    should treat them as "no signal" for colouring.
    """
    key = cell_weight_key(cell)
    if key is None:
        return None
    return SCORE_WEIGHTS.get(key)


def cell_color(cell: Cell) -> str:
    """Rich ansi colour for a cell, derived from its weight."""
    return weight_color(cell_weight(cell))


def _md_score_cell(score: float | None) -> str:
    """Markdown score cell.

    GFM doesn't support inline colour, so we use bolding + a colour-coded
    glyph prefix to convey the gradient.
    """
    if score is None:
        return "—"
    return f"{weight_emoji(score)} **{score:.2f}**"


def _md_format_reason(reason: str, *, collapse: bool = True) -> str:
    """Format a failure reason for markdown.

    When *collapse* is True (default, used for errors), long/multiline reasons
    show the first line as a summary with the full text in a ``<details>``
    block.  When False (used for anomalies), the full reason is always shown
    inline.
    """
    if not reason:
        return ""
    reason_lines = reason.strip().splitlines()
    summary = reason_lines[0].strip()[:120]
    is_long = len(reason_lines) > 1 or len(reason) > 120
    if not is_long or not collapse:
        return f" — {reason.strip()}"
    # Wrap the full reason in a collapsible block.
    # The <details> must not be indented — GFM breaks HTML blocks inside
    # list items when they carry leading whitespace or stray blank lines.
    full = reason.strip()
    return (
        f" — {summary}…\n"
        f"<details><summary>Full traceback</summary>\n\n"
        f"```\n{full}\n```\n\n"
        f"</details>\n"
    )


def render_markdown(statuses: list[ProblemStatus]) -> str:
    """Render a full status report as GitHub-flavored markdown.

    Structure:
      - Legend (glyph meanings)
      - Summary table (one row per problem + overall)
      - Anomalies / failures block (flat list, grouped by problem)
      - Per-problem detail tables inside <details> so the comment stays short
    """
    lines: list[str] = ["## Mosaic status", "", _MD_LEGEND, "", _MD_EXPLAINER, ""]

    # ── summary ─────────────────────────────────────────────────────────────
    # ok = fresh-ok (not stale). Stale ok cells show up only in the stale
    # column and contribute to the score via the `ok*` weight. excl(perm)
    # is categorical (method-intrinsic) — excluded from the score denominator.
    # `score` is the canonical campaign-health metric (see SCORE_WEIGHTS).
    lines += [
        "| problem | ok | anom | fail | missing | excl (work) | excl (perm) | stale | score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    t_fresh = t_anom = t_fail = t_miss = t_excl_work = t_excl_perm = t_stale = (
        t_stale_ok
    ) = 0
    # Aggregate score as a weighted mean across problems: sum(score·n) / sum(n).
    score_num = 0.0
    score_den = 0
    for st in statuses:
        c = tally(st)
        t_fresh += c["fresh_ok"]
        t_anom += c[ANOMALY]
        t_fail += c[FAILED]
        t_miss += c[NOT_RUN]
        t_excl_work += c["excl_work"]
        t_excl_perm += c["excl_perm"]
        t_stale += c["stale"]
        t_stale_ok += c["stale_ok"]
        if c["score"] is not None:
            score_num += c["score"] * c["score_n"]
            score_den += c["score_n"]
        lines.append(
            f"| `{st.problem}` | {c['fresh_ok']} | {c[ANOMALY]} | {c[FAILED]} | "
            f"{c[NOT_RUN]} | {c['excl_work']} | {c['excl_perm']} | "
            f"{c['stale']} | {_md_score_cell(c['score'])} |"
        )
    overall_score = (score_num / score_den) if score_den else None
    lines.append(
        f"| **overall** | **{t_fresh}** | **{t_anom}** | **{t_fail}** | "
        f"**{t_miss}** | **{t_excl_work}** | **{t_excl_perm}** | "
        f"**{t_stale}** | {_md_score_cell(overall_score)} |"
    )
    lines.append("")

    # ── failures & anomalies ────────────────────────────────────────────────
    fa: list[tuple[str, str, str, str, str]] = []
    for st in statuses:
        for row in st.rows:
            for solver, cell in row.cells.items():
                if cell.status in (FAILED, ANOMALY):
                    fa.append((st.problem, row.label, solver, cell.status, cell.reason))
    if fa:
        lines.append("### Failures & anomalies")
        lines.append("")
        for problem, label, solver, status, reason in fa:
            glyph = _MD_GLYPHS[status]
            reason_str = _md_format_reason(reason, collapse=(status == FAILED))
            lines.append(
                f"- {glyph} `{problem}` · `{label}` · **{solver}**{reason_str}"
            )
        lines.append("")

    # ── per-problem tables (collapsed) ─────────────────────────────────────
    for st in statuses:
        lines.append(
            f"<details><summary>{st.problem} — {len(st.rows)} experiment(s)</summary>"
        )
        lines.append("")
        header = "| experiment | " + " | ".join(f"`{s}`" for s in st.solvers) + " |"
        sep = "|---|" + "|".join(":---:" for _ in st.solvers) + "|"
        lines += [header, sep]
        for row in st.rows:
            cells = [f"`{row.label}`"]
            for s in st.solvers:
                cell = row.cells.get(s)
                cells.append(md_cell_glyph(cell) if cell else "?")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ── diff between two snapshots ───────────────────────────────────────────────

# Severity ordering lets us sort transitions so regressions surface first.
_SEVERITY = {OK: 0, EXCLUDED: 1, NOT_RUN: 2, ANOMALY: 3, FAILED: 4}


def _frontier_num(v: str) -> float | None:
    """Parse a frontier sweep-value string to a float for ordering, or None."""
    if not v:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _diff_frontier(
    out: dict,
    problem: str,
    label: str,
    solver: str,
    old_cell: dict,
    new_cell: dict,
) -> None:
    """Record a resource-ceiling frontier move into ``out["frontier_shifts"]``.

    A frontier is the first sweep value at which the solver hit an OOM /
    timeout. Comparing old vs new answers "did the ceiling move?":
      * ceiling appears where there was none, or moves to a *smaller* value
        → regression (solver now fails to scale as far as before)
      * ceiling disappears, or moves to a *larger* value → improvement
    Frontiers that don't parse as numbers are compared for equality only
    (a change is reported as a neutral shift rather than a direction).
    """
    old_f = old_cell.get("resource_frontier", "") or ""
    new_f = new_cell.get("resource_frontier", "") or ""
    if old_f == new_f:
        return
    rec = {
        "problem": problem,
        "label": label,
        "solver": solver,
        "from_frontier": old_f,
        "to_frontier": new_f,
    }
    old_n = _frontier_num(old_f)
    new_n = _frontier_num(new_f)
    # Direction: a ceiling that bites at a smaller size is worse. Treat "no
    # ceiling" as +∞ (scales further is better).
    old_v = old_n if old_n is not None else float("inf") if not old_f else None
    new_v = new_n if new_n is not None else float("inf") if not new_f else None
    if old_v is not None and new_v is not None:
        rec["direction"] = "regression" if new_v < old_v else "improvement"
    else:
        rec["direction"] = "changed"
    out["frontier_shifts"].append(rec)


def _diff_metrics(
    out: dict,
    problem: str,
    label: str,
    solver: str,
    old_cell: dict,
    new_cell: dict,
) -> None:
    """Record numeric-metric shifts into ``out["metric_shifts"]``.

    Only compares cells that are ``ok`` in *both* snapshots: a status
    transition is already reported by the main diff path (comparing numbers
    across a fail↔ok boundary is meaningless and would double-report the
    event). For each metric present in both cells, surfaces a shift when the
    relative change exceeds that metric's ``METRIC_SPECS`` threshold. Wall-clock
    (``cost_time_s``) is flagged ``confidence="indicative"`` — noisy across
    shared CI runners — while deterministic correctness metrics are ``firm``.
    """
    if old_cell.get("status") != OK or new_cell.get("status") != OK:
        return
    old_m = old_cell.get("metrics") or {}
    new_m = new_cell.get("metrics") or {}
    if not isinstance(old_m, dict) or not isinstance(new_m, dict):
        return
    for metric, (threshold, better, confidence) in METRIC_SPECS.items():
        if metric not in old_m or metric not in new_m:
            continue
        old_v = old_m[metric]
        new_v = new_m[metric]
        if not (isinstance(old_v, int | float) and isinstance(new_v, int | float)):
            continue
        if not (math.isfinite(old_v) and math.isfinite(new_v)):
            continue
        # Relative change against the old magnitude; a near-zero baseline uses
        # a tiny floor so a 0 → ε move doesn't report an infinite regression.
        denom = max(abs(old_v), 1e-12)
        pct = (new_v - old_v) / denom
        if abs(pct) < threshold:
            continue
        got_bigger = new_v > old_v
        # "lower is better": bigger = regression. "higher is better": flip.
        is_regression = got_bigger if better == "lower" else not got_bigger
        out["metric_shifts"].append(
            {
                "problem": problem,
                "label": label,
                "solver": solver,
                "metric": metric,
                "from": old_v,
                "to": new_v,
                "pct_change": pct,
                "direction": "regression" if is_regression else "improvement",
                "confidence": confidence,
            }
        )


def _cell_measured(measured: dict | None, problem: str, solver: str) -> bool:
    """Whether ``(problem, solver)`` was actually re-measured this run.

    ``measured`` is ``None`` for a full run (everything measured → True for
    all). Otherwise it carries ``problems`` (empty = every problem) and
    ``solvers`` (the solvers actually re-run). A cell counts as measured only
    when both match — carried-over baseline cells return False so their
    transitions aren't mis-attributed to the PR.
    """
    if measured is None:
        return True
    problems = measured.get("problems") or []
    solvers = measured.get("solvers") or []
    problem_ok = not problems or problem in problems
    solver_ok = not solvers or solver in solvers
    return problem_ok and solver_ok


def diff_snapshots(old: dict, new: dict, measured: dict | None = None) -> dict:
    """Compute transitions between two JSON snapshots produced by ``snapshot_to_dict``.

    Returns a dict with:
      regressions  — cells that got worse (higher severity)
      improvements — cells that got better
      other        — same-severity transitions (e.g. missing → excl)
      added_rows   — experiment rows present in new but not old
      removed_rows — experiment rows present in old but not new

    ``measured`` optionally scopes the diff to the (problem × solver) cells that
    were actually re-run this PR (``{"problems": [...], "solvers": [...]}``);
    cells outside it are carried-over baseline data, so any apparent transition
    is baseline drift rather than a PR effect and is skipped. ``None`` (default)
    diffs every cell — correct for full runs and local ``compare``.
    """
    out: dict = {
        "regressions": [],
        "improvements": [],
        "other": [],
        # Resource-ceiling (OOM / timeout) frontier moves between snapshots,
        # even when both cells stay OK. This is the signal a PR author cares
        # about most for scaling behaviour: did the solver start OOMing at a
        # *smaller* size than baseline (regression) or push the ceiling out
        # (improvement)?
        "frontier_shifts": [],
        # Numeric-metric shifts between cells that stay OK in both snapshots
        # (e.g. gradient rel-error worsened, converged loss improved). The
        # status ladder alone hides these; see ``_diff_metrics``.
        "metric_shifts": [],
        "added_rows": [],
        "removed_rows": [],
        # Snapshot-level score kept under underscored keys so existing callers
        # that iterate over the list-valued transition keys aren't surprised.
        "_old_snapshot": {"score": (old or {}).get("score")},
        "_new_snapshot": {"score": (new or {}).get("score")},
    }

    old_problems = (old or {}).get("problems", {})
    new_problems = (new or {}).get("problems", {})

    for problem, new_p in new_problems.items():
        old_p = old_problems.get(problem, {"rows": [], "solvers": []})
        old_rows = {r["label"]: r for r in old_p.get("rows", [])}
        new_rows = {r["label"]: r for r in new_p.get("rows", [])}
        for label, new_row in new_rows.items():
            if label not in old_rows:
                out["added_rows"].append({"problem": problem, "label": label})
                continue
            old_row = old_rows[label]
            for solver, new_cell in new_row["cells"].items():
                old_cell = old_row["cells"].get(solver)
                if old_cell is None:
                    continue
                # Skip cells this run didn't re-measure: their new value is
                # carried-over baseline data, so a transition is baseline drift,
                # not a PR effect (F3). Full runs / local compare pass
                # measured=None and diff everything.
                if not _cell_measured(measured, problem, solver):
                    continue
                # Resource frontier move (independent of the status transition).
                _diff_frontier(out, problem, label, solver, old_cell, new_cell)
                # Numeric-metric shift for cells that stay OK (no status change).
                _diff_metrics(out, problem, label, solver, old_cell, new_cell)
                same_status = old_cell["status"] == new_cell["status"]
                # A same-status change in category (e.g. excluded → excluded
                # but category moved from not_implemented to categorical) is
                # still worth surfacing as an "other" transition.
                same_category = old_cell.get("category", "") == new_cell.get(
                    "category", ""
                )
                if same_status and same_category:
                    continue
                rec = {
                    "problem": problem,
                    "label": label,
                    "solver": solver,
                    "from": old_cell["status"],
                    "from_category": old_cell.get("category", ""),
                    "to": new_cell["status"],
                    "to_category": new_cell.get("category", ""),
                    "reason": new_cell.get("reason", ""),
                }
                old_sev = _SEVERITY.get(old_cell["status"], 99)
                new_sev = _SEVERITY.get(new_cell["status"], 99)
                if new_sev > old_sev:
                    out["regressions"].append(rec)
                elif new_sev < old_sev:
                    out["improvements"].append(rec)
                else:
                    out["other"].append(rec)
        for label in old_rows:
            if label not in new_rows:
                out["removed_rows"].append({"problem": problem, "label": label})

    return out


# Human-readable labels for the metric keys in ``METRIC_SPECS``.
_METRIC_LABELS: dict[str, str] = {
    "rel_err": "gradient rel-error",
    "cosine": "gradient cosine",
    "final_ratio": "final/initial loss ratio",
    "final_loss": "final loss",
    "fwd_error": "forward error",
    "cost_time_s": "median time",
}


def _fmt_metric_value(metric: str, value: float) -> str:
    """Format a metric scalar: seconds for timing, else compact scientific."""
    if metric == "cost_time_s":
        return f"{value:.3g}s"
    return f"{value:.3g}"


def _fmt_metric_shift(r: dict) -> str:
    """One bullet line for a metric shift: glyph, location, from → to (±pct)."""
    g = "🔴" if r.get("direction") == "regression" else "🟢"
    metric = r["metric"]
    label = _METRIC_LABELS.get(metric, metric)
    old_s = _fmt_metric_value(metric, r["from"])
    new_s = _fmt_metric_value(metric, r["to"])
    pct = r.get("pct_change", 0.0) * 100.0
    sign = "+" if pct >= 0 else ""
    return (
        f"- {g} `{r['problem']}` · `{r['label']}` · **{r['solver']}** · "
        f"{label} {old_s} → {new_s} ({sign}{pct:.0f}%)"
    )


def _render_metric_shifts(lines: list[str], metric_shifts: list[dict]) -> None:
    """Render numeric-metric shifts, splitting firm from indicative (timing).

    Firm (deterministic correctness) metrics go under a headline section;
    indicative wall-clock shifts go under a clearly-labelled sub-section so a
    contributor never mistakes a shared-runner timing wobble for a hard
    regression (see F2 in the reporting audit).
    """
    firm = [r for r in metric_shifts if r.get("confidence") != "indicative"]
    indicative = [r for r in metric_shifts if r.get("confidence") == "indicative"]
    if firm:
        lines.append("### 📊 Metric changes")
        lines.append("")
        for r in firm:
            lines.append(_fmt_metric_shift(r))
        lines.append("")
    if indicative:
        lines.append("### ⏱ Timing (indicative — shared-runner contention)")
        lines.append("")
        lines.append(
            "_Wall-clock varies with CI-runner load, so treat these as "
            "indicative rather than controlled measurements._"
        )
        lines.append("")
        for r in indicative:
            lines.append(_fmt_metric_shift(r))
        lines.append("")


def render_diff_markdown(diff: dict) -> str:
    """Render a snapshot diff as markdown suitable for a PR comment."""
    lines: list[str] = ["## Status diff vs base", "", _MD_LEGEND, ""]
    n_reg = len(diff["regressions"])
    n_imp = len(diff["improvements"])
    n_oth = len(diff["other"])
    n_add = len(diff["added_rows"])
    n_rm = len(diff["removed_rows"])
    frontier_shifts = diff.get("frontier_shifts", [])
    n_front = len(frontier_shifts)
    metric_shifts = diff.get("metric_shifts", [])
    n_metric = len(metric_shifts)

    # Score delta header: uses snapshot-level score if present, falls back to
    # None for legacy snapshots where the field is absent.
    def _snap_score(snap: dict | None) -> float | None:
        if not isinstance(snap, dict):
            return None
        s = snap.get("score")
        return float(s) if isinstance(s, int | float) else None

    # Threaded through diff_snapshots' closure via module-level access to
    # the raw snapshots is awkward; instead look the scores up from any
    # embedded hints the caller attached. If absent, leave the header bare.
    old_score = _snap_score(diff.get("_old_snapshot"))
    new_score = _snap_score(diff.get("_new_snapshot"))

    if n_reg == n_imp == n_oth == n_add == n_rm == n_front == n_metric == 0:
        if old_score is not None or new_score is not None:
            lines.append(
                f"_No status changes._ · score "
                f"{format_score(old_score)} → {format_score(new_score)}"
            )
        else:
            lines.append("_No status changes._")
        return "\n".join(lines) + "\n"

    header_bits = [
        f"**{n_reg} regression(s)**",
        f"**{n_imp} improvement(s)**",
        f"{n_metric} metric change(s)",
        f"{n_oth} other transition(s)",
        f"{n_front} resource-frontier shift(s)",
        f"{n_add} new row(s)",
        f"{n_rm} removed row(s)",
    ]
    if old_score is not None or new_score is not None:
        header_bits.append(
            f"score {format_score(old_score)} → {format_score(new_score)}"
        )
    lines.append(" · ".join(header_bits))
    lines.append("")

    def _glyph(status: str, category: str) -> str:
        if status == EXCLUDED:
            return _MD_EXCL_GLYPHS.get(category, _MD_GLYPHS[EXCLUDED])
        return _MD_GLYPHS.get(status, status)

    def _fmt_rec(r: dict) -> str:
        src = _glyph(r["from"], r.get("from_category", ""))
        dst = _glyph(r["to"], r.get("to_category", ""))
        reason_str = _md_format_reason(
            r.get("reason", ""), collapse=(r["to"] == FAILED)
        )
        return (
            f"- {src}→{dst} `{r['problem']}` · `{r['label']}` · "
            f"**{r['solver']}**{reason_str}"
        )

    if diff["regressions"]:
        lines.append("### 🔴 Regressions")
        lines.append("")
        for r in diff["regressions"]:
            lines.append(_fmt_rec(r))
        lines.append("")

    if diff["improvements"]:
        lines.append("### 🟢 Improvements")
        lines.append("")
        for r in diff["improvements"]:
            lines.append(_fmt_rec(r))
        lines.append("")

    if metric_shifts:
        _render_metric_shifts(lines, metric_shifts)

    if frontier_shifts:
        lines.append("### Resource-frontier shifts (OOM / timeout)")
        lines.append("")
        lines.append(
            "_The size at which a solver first hits a resource ceiling. "
            "A smaller frontier means it now fails to scale as far as before._"
        )
        lines.append("")
        _dir_glyph = {"regression": "🔴", "improvement": "🟢", "changed": "⚪"}
        for r in frontier_shifts:
            g = _dir_glyph.get(r.get("direction", "changed"), "⚪")
            old_f = r.get("from_frontier") or "none"
            new_f = r.get("to_frontier") or "none"
            lines.append(
                f"- {g} `{r['problem']}` · `{r['label']}` · **{r['solver']}** · "
                f"ceiling {old_f} → {new_f}"
            )
        lines.append("")

    if diff["other"]:
        lines.append("### Other transitions")
        lines.append("")
        for r in diff["other"]:
            lines.append(_fmt_rec(r))
        lines.append("")

    if diff["added_rows"]:
        lines.append("### Added experiments")
        lines.append("")
        for r in diff["added_rows"]:
            lines.append(f"- `{r['problem']}` · `{r['label']}`")
        lines.append("")

    if diff["removed_rows"]:
        lines.append("### Removed experiments")
        lines.append("")
        for r in diff["removed_rows"]:
            lines.append(f"- `{r['problem']}` · `{r['label']}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
