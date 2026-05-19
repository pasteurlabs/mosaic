"""Status-based per-(experiment, solver) filter for ``mosaic run --only``.

The CLI builds a filter via :func:`build_filter` from a fresh
:func:`collect_status` snapshot, stashes it as the active filter, then runs
the experiment loop. :func:`run_experiment` calls :func:`filter_solvers`
right after the per-experiment ``selector_fn`` to prune the solver list to
the cells matching the requested state(s) — solvers whose cells are
already fresh-ok pass straight through without being re-executed.

States accepted by ``--only`` (comma-separated, combinable):

- ``failed``    → cells with status FAILED
- ``anom``      → cells with status ANOMALY
- ``missing``   → cells with status NOT_RUN (no result.json yet)
- ``stale``     → any cell with ``Cell.stale == True``
- ``excluded``  → cells with status EXCLUDED (typically the work-to-do
                  category — re-run after dropping an exclusion)

Sub-experiment fan-out (e.g. ``forward/baseline/N_16``) rolls up to its
registered parent (``forward/baseline``): if any sub-cell matches, the
parent (key, solver) pair is included. The runner re-executes the whole
parent — which is the smallest unit it knows how to schedule.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from .status import ANOMALY, EXCLUDED, FAILED, NOT_RUN, collect_status

if TYPE_CHECKING:
    from .config import Problem
    from .status import Cell

_VALID_STATES: frozenset[str] = frozenset(
    {"failed", "anom", "missing", "stale", "excluded"}
)

_STATUS_TO_STATE: dict[str, str] = {
    FAILED: "failed",
    ANOMALY: "anom",
    NOT_RUN: "missing",
    EXCLUDED: "excluded",
}

# Thread-local active filter so concurrent test runs don't collide.
_state = threading.local()


def _cell_matches(cell: Cell, states: set[str]) -> bool:
    """True iff *cell* has at least one of the requested *states*."""
    if "stale" in states and cell.stale:
        return True
    return _STATUS_TO_STATE.get(cell.status) in states


def build_filter(
    cfg: Problem, suites: list[str] | None, states: set[str]
) -> dict[tuple[str, str], bool]:
    """Return ``{(<suite>/<parent_exp>, solver) → True}`` for matching cells.

    Raises ``ValueError`` on unknown state names. Sub-experiments from
    sweeps roll up to their registered parent — if any sub-cell matches,
    the parent (key, solver) pair is included.
    """
    unknown = states - _VALID_STATES
    if unknown:
        raise ValueError(
            f"--only: unknown state(s) {sorted(unknown)}. "
            f"Valid: {sorted(_VALID_STATES)}."
        )
    st = collect_status(cfg, suites)
    registered = set(cfg.experiments)
    keep: dict[tuple[str, str], bool] = {}
    for row in st.rows:
        full = f"{row.suite}/{row.experiment}"
        # Walk up to the nearest registered parent (handles sweep sub-dirs
        # like "forward/baseline/N_16" → "forward/baseline").
        parent = full
        while parent not in registered and "/" in parent:
            parent = parent.rsplit("/", 1)[0]
        if parent not in registered:
            continue
        for solver, cell in row.cells.items():
            if _cell_matches(cell, states):
                keep[(parent, solver)] = True
    return keep


def set_active(filter_map: dict[tuple[str, str], bool] | None) -> None:
    """Install *filter_map* as the active per-cell filter, or clear it
    with ``None``."""
    _state.filter = filter_map


def get_active() -> dict[tuple[str, str], bool] | None:
    return getattr(_state, "filter", None)


def filter_solvers(exp_key: str, solver_names: list[str]) -> list[str]:
    """Prune *solver_names* to those matching the active filter for *exp_key*.

    ``exp_key`` must be the full registered key (``<suite>/<experiment>``).
    Returns *solver_names* unchanged when no filter is active.
    """
    f = get_active()
    if f is None:
        return solver_names
    return [s for s in solver_names if (exp_key, s) in f]
