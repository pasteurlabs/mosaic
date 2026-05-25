"""Partial-result checkpointing for crash-resumable experiments.

Long benchmark runs that OOM, hit a host reboot, or take a SIGKILL can leave
hours of completed solver work unrecoverable if their results only land on
disk at experiment-end. ``write_partial`` snapshots the current in-memory
result dict to ``<exp_dir>/result_partial.json`` after each solver finishes
(or at any other safe checkpoint a harness chooses); ``done_solvers_in_partial``
reads that file back on the next ``mosaic run --continue`` to filter out
solvers that have already completed.

Schema conventions interpreted by ``done_solvers_in_partial``:

* ``by_solver`` / ``by_sweep`` — outer keys are solver names. A solver is
  considered done iff its entry is a dict that does NOT carry
  ``"in_progress": True``.

* ``by_param`` / ``by_N`` / ``by_steps`` — outer keys are sweep values,
  inner keys are solver names. Conservative semantics: a solver is done
  only if it appears in **every** sweep bucket. A solver in some but not
  all buckets is treated as mid-flight and will re-run.

The file is read-only after the experiment completes (``save_experiment``
writes the canonical ``result.json`` separately and merges any prior
per-solver data via its existing flock). It's only consumed by the
``--continue`` codepath in each harness — never by plots or external tools.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from filelock import FileLock

from .io import save_json


def write_partial(
    out_dir: Path,
    payload: dict,
    lock: threading.Lock | None = None,
) -> None:
    """Atomically snapshot *payload* to ``<out_dir>/result_partial.json``.

    Cross-process safety comes from the per-dir ``.save_experiment.lock``
    flock that ``save_experiment`` and ``save_gradient_fields_npz`` already
    use. The optional in-process ``lock`` argument additionally serialises
    threaded harness writes that share the same closure-captured payload
    dict (the parallel-GPU dispatch path in ``run_with_gpu_pool``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if lock is not None:
        lock.acquire()
    try:
        with FileLock(out_dir / ".save_experiment.lock"):
            save_json(payload, out_dir / "result_partial.json")
    finally:
        if lock is not None:
            lock.release()


# Schemas where the outer dict is keyed by solver name. Solver completion
# is decided by inspecting the entry's ``in_progress`` flag.
_OUTER_SOLVER_SCHEMAS = ("by_solver", "by_sweep")

# Schemas where the outer dict is keyed by sweep value, with solver names
# living in the inner buckets. Conservative completion rule: a solver is
# done only if it appears in every bucket.
_OUTER_SWEEP_SCHEMAS = ("by_param", "by_N", "by_steps")


def filter_resumable_solvers(
    solver_names: list[str], out_dir: Path, overrides: dict | None
) -> list[str]:
    """Drop solvers already recorded as complete in ``<out_dir>/result_partial.json``.

    Pass-through when ``overrides`` does not carry ``resume: True`` (i.e. the
    user did not pass ``--continue``) — so callsites can wrap unconditionally
    without re-implementing the gate.
    """
    if not overrides or not overrides.get("resume"):
        return solver_names
    done = done_solvers_in_partial(out_dir / "result_partial.json")
    if not done:
        return solver_names
    return [s for s in solver_names if s not in done]


def done_solvers_in_partial(partial_path: Path) -> set[str]:
    """Return solver names provably complete in *partial_path*.

    Returns an empty set if the file is missing, unreadable, or carries
    no recognised schema — the caller treats that as "nothing to skip".

    The conservative semantics on sweep-style schemas (intersection across
    buckets) mean a solver that died partway through its sweep WILL re-run.
    That's intentional: skipping it would silently drop the un-computed
    buckets.
    """
    if not partial_path.exists():
        return set()
    try:
        with open(partial_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(data, dict):
        return set()

    for key in _OUTER_SOLVER_SCHEMAS:
        top = data.get(key)
        if isinstance(top, dict):
            return {
                name
                for name, entry in top.items()
                if isinstance(entry, dict) and not entry.get("in_progress")
            }

    for key in _OUTER_SWEEP_SCHEMAS:
        top = data.get(key)
        if isinstance(top, dict) and top:
            solver_sets = [
                set(bucket.keys())
                for bucket in top.values()
                if isinstance(bucket, dict)
            ]
            if solver_sets:
                return set.intersection(*solver_sets)

    return set()
