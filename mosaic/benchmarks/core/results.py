"""Experiment-artifact writers.

Owns the three functions that translate an in-memory result dict into the
on-disk artifacts ``mosaic status`` and the plotting layer read back:

  * :func:`save_experiment` — writes ``result.json`` (with merge-aware
    handling of partial reruns), ``params.json``, and optionally
    ``result.csv``. Stamps the result with tesseract content hashes and
    a harness-function hash so the status display can flip cells to the
    ``*`` (stale) annotation when source has drifted from what produced
    the saved result.
  * :func:`save_field_snapshots_npz` — atomically merge-saves the
    positional-indexed per-solver npz used by every suite for storing
    field-array snapshots (gradients, IC/density iteration histories,
    forward agreement outputs, …). Suites that produce per-solver arrays
    at a discrete set of evaluation points should route them through
    this writer rather than calling ``np.savez`` directly, so concurrent
    runs in the same directory don't race-overwrite each other.
  * :func:`_environment_metadata` — gathers Python / mosaic version /
    timestamp metadata stamped into every saved result.

Low-level IO helpers (``save_json``, ``load_json``, ``save_csv``,
``_file_lock``) and the staleness hashers (``tesseract_content_hash``,
``harness_fn_hash``) continue to live in :mod:`mosaic.benchmarks.core.utils`;
this module imports them.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path

import numpy as np

from mosaic.benchmarks.core.config import ProblemConfig
from mosaic.benchmarks.core.utils import (
    _file_lock,
    harness_fn_hash,
    save_csv,
    save_json,
    tesseract_content_hash,
    try_load_json,
    try_load_npz,
)


def save_field_snapshots_npz(
    out_dir: Path,
    solver_names: list[str],
    per_solver_arrays: dict[str, dict[str, np.ndarray]],
    shared_arrays: dict[str, np.ndarray] | None = None,
    filename: str = "gradient_fields.npz",
    prefixes: tuple[str, ...] = ("grad",),
    *,
    flat_keys: bool = False,
) -> None:
    """Atomically merge-save a per-solver npz of field snapshots.

    Two on-disk layouts are supported:

    **Positional** (``flat_keys=False``, default): per-solver arrays are
    written under keys of the form ``{prefix}_{j}`` or
    ``{prefix}_{j}_{suffix}`` where ``j`` is the index of the solver in the
    ``solver_names`` string array. This is the legacy gradient-suite layout
    and is read back by recovering ``solver_names`` from the npz.

    **Flat** (``flat_keys=True``): per-solver arrays are written under keys
    of the form ``{solver_name}_{suffix}``. Used by the forward-agreement
    suite where plotting code reads keys directly by solver name. The
    ``prefixes`` argument is ignored in this mode.

    Both layouts:
      1) Acquire the per-dir ``.save_experiment.lock`` so concurrent
         mosaic processes writing the same directory don't race —
         whichever writes last would otherwise drop the other's solvers.
      2) Load any existing npz, decode its per-solver entries back to
         solver names, and merge the caller's ``per_solver_arrays`` into
         them (caller wins on collision, matching ``by_solver`` merge
         semantics in :func:`save_experiment`).
      3) Write the merged npz with a canonical ``solver_names`` ordering
         (caller's names first, then older solvers appended).

    ``per_solver_arrays``: ``{solver_name: {"<prefix>:<suffix>": array, ...}}``
        for positional mode, or ``{solver_name: {"<suffix>": array, ...}}``
        for flat mode. In positional mode, the key may either be
        ``"<prefix>:<suffix>"`` (explicit prefix) or just ``"<suffix>"``
        (implicit first prefix); use ``suffix=""`` for the plain
        ``{prefix}_{j}`` layout.

    ``shared_arrays``: optional ``{key: array}`` — written verbatim alongside
        per-solver entries (``ic``, ``N_values``, ``consensus_0``…). Caller
        wins on collision with existing shared keys.

    ``prefixes``: tuple of positional-key prefixes to participate in merging
        (e.g. ``("grad",)`` for the gradient suite, ``("rho_final",
        "rho_history")`` for topopt recovery). Ignored when
        ``flat_keys=True``. Non-matching keys are preserved as shared.

    ``shared_prefixes``: keys whose names start with one of these strings
        (e.g. ``"consensus_"``) are treated as shared even in flat mode,
        so they don't get parsed as ``{solver_name}_{suffix}``. Passed via
        ``prefixes`` in flat mode (semantic overload — it's the only
        argument that means "non-per-solver" in either layout).

    Default ``filename`` is ``gradient_fields.npz`` for historical reasons —
    callers handling non-gradient artifacts should pass an explicit name
    (e.g. ``filename="fields.npz"``) so the file's contents are
    self-describing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".save_experiment.lock"
    npz_path = out_dir / filename

    with _file_lock(lock_path):
        # try_load_npz returns {} on missing / corrupt — no try/except here.
        old = try_load_npz(npz_path)
        old_names = [str(n) for n in old.pop("solver_names", np.array([])).tolist()]
        if flat_keys:
            payload = _build_flat_payload(
                old_arrays=old,
                old_names=old_names,
                solver_names=solver_names,
                per_solver_arrays=per_solver_arrays,
                shared_arrays=shared_arrays or {},
                shared_prefixes=tuple(prefixes) if prefixes else (),
            )
        else:
            payload = _build_positional_payload(
                old_arrays=old,
                old_names=old_names,
                solver_names=solver_names,
                per_solver_arrays=per_solver_arrays,
                shared_arrays=shared_arrays or {},
                prefixes=prefixes,
            )
        np.savez(npz_path, **payload)


def _build_flat_payload(
    *,
    old_arrays: dict[str, np.ndarray],
    old_names: list[str],
    solver_names: list[str],
    per_solver_arrays: dict[str, dict[str, np.ndarray]],
    shared_arrays: dict[str, np.ndarray],
    shared_prefixes: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Build the on-disk dict for the flat ``{solver_name}_{suffix}`` layout.

    Pure function: takes the already-loaded existing-npz dict (``old_arrays``)
    and the caller's new contributions, returns the merged payload ready for
    ``np.savez``. No I/O, no exceptions.
    """
    known = set(old_names) | set(solver_names)
    # Sort once, longest first, so e.g. "ins_jl_diff" wins over "ins_jl".
    known_sorted = sorted(known, key=len, reverse=True)

    merged_per_solver: dict[str, dict[str, np.ndarray]] = {}
    merged_shared: dict[str, np.ndarray] = {}
    for k, arr in old_arrays.items():
        if any(k.startswith(p) for p in shared_prefixes):
            merged_shared[k] = arr
            continue
        matched = next(
            (s for s in known_sorted if k == s or k.startswith(s + "_")), None
        )
        if matched is None:
            merged_shared[k] = arr
            continue
        suffix = k[len(matched) :].lstrip("_")
        merged_per_solver.setdefault(matched, {})[suffix] = arr

    for sname, suf_map in per_solver_arrays.items():
        merged_per_solver.setdefault(sname, {}).update(
            {k: np.asarray(v) for k, v in suf_map.items()}
        )
    merged_shared.update(shared_arrays)

    ordered = list(solver_names) + [
        s for s in merged_per_solver if s not in solver_names
    ]
    payload: dict[str, np.ndarray] = {"solver_names": np.array(ordered)}
    payload.update(merged_shared)
    for sname, suf_map in merged_per_solver.items():
        for suffix, arr in suf_map.items():
            payload[sname if suffix == "" else f"{sname}_{suffix}"] = arr
    return payload


def _build_positional_payload(
    *,
    old_arrays: dict[str, np.ndarray],
    old_names: list[str],
    solver_names: list[str],
    per_solver_arrays: dict[str, dict[str, np.ndarray]],
    shared_arrays: dict[str, np.ndarray],
    prefixes: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Build the on-disk dict for the positional ``{prefix}_{j}_{suffix}`` layout.

    Pure function: caller is responsible for loading the existing npz into
    ``old_arrays``. Unknown keys (no matching prefix, or out-of-range solver
    index) are preserved as shared.
    """

    def _parse_positional(key: str) -> tuple[str, int, str] | None:
        for p in prefixes:
            if not (key == p or key.startswith(p + "_")):
                continue
            rest = key[len(p) :]
            if not rest or rest[0] != "_":
                return None
            parts = rest[1:].split("_", 1)
            if not parts[0].lstrip("-").isdigit():
                continue
            return (p, int(parts[0]), parts[1] if len(parts) > 1 else "")
        return None

    def _canonicalise_input(
        per_solver: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[tuple[str, str], np.ndarray]]:
        out: dict[str, dict[tuple[str, str], np.ndarray]] = {}
        default_prefix = prefixes[0]
        for sname, suf_map in per_solver.items():
            norm: dict[tuple[str, str], np.ndarray] = {}
            for k, arr in suf_map.items():
                p, s = k.split(":", 1) if ":" in k else (default_prefix, k)
                norm[(p, s)] = np.asarray(arr)
            out[sname] = norm
        return out

    merged_per_solver: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    merged_shared: dict[str, np.ndarray] = {}
    for k, arr in old_arrays.items():
        parsed = _parse_positional(k)
        if parsed is None:
            merged_shared[k] = arr
            continue
        p, j, suffix = parsed
        if j < 0 or j >= len(old_names):
            merged_shared[k] = arr
            continue
        merged_per_solver.setdefault(old_names[j], {})[(p, suffix)] = arr

    for sname, suf_map in _canonicalise_input(per_solver_arrays).items():
        merged_per_solver.setdefault(sname, {}).update(suf_map)
    merged_shared.update(shared_arrays)

    ordered = list(solver_names) + [
        s for s in merged_per_solver if s not in solver_names
    ]
    payload: dict[str, np.ndarray] = {"solver_names": np.array(ordered)}
    payload.update(merged_shared)
    for j, sname in enumerate(ordered):
        for (p, suffix), arr in merged_per_solver.get(sname, {}).items():
            payload[f"{p}_{j}" if suffix == "" else f"{p}_{j}_{suffix}"] = arr
    return payload


def _environment_metadata() -> dict:
    """Gather environment metadata for result provenance tracking.

    Imports are deferred to avoid module-level overhead.
    """
    import platform as _platform
    from datetime import datetime, timezone

    version = "unknown"
    with contextlib.suppress(Exception):
        from importlib.metadata import version as _pkg_version

        version = _pkg_version("mosaic-bench")

    return {
        "python_version": _platform.python_version(),
        "platform": _platform.platform(),
        "mosaic_version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_experiment(
    result: dict,
    out_dir: Path,
    csv_rows: list[dict] | None = None,
    cfg: ProblemConfig | None = None,
    harness_fn=None,
    wall_time_s: dict[str, float] | None = None,
) -> None:
    """Save result.json, params.json, and optionally result.csv for one experiment.

    If an existing result.json already has a ``by_solver`` dict and the new
    result shares the same *physics* ``params`` (parameter-compatible run),
    merges the new ``by_solver`` entries into the existing ones — preserving
    data from solvers that were NOT part of the current run. This prevents a
    single-solver rerun from silently wiping out previously-benchmarked solvers.
    If params differ, the new result overwrites wholesale (old data is no
    longer valid).

    Runtime/scheduling-only keys (e.g. ``gpu_ids``) are excluded from the
    equality check — they do not affect physics or solver output.

    Staleness stamps (optional): when *cfg* is supplied, a
    ``tesseract_hashes`` dict {solver_name: content-hash} is injected for
    every solver present in the result. When *harness_fn* is supplied, the
    result is also stamped with ``harness_hash`` (SHA of the function's
    source) and ``harness_fn`` (``module.qualname``). ``mosaic status`` uses
    these to flip cells to the ``*`` (stale) annotation when the current
    source differs from what produced the saved result.
    """
    # Keys that are scheduling/runtime only and should not gate merge behaviour.
    _SCHEDULING_KEYS = {"gpu_ids"}

    def _normalise(v):
        """Recursively normalise a value for params equality comparison.

        JSON round-trips convert Python sets to sorted lists (via _NumpyEncoder).
        On a re-run the in-memory ``run`` dict may still carry sets (e.g.
        ``fine.solvers = {"exponax", "jax_cfd"}``), while the previously saved
        ``params.json`` carries ``["exponax", "jax_cfd"]``.  Without
        normalisation the equality check always fails for such params, so the
        merge branch is never reached and existing solver data is overwritten.
        """
        if isinstance(v, set):
            return sorted(v)
        if isinstance(v, dict):
            return {k: _normalise(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_normalise(x) for x in v]
        return v

    def _physics_params(p):
        if not isinstance(p, dict):
            return _normalise(p)
        return _normalise({k: v for k, v in p.items() if k not in _SCHEDULING_KEYS})

    def _solvers_in_result(
        res: dict, known_solvers: set[str] | None = None
    ) -> set[str]:
        """Collect solver names appearing in any known top-level map or in a
        custom schema (e.g. ``per_solver_spectra``, ``grad_norms``,
        ``landscape.by_solver``).

        Strategy:
          1. Hard-coded canonical maps (``by_solver``, ``by_sweep``, ``by_N``,
             ``by_steps``, ``by_param[...]``).
          2. Generic pass: for any top-level value that is a dict whose keys
             overlap the registered solver names in ``cfg.solvers``, treat those
             keys as solver names. Recurse one level so nested maps such as
             ``landscape.by_solver`` are picked up too.
        """
        names: set[str] = set()
        for key in ("by_solver", "by_sweep"):
            top = res.get(key)
            if isinstance(top, dict):
                names.update(str(k) for k in top)
        for key in ("by_N", "by_steps"):
            top = res.get(key)
            if isinstance(top, dict):
                names.update(str(k) for k in top)
        if isinstance(res.get("by_param"), dict):
            for solver_map in res["by_param"].values():
                if isinstance(solver_map, dict):
                    names.update(str(k) for k in solver_map)

        if known_solvers:

            def _scan(node, depth: int) -> None:
                if depth < 0 or not isinstance(node, dict):
                    return
                keys_as_str = {str(k) for k in node}
                if keys_as_str & known_solvers:
                    names.update(keys_as_str & known_solvers)
                for v in node.values():
                    if isinstance(v, dict):
                        _scan(v, depth - 1)

            _scan(res, depth=2)
        return names

    new_tesseract_hashes: dict[str, str] = {}
    if isinstance(result, dict) and cfg is not None:
        known_solvers = {str(k) for k in cfg.solvers}
        for s in _solvers_in_result(result, known_solvers=known_solvers):
            spec = cfg.solvers.get(s)
            if spec is None:
                continue
            tess_dir = cfg.tesseract_dir / spec.dir
            if tess_dir.is_dir():
                h = tesseract_content_hash(tess_dir)
                if h:
                    new_tesseract_hashes[s] = h

    new_harness_hash = ""
    new_harness_fn_qualname = ""
    if harness_fn is not None:
        new_harness_hash = harness_fn_hash(harness_fn)
        mod = getattr(harness_fn, "__module__", "") or ""
        qual = getattr(harness_fn, "__qualname__", "") or getattr(
            harness_fn, "__name__", ""
        )
        new_harness_fn_qualname = f"{mod}.{qual}" if mod else qual

    result_path = out_dir / "result.json"
    lock_path = out_dir / ".save_experiment.lock"
    _flock_t0 = time.monotonic()
    with _file_lock(lock_path):
        existing = try_load_json(result_path) if isinstance(result, dict) else None
        if isinstance(existing, dict) and _physics_params(
            existing.get("params")
        ) == _physics_params(result.get("params")):
            # ── by_solver merge (calibration, gradient, …) ────────────────
            if isinstance(result.get("by_solver"), dict) and isinstance(
                existing.get("by_solver"), dict
            ):
                merged_by_solver = {
                    **existing["by_solver"],
                    **result["by_solver"],
                }
                result = {**existing, **result, "by_solver": merged_by_solver}

            # ── by_param merge (agreement / baseline / physical_laws) ──────
            elif isinstance(result.get("by_param"), dict) and isinstance(
                existing.get("by_param"), dict
            ):
                merged_by_param: dict = {
                    str(k): v for k, v in existing["by_param"].items()
                }
                for pval, solver_map in result["by_param"].items():
                    spval = str(pval)
                    if (
                        spval in merged_by_param
                        and isinstance(merged_by_param[spval], dict)
                        and isinstance(solver_map, dict)
                    ):
                        merged_by_param[spval] = {
                            **merged_by_param[spval],
                            **solver_map,
                        }
                    else:
                        merged_by_param[spval] = solver_map
                merged_spread: dict = {
                    str(k): v for k, v in existing.get("spread", {}).items()
                }
                merged_spread.update(
                    {str(k): v for k, v in result.get("spread", {}).items()}
                )
                result = {
                    **existing,
                    **result,
                    "by_param": merged_by_param,
                    "spread": merged_spread,
                }

            # ── by_sweep merge (recovery / topopt sweeps) ─────────────────
            elif isinstance(result.get("by_sweep"), dict) and isinstance(
                existing.get("by_sweep"), dict
            ):
                merged_by_sweep = {
                    **existing["by_sweep"],
                    **result["by_sweep"],
                }
                result = {**existing, **result, "by_sweep": merged_by_sweep}

            # ── by_N / by_steps merge (cost suite) ────────────────────────
            # Cost suites pre-populate every solver key (empty dict for
            # solvers that failed), so a naive outer-merge would overwrite
            # a successful prior run's per-size entries with empty dicts.
            # Per-solver merge: prefer the new run's non-empty entries,
            # fall back to existing data when the new run produced an
            # empty dict for that solver.
            if isinstance(result.get("by_N"), dict) and isinstance(
                existing.get("by_N"), dict
            ):
                merged_by_N: dict = {**existing["by_N"]}
                for sname, svals in result["by_N"].items():
                    if (isinstance(svals, dict) and svals) or sname not in merged_by_N:
                        merged_by_N[sname] = svals
                result = {**existing, **result, "by_N": merged_by_N}
            if isinstance(result.get("by_steps"), dict) and isinstance(
                existing.get("by_steps"), dict
            ):
                merged_by_steps: dict = {**existing["by_steps"]}
                for sname, svals in result["by_steps"].items():
                    if (
                        isinstance(svals, dict) and svals
                    ) or sname not in merged_by_steps:
                        merged_by_steps[sname] = svals
                result = {**existing, **result, "by_steps": merged_by_steps}

        # tesseract_hashes: preserve peer-solver hashes from any prior run.
        if isinstance(result, dict) and new_tesseract_hashes:
            prior = result.get("tesseract_hashes") or {}
            if isinstance(existing, dict) and isinstance(
                existing.get("tesseract_hashes"), dict
            ):
                prior = {**existing["tesseract_hashes"], **prior}
            merged_hashes = (
                {**prior, **new_tesseract_hashes}
                if isinstance(prior, dict)
                else new_tesseract_hashes
            )
            result["tesseract_hashes"] = merged_hashes
        # harness_hash / harness_fn: latest write wins.
        if isinstance(result, dict) and new_harness_hash:
            result["harness_hash"] = new_harness_hash
            result["harness_fn"] = new_harness_fn_qualname

        if wall_time_s and isinstance(result, dict):
            prior = result.get("wall_time_s") or {}
            if isinstance(existing, dict) and isinstance(
                existing.get("wall_time_s"), dict
            ):
                prior = {**existing["wall_time_s"], **prior}
            result["wall_time_s"] = {**prior, **wall_time_s}

        if isinstance(result, dict) and "environment" not in result:
            result["environment"] = _environment_metadata()

        save_json(result, result_path)
        if "params" in result:
            save_json(result["params"], out_dir / "params.json")
        if csv_rows is not None:
            save_csv(csv_rows, out_dir / "result.csv")

    # Flock held > 60s is anomalous — solvers listed here are strong suspects
    # for whatever blocked the write.
    _flock_dt = time.monotonic() - _flock_t0
    if _flock_dt > 60:
        logging.warning(
            "save_experiment flock held for %.1fs at %s (solvers=%s)",
            _flock_dt,
            result_path,
            sorted(new_tesseract_hashes.keys()),
        )
