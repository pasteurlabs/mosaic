"""All filesystem I/O for the benchmark framework.

This module is the single home for every function that reads from or writes
to the filesystem. Suites, plots, and tests should never call ``open()``,
``json.load/dump``, ``np.load/savez``, ``Path.read_text/write_text``, or
``csv.DictWriter`` directly — go through the helpers here instead. The
exceptions are the few read-only paths inside Jupyter notebooks and other
ad-hoc analysis scripts.

Layout
------
1. **Paths** — :func:`results_dir`, :func:`experiment_dir`.
2. **Content hashing for staleness** — :func:`tesseract_content_hash`,
   :func:`harness_fn_hash`.
3. **Low-level primitives** — :class:`_NumpyEncoder`, :func:`save_json`,
   :func:`load_json`, :func:`try_load_json`, :func:`try_load_npz`,
   :func:`save_csv`. Atomic read-merge-write is serialised via
   :class:`filelock.FileLock` on a sibling ``.lock`` file.
4. **High-level experiment writers** — :func:`save_experiment` (result.json /
   params.json / result.csv with merge-aware partial-rerun handling),
   :func:`save_field_snapshots_npz` (atomic merge-save of per-solver npz).
5. **High-level experiment readers** — :func:`load_experiment_result` and
   :func:`load_field_snapshots_npz` (companion loaders for plots and tests).

The high-level writers handle locking and merge semantics so concurrent
mosaic processes writing to the same experiment directory can't race-clobber
each other's data. Plots use the readers so the on-disk schema lives in one
place.
"""

from __future__ import annotations

import ast
import contextlib
import csv
import fnmatch
import hashlib
import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np
from filelock import FileLock

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.console import print_saved

# ── Paths ────────────────────────────────────────────────────────────────────

RESULTS_DIR_ENV = "MOSAIC_RESULTS_DIR"


def results_dir() -> Path:
    """Return the root results directory, respecting env-var overrides.

    Priority:
      1. ``MOSAIC_RESULTS_DIR`` env var (set by ``--output-dir`` or the shell).
      2. ``Path.cwd() / "mosaic-results"`` otherwise.

    The legacy behaviour wrote inside the git tree (``mosaic/results``); that
    broke read-only installs and made the output location invisible to the
    caller.
    """
    if d := os.environ.get(RESULTS_DIR_ENV):
        return Path(d)
    return Path.cwd() / "mosaic-results"


def experiment_dir(
    results_dir: Path, problem: str, suite: str, experiment: str, suffix: str = ""
) -> Path:
    """Create and return ``results/<problem>/<suite>/<experiment><suffix>/``."""
    d = results_dir / problem / suite / f"{experiment}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Content hashing (for staleness) ─────────────────────────────────────────

# Files / directories excluded from the tesseract-source content hash.
# Build artefacts, caches, and lockfiles don't affect solver behaviour and
# would cause spurious staleness.
_HASH_EXCLUDE_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", "target", ".git", "node_modules", ".mypy_cache"}
)
_HASH_EXCLUDE_GLOBS = ("*.pyc", "*.lock", ".DS_Store")


def _py_source_no_comments(raw: bytes) -> bytes:
    """Return comment-stripped, normalised source for a Python file.

    Uses ``ast.unparse(ast.parse(...))`` which drops all comments and
    normalises whitespace. Falls back to raw bytes on ``SyntaxError`` so
    broken files are still hashed (differently from valid ones — correct).
    """
    try:
        src = raw.decode("utf-8")
        return ast.unparse(ast.parse(src)).encode("utf-8")
    except (SyntaxError, UnicodeDecodeError):
        return raw


def tesseract_content_hash(tesseract_dir: Path) -> str:
    """SHA-256 (first 16 hex chars) of a tesseract directory's source.

    Walks *tesseract_dir* recursively, sorts surviving files by POSIX path,
    and digests ``rel_path + NUL + file_bytes + NUL`` for each. Excludes
    build artefacts (``__pycache__``, ``target``, ``.pytest_cache``) and
    lockfiles so uninteresting churn doesn't flag results as stale.

    Python files are hashed after stripping comments (via ast.unparse) so
    that annotation-only edits (e.g. ``# mosaic:<category>`` tags) do not
    invalidate cached benchmark results.
    """
    tesseract_dir = Path(tesseract_dir)
    if not tesseract_dir.is_dir():
        return ""
    h = hashlib.sha256()
    paths: list[Path] = []
    for p in tesseract_dir.rglob("*"):
        if not p.is_file():
            continue
        parts = p.relative_to(tesseract_dir).parts
        if any(part in _HASH_EXCLUDE_PARTS for part in parts):
            continue
        if any(fnmatch.fnmatchcase(p.name, pat) for pat in _HASH_EXCLUDE_GLOBS):
            continue
        paths.append(p)
    paths.sort(key=lambda q: q.relative_to(tesseract_dir).as_posix())
    for p in paths:
        rel = p.relative_to(tesseract_dir).as_posix().encode()
        h.update(rel)
        h.update(b"\0")
        raw = p.read_bytes()
        data = _py_source_no_comments(raw) if p.suffix == ".py" else raw
        h.update(data)
        h.update(b"\0")
    return h.hexdigest()[:16]


def harness_fn_hash(fn) -> str:
    """SHA-256 (first 16 hex chars) of a function's normalised AST.

    Detects when the ``run_<experiment>`` function that produced a
    ``result.json`` has been edited since the result was saved. The hash is
    of an ``ast.dump`` with docstrings stripped, so that whitespace, comment,
    and docstring-only edits — which cannot affect runtime behaviour — do not
    invalidate previously-saved results. Behavioural edits (new statements,
    renamed locals, reordered expressions) still flip the hash because
    ``ast.dump`` preserves identifier names and statement order.

    Falls back to the raw-source SHA on ``SyntaxError`` so pathological
    sources (e.g. decorator-generated closures whose ``inspect.getsource``
    returns partial text) still yield a stable fingerprint rather than the
    empty string.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    # Strip leading docstring nodes from every scope that can carry one.
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body.pop(0)
    normalised = ast.dump(tree, annotate_fields=False, include_attributes=False)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


# ── Low-level primitives ────────────────────────────────────────────────────
#
# Atomic read-merge-write cycles are serialised via ``filelock.FileLock`` on a
# sibling ``.lock`` file. Two mosaic processes saving the same experiment
# directory would otherwise race: each reads the old ``by_solver``, each
# writes its own ``by_solver``, and whichever writes last silently drops the
# other's entries. Each call site below is responsible for ensuring the lock
# file's parent directory exists (typically by going through
# :func:`experiment_dir` or an explicit ``mkdir(parents=True)`` first).


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy / JAX scalars and arrays.

    Non-finite floats (``inf`` / ``-inf`` / ``nan``) are coerced to
    ``None``: Python's default writer emits the bare tokens ``Infinity``
    / ``NaN`` which are valid Python but **not** valid strict JSON, so
    downstream consumers (browsers, ``jq``, third-party SDKs) reject the
    file. ``None`` is a stable, lossy-but-safe representation that every
    JSON reader accepts. Callers that need the original numeric value
    should keep it on the NPZ side, not in ``result.json``.
    """

    def __init__(self, *args, **kwargs):
        # ``allow_nan=False`` makes the stdlib encoder raise ``ValueError``
        # on non-finite floats, which we intercept in ``iterencode`` to
        # emit ``null`` instead. Setting it at construction time keeps the
        # bare ``json.dumps(...)`` fallback strict too if someone reuses
        # this class outside ``save_json``.
        kwargs.setdefault("allow_nan", True)
        super().__init__(*args, **kwargs)

    def default(self, obj):
        if isinstance(obj, (np.ndarray, jax.Array)):
            # ``.tolist()`` produces native Python floats; the C-level
            # encoder handles those inline (never calls ``default``), so
            # walk the converted list and replace non-finite values now.
            return _strict_float_safe(obj.tolist())
        if isinstance(obj, (np.floating, np.integer)):
            val = obj.item()
            # ``np.float32(np.inf).item()`` → ``inf`` (Python float);
            # the parent encoder would emit ``Infinity``. Trap it here.
            if isinstance(val, float) and not math_isfinite(val):
                return None
            return val
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, set):
            return sorted(obj)
        # Callables / modules / classes flow into ``params`` when an
        # experiment is registered with non-JSON kwargs (e.g.
        # ``diagnostics={"div_rms": fn, ...}`` for physical_laws,
        # ``reference=<callable>`` for the agreement analytic). Stringify
        # them so ``params.json`` retains the metadata without raising.
        if callable(obj) or isinstance(obj, type):
            return f"<callable {getattr(obj, '__qualname__', repr(obj))}>"
        return super().default(obj)

    def iterencode(self, o, _one_shot: bool = False):
        # Intercept native Python floats before stdlib's tokeniser turns
        # ``inf`` / ``nan`` into ``Infinity`` / ``NaN``. ``default()`` is
        # never consulted for floats (they're handled inline by the C
        # encoder), so the trap has to live one level up.
        return super().iterencode(_strict_float_safe(o), _one_shot=_one_shot)


def _strict_float_safe(obj):
    """Recursively replace non-finite Python floats with ``None``."""
    if isinstance(obj, float) and not math_isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _strict_float_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        coerced = [_strict_float_safe(v) for v in obj]
        return type(obj)(coerced) if isinstance(obj, tuple) else coerced
    return obj


# Local alias so the encoder module isn't importing ``math`` at import time
# just for one check (keeps top-of-file imports stable).
from math import isfinite as math_isfinite  # noqa: E402


def save_json(data, path: str | Path) -> None:
    """Pretty-print ``data`` as JSON to ``path``; creates parent dirs.

    Serialises to a string first, then writes atomically via a ``.tmp``
    sibling + ``os.replace``. If the encoder raises (e.g. a kernel result
    contains an unencodable value), the destination is left untouched —
    avoiding the truncated-file failure mode where ``mosaic status``
    later reports ``unreadable result.json`` instead of the real error.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, cls=_NumpyEncoder, indent=2)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload)
    os.replace(tmp_path, path)
    print_saved(path)


def load_json(path: str | Path) -> dict:
    """Read JSON from ``path``. Raises on missing/malformed (use
    :func:`try_load_json` for the no-prior-state semantics)."""
    with open(path) as f:
        return json.load(f)


def try_load_json(path: str | Path) -> dict | None:
    """Parsed JSON or ``None`` if the file is missing or unparseable.

    Use when the caller treats a missing/corrupt file as "no prior state"
    rather than a hard error — avoids a try/except dance at every call site.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        return load_json(p)
    except Exception:
        return None


def try_load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """All arrays from an npz as a plain dict, ``{}`` on failure.

    Eagerly materialises every array into memory and closes the file before
    returning, so callers don't need to manage the ``np.load`` context.
    ``{}`` on missing / unreadable / parse error — same "no prior state"
    semantics as :func:`try_load_json`.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with np.load(p, allow_pickle=False) as f:
            return {k: np.asarray(f[k]) for k in f.files}
    except Exception:
        return {}


def save_npz_merged(
    path: str | Path,
    new_arrays: dict[str, np.ndarray],
    *,
    keep_old=None,
) -> None:
    """Atomically merge-save an npz file.

    Reads ``path`` (if it exists), retains any existing keys for which
    ``keep_old(key)`` returns ``True`` (``None`` keeps everything), then
    layers the caller's ``new_arrays`` on top (new wins on collision). The
    read-merge-write is serialised via a sibling ``.npz.lock`` flock so
    concurrent mosaic processes can't race-clobber each other.

    Suites with ad-hoc npz layouts (drag_opt profiles, flow_fields, rho_fields,
    …) should use this helper rather than calling ``np.savez`` directly so
    every artefact write goes through the same locking discipline. Suites
    with the canonical positional / flat per-solver layout should prefer
    :func:`save_field_snapshots_npz`, which adds the per-solver merge logic
    on top.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with FileLock(lock_path):
        old = try_load_npz(path)
        merged = (
            dict(old)
            if keep_old is None
            else {k: v for k, v in old.items() if keep_old(k)}
        )
        merged.update(new_arrays)
        np.savez(path, **merged)
    print_saved(path)


def save_csv(rows: list[dict], path: str | Path) -> None:
    """Write a list of dicts as CSV; creates parent dirs.

    Collects the *union* of keys across all rows so that partial rows (e.g.
    a solver whose ResourceSampler returned no stats) don't truncate the
    header and reject well-populated rows later. Preserves first-row
    ordering, then appends new keys in the order they appear.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print_saved(path)


# ── Field-snapshot npz writer ───────────────────────────────────────────────


class PartialResultWriter:
    """Process-safe writer for periodic ``result_partial.json`` dumps.

    Long-running per-solver kernels (drag_opt, recovery, …) want
    restartability: a checkpoint of partial progress should survive a
    crash or kill. Under the kernel pattern each solver lives on its own
    worker, so the in-memory ``by_solver`` dict is no longer shared —
    every checkpoint must acquire a :class:`FileLock`, re-read the
    on-disk JSON, merge in the current solver's entry, and re-write.

    Usage::

        writer = PartialResultWriter(
            out_dir,
            base_payload={"run_name": run_name, "params": run, ...},
        )
        # Inside the optimization loop, every N iterations:
        writer.write(solver_name, current_entry)

    ``write(name, None)`` is a no-op so kernels can pass through optional
    entries unconditionally.
    """

    def __init__(self, out_dir: Path, *, base_payload: dict | None = None) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._partial_path = out_dir / "result_partial.json"
        self._lock_path = out_dir / ".result_partial.lock"
        self._base_payload = dict(base_payload or {})

    def write(self, name: str, entry: dict | None) -> None:
        if entry is None:
            return
        with FileLock(self._lock_path):
            existing = try_load_json(self._partial_path) or {}
            by_solver = dict(existing.get("by_solver") or {})
            by_solver[name] = entry
            save_json(
                {**self._base_payload, "by_solver": by_solver}, self._partial_path
            )


def load_cached_field_snapshots(
    snap_path: Path | str,
    sweep_values: list,
    *,
    skip_solvers: set | dict | None = None,
    shared_prefixes: tuple[str, ...] = (),
) -> dict[Any, dict[str, np.ndarray]]:
    """Inverse of :func:`save_field_snapshots_npz` with ``flat_keys=True``.

    Reads ``snap_path`` (a ``fields.npz`` written under the flat
    ``{solver_name}_{idx}`` convention) and returns a mapping from each
    sweep value to ``{solver_name: array}`` for solvers cached in the
    NPZ. Used by partial-rerun support in the forward suite, where
    solvers not being re-executed this invocation still need to
    contribute to the cross-solver consensus.

    Parameters:
      * ``sweep_values`` — the in-memory list whose index maps to the
        ``_{idx}`` suffix on disk.
      * ``skip_solvers`` — solvers to omit (typically the ``tags`` dict
        keys, i.e. solvers being rebuilt this run). Either a set, or any
        mapping that supports ``in``.
      * ``shared_prefixes`` — key prefixes that mark *shared* (non-
        per-solver) arrays (mirrors ``save_field_snapshots_npz`` arg
        of the same name). Keys with these prefixes are filtered out.
    """
    snap = try_load_npz(snap_path)
    if not snap:
        return {}
    skip = skip_solvers or set()
    cached: dict = {}
    for ci, cv in enumerate(sweep_values):
        cached[cv] = {}
        sfx = f"_{ci}"
        for sfile, arr in snap.items():
            if not sfile.endswith(sfx):
                continue
            if shared_prefixes and any(sfile.startswith(p) for p in shared_prefixes):
                continue
            cn = sfile[: -len(sfx)]
            if cn not in skip:
                cached[cv][cn] = arr
    return cached


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
    ``solver_names`` string array. Legacy gradient-suite layout.

    **Flat** (``flat_keys=True``): per-solver arrays are written under keys
    of the form ``{solver_name}_{suffix}``. Used by the forward-agreement
    suite where plotting code reads keys directly by solver name. The
    ``prefixes`` argument is reinterpreted as "shared-key prefixes" in this
    mode (keys starting with any of these are treated as shared, not
    per-solver).

    Both layouts:
      1) Acquire the per-dir ``.save_experiment.lock`` so concurrent mosaic
         processes don't race — whichever writes last would otherwise drop
         the other's solvers.
      2) Load any existing npz, decode its per-solver entries back to solver
         names, and merge the caller's ``per_solver_arrays`` into them
         (caller wins on collision).
      3) Write the merged npz with a canonical ``solver_names`` ordering
         (caller's names first, then older solvers appended).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".save_experiment.lock"
    npz_path = out_dir / filename

    with FileLock(lock_path):
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
    # Longest first, so "ins_jl_diff" wins over "ins_jl".
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


# ── Experiment-result writer ────────────────────────────────────────────────


def _environment_metadata() -> dict:
    """Gather Python / mosaic version / timestamp metadata for provenance."""
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


# Keys that are scheduling/runtime only and should not gate merge behaviour.
_SCHEDULING_KEYS = {"gpu_ids"}


def _normalise_for_compare(v):
    """Recursively normalise a value for params equality comparison.

    JSON round-trips convert Python sets to sorted lists (via _NumpyEncoder).
    On a re-run the in-memory ``run`` dict may still carry sets (e.g.
    ``fine.solvers = {"exponax", "jax_cfd"}``), while the previously saved
    ``params.json`` carries ``["exponax", "jax_cfd"]``. Without
    normalisation the equality check always fails for such params, so the
    merge branch is never reached and existing solver data is overwritten.
    """
    if isinstance(v, set):
        return sorted(v)
    if isinstance(v, dict):
        return {k: _normalise_for_compare(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_normalise_for_compare(x) for x in v]
    return v


def _physics_params(p):
    """Strip scheduling-only keys and normalise for equality comparison."""
    if not isinstance(p, dict):
        return _normalise_for_compare(p)
    return _normalise_for_compare(
        {k: v for k, v in p.items() if k not in _SCHEDULING_KEYS}
    )


def _solvers_in_result(res: dict, known_solvers: set[str] | None = None) -> set[str]:
    """Collect solver names appearing in any known top-level map or in a
    custom schema (e.g. ``per_solver_spectra``, ``grad_norms``,
    ``landscape.by_solver``).
    """
    names: set[str] = set()
    for key in ("by_solver", "by_sweep", "by_N", "by_steps"):
        top = res.get(key)
        if isinstance(top, dict):
            names.update(str(k) for k in top)
    if isinstance(res.get("by_param"), dict):
        for solver_map in res["by_param"].values():
            if isinstance(solver_map, dict):
                names.update(str(k) for k in solver_map)

    if known_solvers:
        _scan_known_solvers(res, known_solvers, names, depth=2)
    return names


def _scan_known_solvers(
    node, known_solvers: set[str], names: set[str], depth: int
) -> None:
    """Recurse into ``node`` collecting dict keys that overlap ``known_solvers``."""
    if depth < 0 or not isinstance(node, dict):
        return
    keys_as_str = {str(k) for k in node}
    overlap = keys_as_str & known_solvers
    if overlap:
        names.update(overlap)
    for v in node.values():
        if isinstance(v, dict):
            _scan_known_solvers(v, known_solvers, names, depth - 1)


def _compute_tesseract_hashes(result: dict, cfg: Problem | None) -> dict[str, str]:
    """Compute fresh tesseract content hashes for solvers present in ``result``."""
    if not isinstance(result, dict) or cfg is None:
        return {}
    known_solvers = {s.name for s in cfg.solvers}
    hashes: dict[str, str] = {}
    for s in _solvers_in_result(result, known_solvers=known_solvers):
        try:
            spec = cfg.solver(s)
        except KeyError:
            continue
        tess_dir = cfg.tesseract_dir / spec.dir
        if tess_dir.is_dir():
            h = tesseract_content_hash(tess_dir)
            if h:
                hashes[s] = h
    return hashes


def _compute_harness_info(harness_fn) -> tuple[str, str]:
    """Return ``(harness_hash, module.qualname)`` for an optional harness fn."""
    if harness_fn is None:
        return "", ""
    h = harness_fn_hash(harness_fn)
    mod = getattr(harness_fn, "__module__", "") or ""
    qual = getattr(harness_fn, "__qualname__", "") or getattr(
        harness_fn, "__name__", ""
    )
    qualname = f"{mod}.{qual}" if mod else qual
    return h, qualname


def _merge_by_param(existing: dict, result: dict) -> dict:
    """Merge ``by_param`` (and accompanying ``spread``) from result into existing."""
    merged_by_param: dict = {str(k): v for k, v in existing["by_param"].items()}
    for pval, solver_map in result["by_param"].items():
        spval = str(pval)
        if (
            spval in merged_by_param
            and isinstance(merged_by_param[spval], dict)
            and isinstance(solver_map, dict)
        ):
            merged_by_param[spval] = {**merged_by_param[spval], **solver_map}
        else:
            merged_by_param[spval] = solver_map
    merged_spread: dict = {str(k): v for k, v in existing.get("spread", {}).items()}
    merged_spread.update({str(k): v for k, v in result.get("spread", {}).items()})
    return {
        **existing,
        **result,
        "by_param": merged_by_param,
        "spread": merged_spread,
    }


def _merge_by_size(existing: dict, result: dict, key: str) -> dict:
    """Per-solver merge for ``by_N`` / ``by_steps`` — keep prior non-empty entries.

    Cost suites pre-populate every solver key (empty dict for solvers that
    failed), so a naive outer-merge would overwrite a successful prior run's
    per-size entries with empty dicts. Prefer the new run's non-empty entries,
    fall back to existing data when the new run produced an empty dict.
    """
    merged: dict = {**existing[key]}
    for sname, svals in result[key].items():
        if (isinstance(svals, dict) and svals) or sname not in merged:
            merged[sname] = svals
    return {**existing, **result, key: merged}


def _merge_with_existing(result: dict, existing: dict) -> dict:
    """Merge a parameter-compatible existing result into ``result``.

    Dispatches across the supported schemas (``by_solver``, ``by_param``,
    ``by_sweep``) for the primary merge, then layers in ``by_N`` / ``by_steps``
    per-solver merges. Caller wins on collision.
    """
    if isinstance(result.get("by_solver"), dict) and isinstance(
        existing.get("by_solver"), dict
    ):
        merged_by_solver = {**existing["by_solver"], **result["by_solver"]}
        result = {**existing, **result, "by_solver": merged_by_solver}
    elif isinstance(result.get("by_param"), dict) and isinstance(
        existing.get("by_param"), dict
    ):
        result = _merge_by_param(existing, result)
    elif isinstance(result.get("by_sweep"), dict) and isinstance(
        existing.get("by_sweep"), dict
    ):
        merged_by_sweep = {**existing["by_sweep"], **result["by_sweep"]}
        result = {**existing, **result, "by_sweep": merged_by_sweep}

    if isinstance(result.get("by_N"), dict) and isinstance(existing.get("by_N"), dict):
        result = _merge_by_size(existing, result, "by_N")
    if isinstance(result.get("by_steps"), dict) and isinstance(
        existing.get("by_steps"), dict
    ):
        result = _merge_by_size(existing, result, "by_steps")
    return result


def _stamp_tesseract_hashes(
    result: dict, existing: dict | None, new_hashes: dict[str, str]
) -> None:
    """In-place: merge peer-solver hashes from prior runs with the new ones."""
    if not (isinstance(result, dict) and new_hashes):
        return
    prior = result.get("tesseract_hashes") or {}
    if isinstance(existing, dict) and isinstance(
        existing.get("tesseract_hashes"), dict
    ):
        prior = {**existing["tesseract_hashes"], **prior}
    merged_hashes = {**prior, **new_hashes} if isinstance(prior, dict) else new_hashes
    result["tesseract_hashes"] = merged_hashes


def _stamp_wall_time(
    result: dict, existing: dict | None, wall_time_s: dict[str, float] | None
) -> None:
    """In-place: merge wall-time entries from prior runs with the new ones."""
    if not (wall_time_s and isinstance(result, dict)):
        return
    prior = result.get("wall_time_s") or {}
    if isinstance(existing, dict) and isinstance(existing.get("wall_time_s"), dict):
        prior = {**existing["wall_time_s"], **prior}
    result["wall_time_s"] = {**prior, **wall_time_s}


def _write_artifacts(
    result: dict,
    result_path: Path,
    out_dir: Path,
    csv_rows: list[dict] | None,
) -> None:
    """Write result.json, params.json (if present), and result.csv (if given)."""
    save_json(result, result_path)
    if "params" in result:
        save_json(result["params"], out_dir / "params.json")
    if csv_rows is not None:
        save_csv(csv_rows, out_dir / "result.csv")


def save_experiment(
    result: dict,
    out_dir: Path,
    csv_rows: list[dict] | None = None,
    cfg: Problem | None = None,
    harness_fn=None,
    wall_time_s: dict[str, float] | None = None,
) -> None:
    """Save ``result.json``, ``params.json``, and optionally ``result.csv``.

    If an existing ``result.json`` already has a ``by_solver`` dict and the
    new result shares the same *physics* ``params`` (parameter-compatible
    run), merges the new ``by_solver`` entries into the existing ones —
    preserving data from solvers that were NOT part of the current run. This
    prevents a single-solver rerun from silently wiping out previously
    benchmarked solvers. If params differ, the new result overwrites
    wholesale (old data is no longer valid).

    Runtime/scheduling-only keys (e.g. ``gpu_ids``) are excluded from the
    equality check — they do not affect physics or solver output.

    Staleness stamps (optional): when *cfg* is supplied, a
    ``tesseract_hashes`` dict ``{solver_name: content-hash}`` is injected for
    every solver present in the result. When *harness_fn* is supplied, the
    result is also stamped with ``harness_hash`` (SHA of the function's
    source) and ``harness_fn`` (``module.qualname``). ``mosaic status`` uses
    these to flip cells to the ``*`` (stale) annotation when the current
    source differs from what produced the saved result.
    """
    new_tesseract_hashes = _compute_tesseract_hashes(result, cfg)
    new_harness_hash, new_harness_fn_qualname = _compute_harness_info(harness_fn)

    result_path = out_dir / "result.json"
    lock_path = out_dir / ".save_experiment.lock"
    _flock_t0 = time.monotonic()
    with FileLock(lock_path):
        existing = try_load_json(result_path) if isinstance(result, dict) else None
        if isinstance(existing, dict) and _physics_params(
            existing.get("params")
        ) == _physics_params(result.get("params")):
            result = _merge_with_existing(result, existing)

        _stamp_tesseract_hashes(result, existing, new_tesseract_hashes)
        # harness_hash / harness_fn: latest write wins.
        if isinstance(result, dict) and new_harness_hash:
            result["harness_hash"] = new_harness_hash
            result["harness_fn"] = new_harness_fn_qualname

        _stamp_wall_time(result, existing, wall_time_s)

        if isinstance(result, dict) and "environment" not in result:
            result["environment"] = _environment_metadata()

        # CI provenance — latest write always wins. Same fields as the
        # main-side `_run_meta` shipped in commit 822cbdd; ported here
        # because the I/O code moved from utils.py to io.py in the refactor.
        if isinstance(result, dict):
            import datetime as _dt
            import platform as _platform

            result["_run_meta"] = {
                "timestamp": _dt.datetime.now(_dt.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "ci": bool(os.environ.get("CI")),
                "runner": os.environ.get("RUNNER_NAME", ""),
                "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
                "github_sha": os.environ.get("GITHUB_SHA", ""),
                "platform": _platform.platform(),
            }

        _write_artifacts(result, result_path, out_dir, csv_rows)

    # Flock held > 60 s is anomalous — solvers listed here are strong suspects
    # for whatever blocked the write.
    _flock_dt = time.monotonic() - _flock_t0
    if _flock_dt > 60:
        logging.warning(
            "save_experiment flock held for %.1fs at %s (solvers=%s)",
            _flock_dt,
            result_path,
            sorted(new_tesseract_hashes.keys()),
        )


def save_harness_result(
    result: dict,
    *,
    cfg: Problem,
    suite: str,
    exp_subdir: str,
    harness_fn,
    wall_time_s: dict[str, float] | None = None,
    csv_rows: list[dict] | None = None,
    debug: bool = False,
) -> Path:
    """Compute the standard experiment directory and save a harness result.

    Wraps the boilerplate every harness uses to land its output: builds
    ``results/<problem>/<suite>/<exp_subdir>[_debug]/`` via
    :func:`experiment_dir`, then delegates to :func:`save_experiment` with the
    canonical ``cfg=`` / ``harness_fn=`` / ``wall_time_s=`` trailer that
    drives the staleness hashing.

    Returns the resolved ``out_dir`` so callers can layer additional artefacts
    (e.g. a per-solver npz via :func:`save_field_snapshots_npz`) into the
    same directory without re-deriving the path.
    """
    out_dir = experiment_dir(
        results_dir(),
        cfg.name,
        suite,
        exp_subdir,
        suffix="_debug" if debug else "",
    )
    save_experiment(
        result,
        out_dir,
        csv_rows=csv_rows,
        cfg=cfg,
        harness_fn=harness_fn,
        wall_time_s=wall_time_s,
    )
    return out_dir


# ── Experiment-result readers ───────────────────────────────────────────────


def load_experiment_result(out_dir: Path | str) -> dict | None:
    """Read ``out_dir/result.json`` and return its parsed contents.

    Returns ``None`` if the file is missing or unparseable — same "no prior
    state" semantics as :func:`try_load_json`. Use this from plots and tests
    instead of opening ``result.json`` directly so the on-disk schema lives
    in one place.
    """
    return try_load_json(Path(out_dir) / "result.json")


def load_field_snapshots_npz(
    out_dir: Path | str, filename: str = "gradient_fields.npz"
) -> dict[str, np.ndarray]:
    """Read ``out_dir/filename`` and return all arrays as a plain dict.

    Companion reader for :func:`save_field_snapshots_npz`. Returns ``{}`` if
    the file is missing or unreadable. The returned dict is fully
    materialised — the underlying npz file is closed before this returns, so
    callers don't need to manage the load context.
    """
    return try_load_npz(Path(out_dir) / filename)
