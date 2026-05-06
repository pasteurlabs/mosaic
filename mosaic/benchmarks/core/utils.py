"""JSON helpers, array math, and small algorithm utilities."""

from __future__ import annotations

import ast
import contextlib
import csv
import fcntl
import fnmatch
import hashlib
import inspect
import json
import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .config import ProblemConfig
from .console import print_saved

# ── Results directory resolution ─────────────────────────────────────────────
#
# Priority:
#   1. MOSAIC_RESULTS_DIR env var  (set by --output-dir or the user's shell)
#   2. Current working directory / "mosaic-results"
#
# The old behaviour wrote to ``Path(__file__).parent.parent / "results"``
# (inside the git tree).  That breaks read-only installs and makes the output
# location invisible to the caller.

_RESULTS_DIR_ENV = "MOSAIC_RESULTS_DIR"


def results_dir() -> Path:
    """Return the root results directory, respecting env-var overrides."""
    if d := os.environ.get(_RESULTS_DIR_ENV):
        return Path(d)
    return Path.cwd() / "mosaic-results"


# Staleness detection: files/directories excluded from the tesseract-source
# content hash. Build artefacts, caches, and lockfiles don't affect the
# solver's behaviour and would cause spurious staleness.
_HASH_EXCLUDE_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", "target", ".git", "node_modules", ".mypy_cache"}
)
_HASH_EXCLUDE_GLOBS = ("*.pyc", "*.lock", ".DS_Store")


def _py_source_no_comments(raw: bytes) -> bytes:
    """Return comment-stripped, normalised source for a Python file.

    Uses ast.unparse(ast.parse(...)) which drops all comments and normalises
    whitespace.  Falls back to raw bytes on SyntaxError so broken files are
    still hashed (differently from valid ones, which is correct).
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

    Used to detect when the ``run_<experiment>`` function that produced a
    result.json has been edited since the result was saved. We hash an
    ``ast.dump`` of the source with docstrings stripped, rather than the raw
    source bytes, so that whitespace, comment, and docstring-only edits — which
    cannot affect runtime behaviour — do not invalidate previously-saved
    results. Behavioural edits (new statements, renamed locals, reordered
    expressions) still flip the hash because ``ast.dump`` preserves identifier
    names and statement order.

    Falls back to the raw-source SHA on ``SyntaxError`` so pathological sources
    (e.g. decorator-generated closures whose ``inspect.getsource`` returns
    partial text) still yield a stable fingerprint rather than the empty
    string.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    # Strip leading docstring Expr(Constant(str)) nodes from every scope that
    # can carry one — Module, FunctionDef, AsyncFunctionDef, ClassDef.
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


@contextlib.contextmanager
def _file_lock(lock_path: Path):
    """Advisory flock on ``lock_path`` — used to serialize concurrent
    ``save_experiment`` invocations so the read-merge-write cycle is atomic.

    Two mosaic processes saving the same experiment directory would otherwise
    race: each reads the old ``by_solver``, each writes its own ``by_solver``,
    and whichever writes last silently drops the other's entries. The lock
    file is created alongside the result (not the result itself, so we never
    interfere with the JSON write).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def save_gradient_fields_npz(
    out_dir: Path,
    solver_names: list[str],
    per_solver_arrays: dict[str, dict[str, np.ndarray]],
    shared_arrays: dict[str, np.ndarray] | None = None,
    filename: str = "gradient_fields.npz",
    prefixes: tuple[str, ...] = ("grad",),
) -> None:
    """Atomically merge-save a positional-indexed per-solver npz.

    Our npz convention stores a ``solver_names`` string array and per-solver
    arrays under positional keys like ``{prefix}_{j}`` or
    ``{prefix}_{j}_<suffix>`` where ``j`` is the index of the solver in
    ``solver_names``. Two concurrent mosaic processes writing the same npz
    without coordination would race: whichever writes last wins wholesale,
    dropping the other process's solvers.

    This helper:
      1) Acquires the same per-dir ``.save_experiment.lock`` used by
         ``save_experiment`` so npz and json writes share a critical section.
      2) Loads any existing npz, decodes its ``solver_names`` and remaps its
         per-solver entries back to solver names.
      3) Merges the caller's ``per_solver_arrays`` with the existing entries
         (caller wins on collision, matching by_solver merge semantics).
      4) Writes the merged npz with a new canonical ``solver_names`` ordering
         (caller's names first, then older solvers appended) and re-emits
         the positional keys.

    ``per_solver_arrays``: ``{solver_name: {"<prefix>:<suffix>": array, ...}}``.
        The key has the form ``"<prefix>:<suffix>"`` where ``<prefix>`` is one
        of ``prefixes`` and ``<suffix>`` is whatever appears after
        ``{prefix}_{j}_`` in the npz key (e.g. ``"N32"``). Use ``suffix=""``
        for the plain ``{prefix}_{j}`` layout. For single-prefix use, the
        convenience form ``{solver: {suffix: array}}`` (no colon, implicit
        first prefix) is also accepted.

    ``shared_arrays``: optional ``{key: array}`` — written verbatim alongside
        (``ic``, ``N_values``, ``steps``…). Caller wins on collision.

    ``prefixes``: tuple of positional-key prefixes to participate in merging
        (e.g. ``("grad",)`` for gradient.py, ``("rho_final", "rho_history")``
        for recovery.py topopt). Non-matching keys are preserved as shared.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".save_experiment.lock"
    npz_path = out_dir / filename

    def _parse_positional(key: str) -> tuple[str, int, str] | None:
        """Return (prefix, j, suffix) if key matches any of ``prefixes``."""
        for p in prefixes:
            if key == p or key.startswith(p + "_"):
                rest = key[len(p) :]
                if not rest:
                    return None
                if rest[0] != "_":
                    continue
                parts = rest[1:].split("_", 1)
                try:
                    j = int(parts[0])
                except ValueError:
                    continue
                suffix = parts[1] if len(parts) > 1 else ""
                return (p, j, suffix)
        return None

    def _canonicalise_input(
        per_solver: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[tuple[str, str], np.ndarray]]:
        """Convert caller's keys to (prefix, suffix) tuples."""
        out: dict[str, dict[tuple[str, str], np.ndarray]] = {}
        default_prefix = prefixes[0]
        for sname, suf_map in per_solver.items():
            norm: dict[tuple[str, str], np.ndarray] = {}
            for k, arr in suf_map.items():
                if ":" in k:
                    p, s = k.split(":", 1)
                else:
                    p, s = default_prefix, k
                norm[(p, s)] = np.asarray(arr)
            out[sname] = norm
        return out

    with _file_lock(lock_path):
        merged_per_solver: dict[str, dict[tuple[str, str], np.ndarray]] = {}
        merged_shared: dict[str, np.ndarray] = {}
        if npz_path.exists():
            try:
                with np.load(npz_path, allow_pickle=True) as _old:
                    old_names_arr = _old.get("solver_names")
                    old_names: list[str] = (
                        [str(n) for n in list(old_names_arr)]
                        if old_names_arr is not None
                        else []
                    )
                    for k in _old.files:
                        if k == "solver_names":
                            continue
                        parsed = _parse_positional(k)
                        if parsed is None:
                            merged_shared[k] = np.asarray(_old[k])
                            continue
                        p, j, suffix = parsed
                        if j < 0 or j >= len(old_names):
                            merged_shared[k] = np.asarray(_old[k])
                            continue
                        sname = old_names[j]
                        merged_per_solver.setdefault(sname, {})[(p, suffix)] = (
                            np.asarray(_old[k])
                        )
            except Exception:
                merged_per_solver = {}
                merged_shared = {}

        new_per_solver = _canonicalise_input(per_solver_arrays)
        for sname, suf_map in new_per_solver.items():
            merged_per_solver.setdefault(sname, {}).update(suf_map)
        if shared_arrays:
            merged_shared.update(shared_arrays)

        ordered: list[str] = list(solver_names)
        for sname in merged_per_solver:
            if sname not in ordered:
                ordered.append(sname)

        gsnap: dict[str, np.ndarray] = {"solver_names": np.array(ordered)}
        gsnap.update(merged_shared)
        for j, sname in enumerate(ordered):
            suf_map = merged_per_solver.get(sname, {})
            for (p, suffix), arr in suf_map.items():
                key = f"{p}_{j}" if suffix == "" else f"{p}_{j}_{suffix}"
                gsnap[key] = arr
        np.savez(npz_path, **gsnap)


# ── JSON serialization ────────────────────────────────────────────────────────


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, jax.Array)):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


def save_json(data, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=_NumpyEncoder, indent=2)
    print_saved(path)


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Collect the UNION of keys across all rows so that partial rows (e.g. a
    # solver whose ResourceSampler silently returned no stats) don't truncate
    # the header and cause DictWriter to reject well-populated rows that come
    # later.  Preserves first-row ordering, then appends new keys in the order
    # they appear.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print_saved(path)


# ── Array statistics ──────────────────────────────────────────────────────────


def trimmed_mean(arrays: list, q_lo: float = 0.05, q_hi: float = 0.95) -> jax.Array:
    """Per-element trimmed mean across a list of same-shape arrays.

    With n <= 2 arrays, quantile linear interpolation places lo strictly above
    the minimum and hi strictly below the maximum, so both values fail the mask
    and the result collapses to zero.  For small n we fall back to the plain mean.
    """
    stacked = jnp.stack(arrays, axis=0)  # (n, ...)
    if len(arrays) <= 2:
        return stacked.mean(axis=0)
    lo = jnp.quantile(stacked, q_lo, axis=0)
    hi = jnp.quantile(stacked, q_hi, axis=0)
    mask = (stacked >= lo) & (stacked <= hi)
    count = jnp.maximum(mask.sum(axis=0), 1)
    return (stacked * mask).sum(axis=0) / count


def l2_error_rel(pred, ref) -> float:
    """Relative L2 error ||pred-ref|| / ||ref||."""
    return float(
        jnp.sqrt(jnp.mean((pred - ref) ** 2)) / (jnp.sqrt(jnp.mean(ref**2)) + 1e-30)
    )


def is_valid(arr) -> bool:
    return arr is not None and bool(jnp.all(jnp.isfinite(arr)))


# ── Misc ──────────────────────────────────────────────────────────────────────


# Keys consumed by suite runners — never forwarded to make_ic / make_inputs.
# Used only by run_diagnostics which retains the legacy flat-dict config format.
SUITE_KEYS: frozenset = frozenset(
    {
        "N_values",
        "ic",
        "seed",
        "sweep_key",
        "sweep_values",
        "fine_solvers",
        "fine_dt",
        "fine_steps",
        "chunk_steps",
        "n_chunks",
        "fine_chunk",
        "n_trials",
        "steps_values",
        "ics",
        "eps_values",
        "n_dirs",
        "horizons",
        "perturb_sigma",
        "lr",
        "max_iters",
        "patience",
        "failure_threshold",
        "v_frac",
        "penalty_weight",
        "x_min",
        "compliance_key",
        "debug",
        "ic_names",
        "n_alphas",
        "alpha_range",
        "k_svd",
    }
)


def physics_params(p: dict, extra: frozenset = frozenset()) -> dict:
    """Physics kwargs: everything in p that is not a suite bookkeeping key.

    Used only by run_diagnostics (legacy flat-dict config). All other suite
    functions use iter_runs + named sub-dicts instead.
    """
    return {k: v for k, v in p.items() if k not in SUITE_KEYS | extra}


def extract_runs(exp_def: "list[dict] | dict") -> "list[dict]":
    """Return the runs list from an experiment def (wrapper dict or bare list)."""
    if isinstance(exp_def, dict):
        return exp_def.get("runs", [])
    return exp_def if exp_def is not None else []


def iter_runs(runs: "list[dict] | dict", cli_overrides: dict):
    """Yield run configs from an experiment def, applying debug and IC filter.

    Accepts either:
    - A list[dict] of run configs (bare list form).
    - A wrapper dict with a ``runs`` key (experiment wrapper form):
        dict(description=..., plot_description=..., runs=[...])

    Each run dict has named sub-groups: ic, physics, sweep, fd, optim, jacobian,
    fine, cost.  Yields a deep-copied, debug-adjusted run per matching entry.

    cli_overrides keys consumed here (read-only, never mutated):
        ic_names   — list[str]: filter to runs whose ic.name is in the list
        run_names  — list[str]: filter to runs whose name field is in the list
        debug      — bool: cap physics sizes and truncate sweep lists
    """
    import copy

    if isinstance(runs, dict):
        runs = runs.get("runs", [])

    ic_filter = cli_overrides.get("ic_names")
    run_filter = cli_overrides.get("run_names")
    debug = cli_overrides.get("debug", False)

    for run in runs:
        ic_name = run.get("ic", {}).get("name", "")
        if ic_filter and ic_name not in ic_filter:
            continue
        run_name = run.get("name", ic_name)
        if run_filter and run_name not in run_filter:
            continue
        run = copy.deepcopy(run)
        if debug:
            _debug_run(run)
        yield run


def _debug_run(run: dict) -> None:
    """Reduce run complexity for debug mode (mutates run in place)."""
    phys = run.setdefault("physics", {})
    for key, cap in [
        ("N", 16),
        ("nx", 8),
        ("steps", 5),
        ("chunk_steps", 5),
        ("n_chunks", 2),
    ]:
        if key in phys:
            phys[key] = min(phys[key], cap)
    # Keep ny consistent with capped nx (canonical 2:1 aspect ratio)
    if "nx" in phys and "ny" in phys:
        phys["ny"] = min(phys["ny"], max(1, phys["nx"] // 2))
    fd = run.setdefault("fd", {})
    for key, cap in [("n_dirs", 2), ("n_alphas", 5)]:
        if key in fd:
            fd[key] = min(fd[key], cap)
    if "eps_values" in fd:
        fd["eps_values"] = list(fd["eps_values"])[:2]
    sweep = run.get("sweep", {})
    if "values" in sweep:
        vals = list(sweep["values"])
        # For resolution (N) sweeps, skip pathologically coarse values where
        # all FEM solvers return identical results (trivial DOF count) or the
        # reference field is near-zero giving astronomical relative errors.
        if sweep.get("key") == "N":
            sane = [v for v in vals if v >= 4]
            if sane:
                vals = sane
        sweep["values"] = vals[:2]
    optim = run.get("optim", {})
    for key, cap in [("max_iters", 50), ("patience", 10)]:
        if key in optim:
            optim[key] = min(optim[key], cap)
    cost = run.get("cost", {})
    for key in ("N_values", "steps_values"):
        if key in cost:
            cost[key] = list(cost[key])[:2]
    if "n_trials" in cost:
        cost["n_trials"] = 1


def _has_vjp(spec) -> bool:
    """Return True if the solver's Docker image exposes vector_jacobian_product.

    Respects the explicit ``spec.differentiable`` flag when set (avoids slow
    Docker container startup for known-differentiable solvers).
    """
    explicit = getattr(spec, "differentiable", None)
    if explicit is not None:
        return bool(explicit)
    from tesseract_core import Tesseract

    tag = spec.image_tag
    if not tag:
        return False
    try:
        with Tesseract.from_image(tag) as t:
            return "vector_jacobian_product" in t.available_endpoints
    except Exception:
        return False


def exclusion_candidate_keys(
    suite: str, experiment: str | None = None, sub: str | None = None
) -> tuple[str, ...]:
    """Return the ordered candidate exclusion keys for a (suite, experiment[, sub]).

    Ordered MOST-SPECIFIC-FIRST so a lookup that breaks on first match reflects
    the narrowest scope that the config declared. Example, for
    ``suite="optimization"``, ``experiment="drag_opt"``::

        ("optimization/drag_opt", "drag_opt", "optimization")

    The bare-experiment form (``"drag_opt"``) is kept for backward compatibility
    with existing configs that hadn't adopted the ``suite/experiment`` convention
    yet. New configs should prefer the fully-qualified key.

    Sub-experiment (e.g. ``"agreement/tgv"``) is split and considered at two
    levels of granularity: the full ``experiment/sub`` label and the leading
    ``experiment`` token. This mirrors :func:`status._lookup_check` so config
    keys like ``"forward/agreement"`` work whether the experiment directory is
    ``agreement`` or ``agreement/tgv``.
    """
    # Split a sub-dir out of experiment if the caller passed one inline (e.g.
    # ``experiment="agreement/tgv"`` with sub=None).
    if experiment and sub is None and "/" in experiment:
        experiment, sub = experiment.split("/", 1)

    keys: list[str] = []
    if experiment and sub:
        keys.append(f"{suite}/{experiment}/{sub}")
        keys.append(f"{experiment}/{sub}")
    if experiment:
        keys.append(f"{suite}/{experiment}")
        # Bare experiment key — legacy config form (kept for backward compat).
        keys.append(experiment)
    keys.append(suite)
    # De-duplicate while preserving order (suite == experiment edge case).
    seen: set[str] = set()
    unique: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return tuple(unique)


def exclusion_lookup(
    exclusions: dict,
    suite: str,
    experiment: str | None = None,
    sub: str | None = None,
) -> tuple[str, object] | None:
    """Look up the most-specific exclusion entry for (suite, experiment[, sub]).

    Returns ``(matched_key, value)`` if any candidate key is present in
    ``exclusions``, else ``None``. The value is whatever the config stored —
    typically a ``{"category": ..., "reason": ...}`` dict, but legacy string
    exclusions are also accepted.

    Used by both :func:`active_solvers` (runtime gating) and
    ``core.status.collect_status`` (display) so the two paths can't drift on
    which exclusion key takes precedence.
    """
    if not exclusions:
        return None
    for key in exclusion_candidate_keys(suite, experiment, sub):
        if key in exclusions:
            return key, exclusions[key]
    return None


def active_solvers(
    cfg: ProblemConfig, suite: str, experiment: str | None = None
) -> list[str]:
    """Solver names not excluded for *suite* (and optionally *experiment*).

    Checks ``spec.exclusions`` via :func:`exclusion_lookup`, which tries the
    most-specific key first (``"suite/experiment"``) and falls back to the
    bare experiment key and finally to ``suite``. This matches the lookup
    used by ``mosaic status`` so the runtime filter and the status display
    can never disagree on which cells are gated.

    Prints a one-line warning per excluded solver, naming the matched key so
    the exclusion source is visible in runner output.
    """
    from .console import console

    result = []
    for name, spec in cfg.solvers.items():
        match = exclusion_lookup(spec.exclusions, suite, experiment)
        if match is None:
            result.append(name)
        else:
            matched_key, reason = match
            console.print(
                f"  [yellow]SKIP {name}[/] excluded from {matched_key!r}: {reason}"
            )
    return result


def _diff_solvers(
    cfg: ProblemConfig, suite: str = "gradient", experiment: str | None = None
) -> list[str]:
    """Differentiable solvers not excluded for *suite* (and optionally *experiment*)."""
    return [
        name
        for name in active_solvers(cfg, suite, experiment)
        if _has_vjp(cfg.solvers[name])
    ]


def experiment_dir(
    results_dir: Path, problem: str, suite: str, experiment: str, suffix: str = ""
) -> Path:
    """Create and return results/<problem>/<suite>/<experiment><suffix>/."""
    d = results_dir / problem / suite / f"{experiment}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    return d


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

    # ── staleness stamps: compute here, merge with existing inside the lock ─
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

        # Generic custom-schema pass: keyed off the registered solver names so
        # we don't have to enumerate every custom harness output shape
        # (per_solver_*, grad_norms, landscape.by_solver, …).
        if known_solvers:

            def _scan(node, depth: int) -> None:
                if depth < 0 or not isinstance(node, dict):
                    return
                keys_as_str = {str(k) for k in node.keys()}
                if keys_as_str & known_solvers:
                    names.update(keys_as_str & known_solvers)
                for v in node.values():
                    if isinstance(v, dict):
                        _scan(v, depth - 1)

            # Depth 2 covers top-level dicts and one nested level
            # (e.g. landscape → by_solver → {solver_name: …}).
            _scan(res, depth=2)
        return names

    new_tesseract_hashes: dict[str, str] = {}
    if isinstance(result, dict) and cfg is not None:
        known_solvers = {str(k) for k in cfg.solvers.keys()}
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
    # Serialize the read-merge-write cycle across concurrent mosaic processes
    # writing to the same experiment dir. Without this lock, two solvers
    # finishing near-simultaneously can each read the old by_solver and each
    # write their own, silently dropping entries. See cycle-18 feedback.
    lock_path = out_dir / ".save_experiment.lock"
    # Track flock hold time — a healthy save_experiment critical section is
    # milliseconds; > 60s indicates a solver that wedged inside its apply() call.
    _flock_t0 = time.monotonic()
    with _file_lock(lock_path):
        existing = None
        if isinstance(result, dict) and result_path.exists():
            try:
                existing = load_json(result_path)
            except Exception:
                existing = None
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
                # Structure: {param_val: {solver_name: {...}, ...}, ...}
                # Merge per-solver entries within each param value so a
                # single-solver rerun does not overwrite other solvers' data.
                elif isinstance(result.get("by_param"), dict) and isinstance(
                    existing.get("by_param"), dict
                ):
                    # Normalise keys to str so JSON-round-tripped str keys and
                    # fresh int keys don't coexist as duplicate entries.
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
                    # Also merge the spread dict (per-param scalar), normalising
                    # keys to str for the same reason.
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
                # Structure: {solver_name: {sweep_key: {...}, ...}, ...}
                # Same per-solver shape as by_solver: preserve existing solvers'
                # entries so a single-solver Stage 3 rerun does not overwrite
                # peer solvers' sweep data.
                elif isinstance(result.get("by_sweep"), dict) and isinstance(
                    existing.get("by_sweep"), dict
                ):
                    merged_by_sweep = {
                        **existing["by_sweep"],
                        **result["by_sweep"],
                    }
                    result = {**existing, **result, "by_sweep": merged_by_sweep}

                # ── by_N / by_steps merge (cost suite) ────────────────────────
                # Structure: {solver_name: {size_value: {mean, std, ...}}}.
                # Cost suites (spatial_cost / temporal_cost / vjp_cost) always
                # pre-populate every solver key (empty dict for solvers that
                # failed), so a naive outer-merge {**existing, **new} would
                # overwrite a successful prior run's per-size entries with
                # this run's empty dicts. Per-solver merge: prefer the new
                # run's non-empty entries, fall back to existing data when
                # the new run produced an empty dict for that solver.
                # Without this merge, any run where all solvers failed
                # silently wiped out previously valid data.
                if isinstance(result.get("by_N"), dict) and isinstance(
                    existing.get("by_N"), dict
                ):
                    merged_by_N: dict = {**existing["by_N"]}
                    for sname, svals in result["by_N"].items():
                        if isinstance(svals, dict) and svals:
                            merged_by_N[sname] = svals
                        elif sname not in merged_by_N:
                            # new solver with empty data — record empty so
                            # status surfaces the failure, but do not clobber
                            # peer entries. Equivalent to the old behaviour
                            # for first-time runs.
                            merged_by_N[sname] = svals
                    result = {**existing, **result, "by_N": merged_by_N}
                if isinstance(result.get("by_steps"), dict) and isinstance(
                    existing.get("by_steps"), dict
                ):
                    merged_by_steps: dict = {**existing["by_steps"]}
                    for sname, svals in result["by_steps"].items():
                        if isinstance(svals, dict) and svals:
                            merged_by_steps[sname] = svals
                        elif sname not in merged_by_steps:
                            merged_by_steps[sname] = svals
                    result = {**existing, **result, "by_steps": merged_by_steps}

        # ── staleness stamps: merge with whatever we just loaded / merged ──
        # tesseract_hashes: preserve peer-solver hashes from any prior run.
        # Include existing hashes even when no merge branch was taken (e.g.
        # jacobian_svd uses per_solver_spectra — without this seed a partial
        # rerun drops all peer hashes).
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
        # harness_hash / harness_fn: latest write wins (the function that
        # actually produced this save call is the authoritative one).
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
