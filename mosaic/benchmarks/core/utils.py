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
import os
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


def try_load_json(path: str | Path) -> dict | None:
    """Return parsed JSON, or ``None`` if the file is missing or unparseable.

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
    """Return all arrays from an npz as a plain dict, or ``{}`` on failure.

    Eagerly materialises every array into memory and closes the file before
    returning, so callers don't need to manage the ``np.load`` context.
    Returns ``{}`` if the file is missing, unreadable, or fails to parse —
    same "no prior state" semantics as :func:`try_load_json`.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with np.load(p, allow_pickle=False) as f:
            return {k: np.asarray(f[k]) for k in f.files}
    except Exception:
        return {}


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
        for key in row:
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


def extract_runs(exp_def: list[dict] | dict) -> list[dict]:
    """Return the runs list from an experiment def (wrapper dict or bare list)."""
    if isinstance(exp_def, dict):
        return exp_def.get("runs", [])
    return exp_def if exp_def is not None else []


def iter_runs(runs: list[dict] | dict, cli_overrides: dict):
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


def active_differentiable_solvers(
    cfg: ProblemConfig, suite: str = "gradient", experiment: str | None = None
) -> list[str]:
    """Differentiable solvers not excluded for *suite* (and optionally *experiment*)."""
    from mosaic.benchmarks.core.config import has_vjp

    return [
        name
        for name in active_solvers(cfg, suite, experiment)
        if has_vjp(cfg.solvers[name])
    ]


def experiment_dir(
    results_dir: Path, problem: str, suite: str, experiment: str, suffix: str = ""
) -> Path:
    """Create and return results/<problem>/<suite>/<experiment><suffix>/."""
    d = results_dir / problem / suite / f"{experiment}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    return d
