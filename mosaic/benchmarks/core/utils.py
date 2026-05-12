"""Array math, run iteration, and solver-filtering utilities.

Filesystem I/O (JSON/CSV/NPZ helpers, ``save_experiment``,
``save_field_snapshots_npz``, content hashing, ``results_dir``,
``experiment_dir``) lives in :mod:`mosaic.benchmarks.core.io`. This module
contains the non-I/O utilities that don't fit anywhere more specific:

  * Array math: :func:`trimmed_mean`, :func:`l2_error_rel`, :func:`is_valid`.
  * Run iteration: :func:`physics_params`, :func:`extract_runs`,
    :func:`iter_runs`, :func:`_debug_run`.
  * Solver filtering: :func:`exclusion_candidate_keys`,
    :func:`exclusion_lookup`, :func:`active_solvers`,
    :func:`active_differentiable_solvers`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import ProblemConfig

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
