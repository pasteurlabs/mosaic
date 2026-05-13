"""Array math, run iteration, and solver-filtering utilities.

Filesystem I/O (JSON/CSV/NPZ helpers, ``save_experiment``,
``save_field_snapshots_npz``, content hashing, ``results_dir``,
``experiment_dir``) lives in :mod:`mosaic.benchmarks.core.io`. This module
contains the non-I/O utilities that don't fit anywhere more specific:

  * Array math: :func:`trimmed_mean`, :func:`l2_error_rel`, :func:`is_valid`.
  * Run iteration: :func:`iter_runs`, :func:`_debug_run`.
  * Solver filtering: :func:`exclusion_lookup`, :func:`active_solvers`,
    :func:`active_differentiable_solvers`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Problem

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


# ── Run iteration ─────────────────────────────────────────────────────────────


def iter_runs(runs: list[dict] | None, cli_overrides: dict):
    """Yield run configs from a runs list, applying debug and IC filters.

    Each run dict has named sub-groups: ic, physics, sweep, fd, optim, jacobian,
    fine, cost.  Yields a deep-copied, debug-adjusted run per matching entry.

    cli_overrides keys consumed here (read-only, never mutated):
        ic_names   — list[str]: filter to runs whose ic.name is in the list
        debug      — bool: cap physics sizes and truncate sweep lists
    """
    import copy

    if runs is None:
        return

    ic_filter = cli_overrides.get("ic_names")
    debug = cli_overrides.get("debug", False)

    for run in runs:
        ic_name = run.get("ic", {}).get("name", "")
        if ic_filter and ic_name not in ic_filter:
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


def exclusion_lookup(
    exclusions: dict,
    suite: str,
    experiment: str | None = None,
    sub: str | None = None,
) -> tuple[str, object] | None:
    """Look up the most-specific exclusion entry for (suite, experiment[, sub]).

    Returns ``(matched_key, value)`` or ``None``. A key matches if it equals
    the full ``"<suite>/<experiment>[/<sub>]"`` path OR is a prefix-component
    of it (``"<suite>/<experiment>"``, ``"<suite>"``). The longest matching
    key wins, so a config can target one experiment (``forward/cylinder``) or
    a whole suite (``gradient``) without needing a fallback chain.

    Used by both :func:`active_solvers` (runtime gating) and
    :mod:`core.status` (display) so the two paths can't drift on precedence.
    """
    if not exclusions:
        return None
    # Full path: "<suite>", "<suite>/<exp>", "<suite>/<exp>/<sub>".
    parts: list[str] = [suite]
    if experiment:
        for p in experiment.split("/"):
            parts.append(p)
    if sub:
        for p in sub.split("/"):
            parts.append(p)
    # Try longest prefix first.
    for n in range(len(parts), 0, -1):
        key = "/".join(parts[:n])
        if key in exclusions:
            return key, exclusions[key]
    return None


def active_solvers(
    cfg: Problem, suite: str, experiment: str | None = None
) -> list[str]:
    """Solver names not excluded for *suite* (and optionally *experiment*).

    Reads ``cfg.exclusions[name]`` via :func:`exclusion_lookup`, which tries
    the most-specific key first (``"suite/experiment"``) and falls back to
    the bare experiment key and finally to ``suite``. ``mosaic status`` uses
    the same lookup so the runtime filter and the display can never disagree
    on which cells are gated.

    ``Exclusion(category="anomaly_explained", ...)`` is treated as
    *not excluded* — the solver runs and produces output that's flagged in
    the status display but not skipped at runtime.

    Prints a one-line warning per excluded solver, naming the matched key so
    the exclusion source is visible in runner output.
    """
    from .console import console

    result = []
    for spec in cfg.solvers:
        name = spec.name
        match = exclusion_lookup(cfg.exclusions.get(name, {}), suite, experiment)
        if match is None:
            result.append(name)
            continue
        matched_key, value = match
        if getattr(value, "category", None) == "anomaly_explained":
            # Explained-anomaly is a display annotation, not a runtime skip.
            result.append(name)
            continue
        reason = getattr(value, "reason", value)
        console.print(
            f"  [yellow]SKIP {name}[/] excluded from {matched_key!r}: {reason}"
        )
    return result


def active_differentiable_solvers(
    cfg: Problem, suite: str = "gradient", experiment: str | None = None
) -> list[str]:
    """Differentiable solvers not excluded for *suite* (and optionally *experiment*)."""
    from mosaic.benchmarks.core.config import has_vjp

    return [
        name
        for name in active_solvers(cfg, suite, experiment)
        if has_vjp(cfg.solver(name))
    ]
