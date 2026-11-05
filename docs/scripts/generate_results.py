#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the results pages of the docs from benchmark results.

Writes into ``docs/_build/_generated/``: the body of each setup's page
(``results/<setup>.qmd``, included by ``docs/results/<setup>.qmd``), the
includes of the Results overview (``scoreboard.qmd``, ``explorer.qmd``,
``gates.qmd``), the per-solver checks for the solver reference
(``scores.json``) and a copy of the plots (``plots/``). Without results under
``mosaic-results/`` they hold placeholders, so the site still renders.

Usage:
    python docs/scripts/generate_results.py
"""

from __future__ import annotations

import functools
import json
import math
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[1]
ROOT = DOCS_DIR.parent
RESULTS_DIR = ROOT / "mosaic-results"
OUTPUT_DIR = DOCS_DIR / "_build" / "_generated"


def _load_docs_module(stem: str):
    """Load docs/scripts/<stem>.py whether this runs as a script or a module."""
    import importlib.util

    name = f"_mosaic_{stem}"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


views = _load_docs_module("intervention_views")
scoreboard = _load_docs_module("scoreboard")

# Every generated page and include ends up in docs/results/, so links and
# images are relative to that directory. The plots are copied from
# mosaic-results/, which lies outside the Quarto project.
_PLOT_SUFFIXES = (".png", ".gif", ".vl.json")
_IMG_BASE = "../_generated/plots"

# Suite traversal order
_SUITE_ORDER = ["ics", "forward", "cost", "gradient", "optimization"]

# Experiment traversal order within each suite
_EXPERIMENT_ORDER = [
    "agreement",
    "convergence",
    "diagnostics",
    "stability",
    "fd_check",
    "analytic_check",
    "analytic_check_nu",
    "param_sweep",
    "resolution_sweep",
    "horizon_sweep",
    "viscosity_recovery",
    "ic_recovery",
    "param_recovery",
    "topopt",
]

PROBLEM_LABELS = {
    "ns-grid": "Navier\u2013Stokes (2D)",
    "ns-3d-grid": "Navier\u2013Stokes (3D)",
    "structural-mesh": "Structural mechanics",
    "thermal-mesh": "Heat transfer",
}

# Schematic of each setup, drawn by docs/scripts/make_domain_figures.py. Path
# is relative to docs/.
SETUP_ILLUSTRATIONS = {
    key: f"figures/setup_{key.replace('-', '_')}.svg"
    for key in (
        "ns-2d-periodic",
        "ns-2d-cylinder",
        "ns-2d-kolmogorov",
        "ns-3d",
        "structural",
        "heat",
    )
}

# One line on what each setup measures that the others don't, shown at the top
# of its page, aimed at a reader who knows ML but
# not the specific solver community.
SETUP_TAGLINES = {
    "ns-2d-periodic": (
        "Forward accuracy against the analytic Taylor\u2013Green solution, and "
        "how gradients hold up as the rollout lengthens and the flow becomes "
        "chaotic, in a 2D box without walls."
    ),
    "ns-2d-cylinder": (
        "Optimize the inflow of a 2D channel to minimize drag on a cylinder, "
        "which requires solvers to handle walls, an obstacle, an inflow and an "
        "outflow."
    ),
    "ns-2d-kolmogorov": (
        "Forced 2D turbulence that settles into a statistically steady state, "
        "and gradients through a chain of solver calls, as in training a model "
        "through a solver."
    ),
    "ns-3d": (
        "Recover the initial velocity field of a 3D flow from a later snapshot, "
        "differentiating through the Navier\u2013Stokes rollout."
    ),
    "structural": (
        "Place a fixed budget of material in a clamped beam to make it as "
        "stiff as possible, differentiating through a finite-element solve."
    ),
    "heat": (
        "Invert for the conductivity field of a slab from its temperature, "
        "differentiating through a steady heat-conduction solve."
    ),
}

# Setup-specific background, shown after the problem's description on pages of
# problems with more than one setup.
SETUP_DESCRIPTIONS = {
    "ns-2d-periodic": (
        "The nonlinear advection term "
        "$\\nabla\\cdot(\\mathbf{u}\\otimes\\mathbf{u})$ transfers energy across "
        "scales. At low $\\nu$ the flow develops turbulent cascades and a positive "
        "Lyapunov exponent. Long-horizon gradients then become exponentially "
        "sensitive to perturbations, so backpropagation through the simulation "
        "is hardest in this regime."
    ),
    "ns-2d-cylinder": (
        "The channel follows the Sch\u00e4fer\u2013Turek DFG 2D-1 benchmark at "
        "$Re = 20$, where the flow is steady and laminar."
    ),
    "ns-2d-kolmogorov": (
        "The Kolmogorov flow adds a steady body force "
        "$\\mathbf{f} = (\\sin 4y, 0)$ and a linear drag $-\\alpha\\mathbf{u}$ "
        "to the right-hand side. The force drives the flow, and the drag removes "
        "the energy that piles up at the largest scales, so the turbulent flow "
        "settles into a statistically steady state."
    ),
}

# Boundary conditions of setups that differ from their problem's
# ``bc_description``.
SETUP_BCS = {
    "ns-2d-cylinder": (
        "The Sch\u00e4fer\u2013Turek DFG 2D-1 channel "
        "$[0, 2.2] \\times [0, 0.41]$ with a cylinder of diameter 0.1 at "
        "$(0.2, 0.2)$, a parabolic inflow of mean speed 0.2 on the left, a "
        "zero-gradient outflow at $p=0$ on the right, and no-slip walls. The "
        "viscosity $\\nu = 10^{-3}$ gives $Re = 20$."
    ),
}

# Shared "how the results were produced" callout, surfaced near the top of
# every results page. The benchmark runs on GitHub-hosted runners: GPU solvers
# on a Tesla T4 node, CPU-only solvers (``uses_gpu: false``) on a separate CPU
# node, one VM per job.
RESULTS_PROVENANCE = (
    "These are **example results**, produced automatically on GitHub-hosted "
    "runners and refreshed on every release. Each solver runs on its intended "
    "device: GPU-capable solvers on a Tesla T4 GPU node, CPU-only solvers "
    "(OpenFOAM, deal.II, FEniCS, Firedrake, TopOpt.jl) on a separate CPU node. "
    "Each benchmark job has its virtual machine to itself, but the hardware "
    "and the load on the host can change from run to run, so wall-clock times "
    "vary. Read them for relative scaling between solvers rather than as "
    "absolute timings. Accuracy and gradient metrics change little across "
    "hardware, although float32 arithmetic is not identical across devices "
    "and chaotic rollouts amplify small discrepancies. For numbers that "
    "reflect *your* setup, [run the benchmark yourself](../getting-started.qmd) "
    "on your target hardware."
)

SUITE_LABELS = {
    "ics": "Initial conditions",
    "forward": "Forward",
    "gradient": "Gradient",
    "optimization": "Optimization",
    "cost": "Cost",
}

SUITE_DESCRIPTIONS = {
    "ics": (
        "The initial conditions and fields that this setup's experiments start "
        "from. They are plotted without running any solver."
    ),
    "forward": (
        "**Is the prediction right?** Forward experiments compare each solver's "
        "output with a trusted reference, an analytic solution where one exists, "
        "and the other solvers."
    ),
    "gradient": (
        "**Is the gradient right?** Gradient experiments compare each solver's "
        "AD or adjoint gradient with finite differences, in magnitude (relative "
        "$L^2$ error) and direction (cosine similarity)."
    ),
    "optimization": (
        "**Can you optimize through it?** Optimization experiments run a "
        "gradient-based optimizer on each solver's own gradients. A gradient "
        "can pass the finite-difference check and still fail to drive an "
        "optimization loop."
    ),
    "cost": (
        "**What does it cost?** Wall-clock time and peak memory of the forward "
        "and VJP passes as the problem size $N$ and the number of steps grow. "
        "See the note on how these results were produced before reading "
        "absolute numbers."
    ),
}

# What each experiment measures, as its heading. Sub-experiments read
# "Parent (sub)" unless listed under their full key.
EXPERIMENT_LABELS = {
    # forward
    "baseline": "Agreement with the reference",
    "agreement": "Agreement between solvers",
    "agreement/tgv": "Agreement between solvers (Taylor\u2013Green)",
    "agreement/multimode": "Agreement between solvers (multimode)",
    "tgv_nu_sweep": "Accuracy across viscosity (Taylor\u2013Green)",
    "physical_laws": "Conservation and decay laws",
    "physical_laws/vs_N": "Conservation and decay laws (across resolution)",
    "physical_laws/vs_steps": "Conservation and decay laws (across steps)",
    "physical_laws/vs_nu": "Conservation and decay laws (across viscosity)",
    "cylinder": "Flow past a cylinder against the DFG reference",
    "kolmogorov_laminar": "Accuracy on the laminar Kolmogorov transient",
    "kolmogorov": "Accuracy and speed on turbulent Kolmogorov flow",
    "source_baseline": "Agreement with the reference (heat source)",
    "source_linearity": "Linearity in the heat source",
    # cost
    "spatial_cost": "Forward cost across resolution",
    "temporal_cost": "Forward cost across steps",
    "vjp_cost": "VJP cost",
    "vjp_cost/by_N": "VJP cost across resolution",
    "vjp_cost/by_steps": "VJP cost across steps",
    # gradient
    "fd_check": "Gradient against finite differences",
    "chain_fd_check": "Gradient through chained solver calls",
    "analytic_check": "Gradient against the analytic Taylor\u2013Green gradient",
    "analytic_check_nu": "Viscosity gradient against the analytic gradient",
    "param_sweep": "Gradient accuracy across the control parameter",
    "horizon_sweep": "Gradient accuracy across rollout length",
    "horizon_sweep_limits": "How far gradients reach in rollout length",
    "jacobian_svd": "Jacobian singular values",
    "jacobian_svd_steps20": "Jacobian singular values (20 steps)",
    "jacobian_svd_steps40": "Jacobian singular values (40 steps)",
    "jacobian_svd_nu01": "Jacobian singular values (\u03bd = 0.1)",
    "source_fd_check": "Gradient with respect to the heat source",
    "source_width_sweep": "Heat-source gradient across source width",
    # optimization
    "drag_opt": "Inflow optimization for drag",
    "recovery": "Initial-condition recovery",
    "recovery_constant_ic_bfgs_proj": "Initial-condition recovery (L-BFGS)",
    "topopt": "Topology optimization",
    "conductivity_recovery_bfgs": "Conductivity recovery (L-BFGS)",
    # initial conditions
    "tgv": "Taylor\u2013Green vortex",
    "multimode": "Multimode vortices",
    "rest": "Fluid at rest",
    "dfg_inflow": "DFG 2D-1 parabolic inflow",
    "flat_inflow": "Flat inflow",
    "tgv3d": "3D Taylor\u2013Green vortex",
    "abc": "Arnold\u2013Beltrami\u2013Childress flow",
    "rand_div_free": "Random divergence-free field",
    "uniform": "Uniform field",
    "random": "Random field",
    "two_density_bumps": "Two density bumps",
    "gaussian_source": "Gaussian heat source",
    "zero_source": "Zero heat source",
    "two_gaussians": "Two Gaussian heat sources",
}

# PNG display order within each experiment directory
_PNG_ORDER = [
    # ics suite
    "ic",
    # forward
    "curves",
    "fields",
    "error",
    "scalars",
    "rdf",
    "spectra",
    "energy",
    "eigval_min",
    "n_negative_modes",
    # gradient / recovery
    "fd_check",
    "gradient_fields",
    "analytic_check",
    "param_sweep",
    "resolution_sweep",
    "horizon_sweep",
    "ucurves",
    "error_vs_param",
    "best_eps_vs_param",
    "error_vs_N",
    "best_eps_vs_N",
    "error_vs_steps",
    "best_eps_vs_steps",
    "mu_paths",
    "loss_curves",
    "final_error",
    "recovery",
    "convergence_curves",
    "recovery_fields",
    "param_recovery",
    "topopt_convergence",
    "topopt_fields",
    "topopt_3d",
    "topopt_density",
    # cost
    "cost",
    # pairwise
    "pairwise",
    "divergence_rms",
]

# ── Problem config import ─────────────────────────────────────────────────────

_SOLVER_GYM = ROOT / "mosaic"
if str(_SOLVER_GYM) not in sys.path:
    sys.path.insert(0, str(_SOLVER_GYM))


def _get_config(problem: str):
    """Lazily import and return the problem config.

    ``mosaic.benchmarks.problems`` transitively imports the full compute stack
    (jax, jaxlib, scipy, ...), which is hundreds of MB resident.
    Importing it at module load OOM-kills resource-constrained docs builders
    (e.g. Read the Docs) even when there are no results to describe. Defer the
    import to the few call sites that actually need problem metadata.
    """
    from mosaic.benchmarks.problems import get_config

    return get_config(problem)


def _plot_description(problem: str, suite: str, experiment: str) -> str:
    """Return the plot description for (suite, experiment) from Problem."""
    try:
        cfg = _get_config(problem)
        if suite == "ics":
            return cfg.get_ic_description(experiment)
        return cfg.get_plot_description(suite, experiment)
    except Exception:
        return ""


def _experiment_description(problem: str, suite: str, experiment: str) -> str:
    """Return the short experiment description (what it measures) from Problem."""
    try:
        return _get_config(problem).get_experiment_description(suite, experiment)
    except Exception:
        return ""


def _problem_description(problem: str) -> str:
    """Return problem-level description from Problem."""
    try:
        return _get_config(problem).description
    except Exception:
        return ""


def _bc_description(problem: str) -> str:
    """Return boundary-condition description from Problem."""
    try:
        return _get_config(problem).bc_description
    except Exception:
        return ""


# ── Per-suite results by solver ──────────────────────────────────────────────
#
# Each suite is ranked on its headline metric, computed directly from the
# result.json files so the table can't drift from the plots, and only over the
# experiments of the page's setup. Rules are deliberately conservative: a
# solver appears only when its metric is finite, and a suite's table is
# emitted only when at least two solvers can be ranked (a one-row table says
# nothing).


def _median(xs: list[float]) -> float:
    vals = sorted(v for v in xs if isinstance(v, int | float) and math.isfinite(v))
    if not vals:
        return float("nan")
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def _mean_finite(xs: list[float]) -> float:
    vals = [v for v in xs if isinstance(v, int | float) and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _fmt_sci(v: float) -> str:
    if not math.isfinite(v):
        return "—"
    if v == 0:
        return "0"
    return f"{v:.2e}"


def _load_suite_results(problem: str, suite: str, experiment: str) -> list[dict]:
    """Return the flat ``results`` list for one (suite, experiment), or []."""
    path = RESULTS_DIR / problem / suite / experiment / "result.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("results", []) or []
    except Exception:
        return []


def _owned(problem: str, setup: str, suite: str, experiments: list[str]) -> list[str]:
    """The experiments of ``experiments`` (under ``suite``) that ``setup`` owns."""
    return [
        exp
        for exp in experiments
        if (owner := views.iv.setup_of(problem, f"{suite}/{exp}")) is not None
        and owner.key == setup
    ]


def _experiment_dirs(problem: str, suite: str) -> list[str]:
    suite_dir = RESULTS_DIR / problem / suite
    if not suite_dir.is_dir():
        return []
    return sorted(p.name for p in suite_dir.iterdir() if p.is_dir())


def _rank_forward(
    problem: str, setup: str
) -> tuple[list[str], list[tuple], str] | None:
    """Rank solvers by mean relative error vs the reference (lower is better)."""
    # Prefer the viscosity/parameter sweep, then the other experiments that
    # carry an ``error`` metric.
    preferred = ["tgv_nu_sweep", "agreement", "baseline", "source_baseline"]
    others = [e for e in _experiment_dirs(problem, "forward") if e not in preferred]
    by_solver: dict[str, list[float]] = {}
    for exp in _owned(problem, setup, "forward", preferred + others):
        for r in _load_suite_results(problem, "forward", exp):
            m = r.get("metrics") or {}
            if isinstance(m.get("error"), int | float) and m.get("valid", True):
                by_solver.setdefault(r["solver"], []).append(m["error"])
        if by_solver:
            break
    else:
        return None
    rows = sorted(
        ((s, _mean_finite(v)) for s, v in by_solver.items()),
        key=lambda t: (math.isnan(t[1]), t[1]),
    )
    rows = [(s, e) for s, e in rows if math.isfinite(e)]
    if len(rows) < 2:
        return None
    header = ["Solver", "Mean rel. error"]
    return header, [(s, _fmt_sci(e)) for s, e in rows], f"forward/{exp}"


def _rank_gradient(
    problem: str, setup: str
) -> tuple[list[str], list[tuple], str] | None:
    """Rank by best-ε FD error (lower better); show direction cosine alongside."""
    by_solver: dict[str, tuple[float, float]] = {}
    for exp in _owned(problem, setup, "gradient", ["fd_check", "chain_fd_check"]):
        for r in _load_suite_results(problem, "gradient", exp):
            eps_sweep = (r.get("metrics") or {}).get("eps_sweep") or {}
            best_err, best_cos = float("inf"), float("nan")
            for entry in eps_sweep.values():
                re = entry.get("rel_error")
                med = _median(re) if isinstance(re, list) else _median([re])
                if math.isfinite(med) and med < best_err:
                    best_err = med
                    cos = entry.get("cosine")
                    best_cos = cos if isinstance(cos, int | float) else float("nan")
            if math.isfinite(best_err):
                by_solver[r["solver"]] = (best_err, best_cos)
        if by_solver:
            break
    else:
        return None
    rows = sorted(by_solver.items(), key=lambda t: t[1][0])
    if len(rows) < 2:
        return None
    # Cosines sit so close to 1 that a fixed-decimal column reads as a wall of
    # "1.00000"; report the defect 1 − cos instead, where smaller is better.
    header = ["Solver", "Best-ε FD error", "1 − cosine"]
    return (
        header,
        [
            (s, _fmt_sci(err), _fmt_sci(1.0 - cos) if math.isfinite(cos) else "—")
            for s, (err, cos) in rows
        ],
        f"gradient/{exp}",
    )


def _fmt_mib(v: object) -> str:
    if not isinstance(v, int | float) or not math.isfinite(v):
        return "—"
    return f"{v / 1024:.3g} GiB" if v >= 1024 else f"{v:.3g} MiB"


def _rank_cost(problem: str, setup: str) -> tuple[list[str], list[tuple], str] | None:
    """Rank by forward wall-clock, reporting forward and VJP time and memory.

    Forward time comes from the ``spatial_cost`` sweep and VJP time from
    ``vjp_cost/by_N``, both at the problem size the check tables compare the
    solver's device class at (``scoreboard.comparison_n``). A solver without
    a time at that size shows where it stopped, and forward-only solvers have
    no VJP entry, which shows as "—". Sorted by forward time, GPU solvers
    first, then the solvers left out. The forward pass's first call, which includes compilation and
    setup, is shown next to it and does not affect the order. Peak memory is
    device memory for GPU solvers and container memory for CPU solvers.
    """
    if not _owned(problem, setup, "cost", ["spatial_cost"]):
        return None
    fwd, vjp = (
        scoreboard._times_by_n(scoreboard.cost_result(RESULTS_DIR, problem, exp))
        for exp in ("spatial_cost", "vjp_cost/by_N")
    )
    if len(fwd) < 2:
        return None
    gpu = {n: m.get("gpu", True) for n, m in _solver_meta().get(problem, {}).items()}

    compared_at: dict[str, float | None] = {}
    for on_gpu in (True, False):
        group = [s for s in fwd if gpu.get(s, True) is on_gpu]
        n = scoreboard.comparison_n([fwd, vjp], group)
        compared_at |= dict.fromkeys(group, n)

    def _point(times: dict, s: str) -> tuple[float, dict | None] | None:
        """``(N, metrics)`` at the compared size, ``(reach, None)`` for a
        solver that stopped short of it, or None without results."""
        t = times.get(s)
        if not t:
            return None
        n = compared_at[s]
        return (n, t[n]) if n in t else (max(t), None)

    points = {s: (_point(fwd, s), _point(vjp, s)) for s in fwd}

    def _common_n(column: int) -> float | None:
        """The N of every time in a column, if it is the same for all."""
        ns = {p[column][0] for p in points.values() if p[column] and p[column][1]}
        return ns.pop() if len(ns) == 1 else None

    fwd_n, vjp_n = _common_n(0), _common_n(1)

    def _time(v: tuple[float, dict | None] | None, common: float | None) -> str:
        # Without one N for the whole column, each time carries its own.
        if not v:
            return "—"
        n, m = v
        if m is None:
            return f"— (stops at N={int(n)})"
        return f"{m['mean']:.3g} s" + (f" @ N={int(n)}" if common is None else "")

    def _at(label: str, common: float | None) -> str:
        return f"{label} (N={int(common)})" if common is not None else label

    def _first(v: tuple[float, dict | None]) -> str:
        first = v[1]["first"] if v[1] else math.nan
        return f"{first:.3g} s" if math.isfinite(first) else "—"

    def _ratio(f: tuple[float, dict | None], b: tuple[float, dict | None] | None):
        if not (b and b[1] and f[1]) or f[1]["mean"] <= 0:
            return "—"
        return f"{b[1]['mean'] / f[1]['mean']:.3g}×"

    def _memory(v: tuple[float, dict | None] | None, on_gpu: bool) -> str:
        return _fmt_mib(v[1]["vram" if on_gpu else "ram"]) if v and v[1] else "—"

    def _order(item) -> tuple:
        s, (fv, _) = item
        return (not gpu.get(s, True), fv[1] is None, fv[1]["mean"] if fv[1] else 0)

    rows = sorted(points.items(), key=_order)
    header = [
        "Solver",
        "Device",
        _at("Forward time", fwd_n),
        "Forward first call",
        _at("VJP time", vjp_n),
        "VJP / forward",
        "Forward peak memory",
        "VJP peak memory",
    ]
    return (
        header,
        [
            (
                s,
                "GPU" if gpu.get(s, True) else "CPU",
                _time(fv, fwd_n),
                _first(fv),
                _time(vv, vjp_n),
                _ratio(fv, vv),
                _memory(fv, gpu.get(s, True)),
                _memory(vv, gpu.get(s, True)),
            )
            for s, (fv, vv) in rows
        ],
        "cost/spatial_cost",
    )


# final-objective metric name by domain optimization experiment, in priority
# order (the first key present in a result's metrics is used to rank it).
_OPT_FINAL_KEYS = ("final_error", "final_drag", "final_compliance", "final_ic_error")

# Human-readable column labels for the ranked metric. Falls back to a generic
# de-underscored title when a key is missing here.
_OPT_METRIC_LABELS = {
    "final_error": "Final error",
    "final_drag": "Final drag",
    "final_compliance": "Final compliance",
    "final_ic_error": "Final IC recovery error",
}


def _rank_optimization(
    problem: str, setup: str
) -> tuple[list[str], list[tuple], str] | None:
    """Rank by final optimization objective (lower is better), with the
    iterations and estimated time until the loss levelled off
    (``scoreboard.plateau``)."""
    # The first of the setup's optimization experiments with a recognised
    # final metric.
    rows: list[tuple[str, float]] = []
    metric_label = None
    for exp in _owned(
        problem, setup, "optimization", _experiment_dirs(problem, "optimization")
    ):
        path = RESULTS_DIR / problem / "optimization" / exp / "result.json"
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: S112 — no or malformed result file
            continue
        plateaus = scoreboard.plateau(result)
        for r in result.get("results") or []:
            m = r.get("metrics") or {}
            key = next((k for k in _OPT_FINAL_KEYS if k in m), None)
            if key is None:
                continue
            val = m.get(key)
            if isinstance(val, int | float) and math.isfinite(val):
                metric_label = _OPT_METRIC_LABELS.get(
                    key, key.replace("final_", "final ").replace("_", " ").capitalize()
                )
                rows.append((r["solver"], float(val)))
        if rows:
            break
    else:
        return None
    if len(rows) < 2:
        return None
    rows.sort(key=lambda t: t[1])

    def _iters(s: str) -> str:
        p = plateaus.get(s)
        if not p:
            return "—"
        return f"{p['iters']:.0f}" if p["iters"].is_integer() else f"{p['iters']:.1f}"

    def _time(s: str) -> str:
        p = plateaus.get(s)
        return scoreboard.duration(p["seconds"]) if p and p["seconds"] else "—"

    timed = any(p["seconds"] is not None for p in plateaus.values())
    header = ["Solver", metric_label, "Iterations to plateau"]
    header += ["Time to plateau"] if timed else []
    return (
        header,
        [(s, _fmt_sci(v), _iters(s), *((_time(s),) if timed else ())) for s, v in rows],
        f"optimization/{exp}",
    )


_SUITE_RANKERS = {
    "forward": _rank_forward,
    "gradient": _rank_gradient,
    "cost": _rank_cost,
    "optimization": _rank_optimization,
}


def _leaderboard_block(
    problem: str, suite: str, setup: str, shown: frozenset[str] = frozenset()
) -> list[str]:
    """Return QMD lines for the per-suite results-by-solver table, or [] if none.

    Rendered as a plain, always-visible table under a "Results by solver"
    lead-in, suited to a scientific report. ``shown`` holds the experiments of
    ``suite`` that have a heading on the page, which the caption links to.
    """
    ranker = _SUITE_RANKERS.get(suite)
    if ranker is None:
        return []
    try:
        ranked = ranker(problem, setup)
    except Exception:
        return []
    if not ranked:
        return []
    header, rows, experiment = ranked
    # Wrap the table in a .sortable-table div so docs/assets/sortable-tables.html
    # (injected via include-after-body) can attach click-to-sort handlers
    # (numeric-aware). Static and correctly sorted without JS, so it
    # degrades gracefully.
    # Rows are already sorted, so a Rank column adds nothing; rely on row
    # order (and click-to-sort) instead.
    lines = [
        "**Results by solver**",
        "",
        "::: {.column-page}",
        "::: {.sortable-table .leaderboard}",
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines += [":::", ":::"]
    exp_suite, exp_key = experiment.split("/", 1)
    anchor = (
        _exp_anchor(exp_suite, exp_key) if exp_key in shown else f"suite-{exp_suite}"
    )
    lines += ["", f"*From [{_exp_label(exp_key)}](#{anchor}).*", ""]
    return lines


# ── Setup and effort (intervention records) ──────────────────────────────────


def _solvers_by_setup(problem: str) -> dict[str, list[str]]:
    """Solvers with results in each setup of ``problem``, in first-seen order."""
    by_setup: dict[str, list[str]] = {}
    problem_dir = RESULTS_DIR / problem
    if not problem_dir.is_dir():
        return by_setup
    for path in sorted(problem_dir.rglob("result.json")):
        experiment = str(path.parent.relative_to(problem_dir))
        if "_debug" in experiment or experiment.startswith("ics"):
            continue
        setup = views.iv.setup_of(problem, experiment)
        if setup is None:
            continue
        try:
            results = json.loads(path.read_text(encoding="utf-8")).get("results")
        except Exception:  # noqa: S112 — malformed result file, skip it
            continue
        names = by_setup.setdefault(setup.key, [])
        for r in results or []:
            solver = r.get("solver") if isinstance(r, dict) else None
            if solver and solver not in names:
                names.append(solver)
    return by_setup


def _effort_block(problem: str, setup: str) -> list[str]:
    """QMD lines for the effort and deviations table of one setup, or []."""
    solvers = _solvers_by_setup(problem).get(setup)
    if not solvers:
        return []
    records = views.records_for(setup)
    lines = [
        "## Effort and deviations",
        "",
        "What it took to run each solver on this setup. " + views.EFFORT_EXPLAINER,
        "",
        "::: {.sortable-table}",
        f"| Solver | {views.EFFORT_LABEL} | {views.DEVIATIONS_LABEL} | Interventions |",
        "|---|---|---|---|",
    ]
    for solver in sorted(solvers, key=str.lower):
        record = records.get(solver)
        lines.append(
            f"| {solver} | {views.effort_cell(record, setup)} "
            f"| {views.deviations_cell(record, setup)} "
            f"| {views.interventions_cell(record, setup)} |"
        )
    lines += [":::", ""]
    return lines


# ── Helpers ───────────────────────────────────────────────────────────────────


def setup_page(setup: str) -> str:
    """The results page of ``setup``, relative to docs/results/."""
    return f"{setup}.qmd"


def _exp_anchor(suite: str, experiment: str) -> str:
    """Stable heading id for an experiment, shared by the setup page and the
    Results overview explorer (e.g. ``exp-forward-agreement-tgv``)."""
    key = experiment.replace("/", "-").replace("_", "-") or "overview"
    return f"exp-{suite}-{key}"


def _exp_label(experiment: str) -> str:
    """Display label for an experiment key, "Parent (sub)" for sub-experiments."""
    if experiment in EXPERIMENT_LABELS:
        return EXPERIMENT_LABELS[experiment]
    if "/" in experiment:
        parent_exp, sub_exp = experiment.split("/", 1)
        parent_label = EXPERIMENT_LABELS.get(
            parent_exp, parent_exp.replace("_", " ").capitalize()
        )
        return f"{parent_label} ({sub_exp.replace('_', ' ')})"
    return experiment.replace("_", " ").capitalize()


def _sort_pngs(paths: list[Path]) -> list[Path]:
    def key(p: Path) -> tuple[int, str]:
        stem = p.stem
        for i, prefix in enumerate(_PNG_ORDER):
            if stem == prefix or stem.startswith(prefix + "_"):
                return (i, stem)
        return (len(_PNG_ORDER), stem)

    return sorted(paths, key=key)


def _img_src(problem: str, suite: str, experiment: str, png: Path) -> str:
    if experiment:
        return f"{_IMG_BASE}/{problem}/{suite}/{experiment}/{png.name}"
    return f"{_IMG_BASE}/{problem}/{suite}/{png.name}"


def _img_tag(problem: str, suite: str, experiment: str, png: Path) -> str:
    return f"![]({_img_src(problem, suite, experiment, png)}){{.lightbox}}"


def _chart_block(problem: str, suite: str, experiment: str, png: Path) -> str | None:
    """Interactive version of a plot, or None if none was exported.

    The plot harness writes the Vega-Lite spec ``<plot>.vl.json`` next to each
    ``<plot>.png`` from the same chart definition. The figure fetches it
    lazily (docs/assets/charts.html) and keeps the PNG as the fallback when there is
    no JS or the chart fails to load.
    """
    if not png.with_suffix(".vl.json").exists():
        return None
    src = _img_src(problem, suite, experiment, png)
    spec = src.removesuffix(".png") + ".vl.json"
    return (
        "```{=html}\n"
        f'<figure class="m-chart" data-spec="{spec}">'
        f'<img class="m-chart-fallback" src="{src}" alt="" loading="lazy">'
        "</figure>\n```"
    )


def _sweep_line(params: dict) -> str:
    """Return a one-line sweep description, or empty string if no sweep.

    Handles both the unified result format (top-level ``sweep`` dict in
    ``result.json``) and the legacy ``params.json`` formats.
    """
    # New nested format: params["sweep"] = {"key": ..., "values": [...]}
    if isinstance(params.get("sweep"), dict):
        sweep = params["sweep"]
        key = sweep.get("key", "?")
        vals = ", ".join(str(v) for v in sweep.get("values", []))
        return f"Sweeps `{key}` ∈ {{{vals}}}"
    # New nested format: params["cost"] = {"N_values": [...], "steps_values": [...]}
    if isinstance(params.get("cost"), dict):
        cost = params["cost"]
        parts: list[str] = []
        if "N_values" in cost:
            vals = ", ".join(str(v) for v in cost["N_values"])
            parts.append(f"N ∈ {{{vals}}}")
        if "steps_values" in cost:
            vals = ", ".join(str(v) for v in cost["steps_values"])
            parts.append(f"steps ∈ {{{vals}}}")
        return "Sweeps " + ", ".join(parts) if parts else ""
    # Legacy flat format (old result files)
    if "sweep_key" in params and "sweep_values" in params:
        vals = ", ".join(str(v) for v in params["sweep_values"])
        return f"Sweeps `{params['sweep_key']}` ∈ {{{vals}}}"
    if "horizons" in params:
        vals = ", ".join(str(v) for v in params["horizons"])
        return f"Sweeps horizon ∈ {{{vals}}}"
    parts = []
    if "N_values" in params:
        vals = ", ".join(str(v) for v in params["N_values"])
        parts.append(f"N ∈ {{{vals}}}")
    if "steps_values" in params:
        vals = ", ".join(str(v) for v in params["steps_values"])
        parts.append(f"steps ∈ {{{vals}}}")
    if "mu_true_values" in params:
        vals = ", ".join(str(v) for v in params["mu_true_values"])
        parts.append(f"μ ∈ {{{vals}}}")
    return "Sweeps " + ", ".join(parts) if parts else ""


def _load_params(params_path: Path) -> dict | None:
    """Load params from params.json or fall back to result.json's params field."""
    if params_path.exists():
        return json.loads(params_path.read_text(encoding="utf-8"))
    # Try result.json in the same directory
    result_path = params_path.parent / "result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            # v1 format: params is a top-level field; also expose sweep info
            params = result.get("params", {})
            if result.get("schema_version") == 1 and isinstance(
                result.get("sweep"), dict
            ):
                params = {**params, "sweep": result["sweep"]}
            return params
        except Exception:  # noqa: S110 — malformed result file, skip its params
            pass
    return None


def _params_block(
    params_path: Path | None, sub_params: list[tuple[str, Path]] | None = None
) -> list[str]:
    """Return QMD lines for a collapsible settings block."""
    lines: list[str] = [
        "",
        "::: {.callout-note collapse='true' title='Settings'}",
    ]

    if params_path is not None:
        params = _load_params(params_path)
        if params:
            sweep = _sweep_line(params)
            if sweep:
                lines += ["", sweep]
            lines += [
                "",
                "```json",
                json.dumps(params, indent=2),
                "```",
            ]
    elif sub_params:
        for name, p in sub_params:
            params = _load_params(p)
            if not params:
                continue
            sweep = _sweep_line(params)
            lines += ["", f"**`{name}`**"]
            if sweep:
                lines += ["", sweep]
            lines += [
                "",
                "```json",
                json.dumps(params, indent=2),
                "```",
            ]

    lines += ["", ":::"]
    return lines


def _scan_results() -> dict[str, dict[str, dict[str, list[Path]]]]:
    """Return {problem: {suite: {experiment: [png_paths]}}.

    Experiment key is the experiment directory name, or "" for PNGs that sit
    directly inside a suite directory (e.g. a combined cost overview plot).
    """
    tree: dict[str, dict[str, dict[str, list[Path]]]] = {}
    if not RESULTS_DIR.exists():
        return tree
    for problem_dir in sorted(RESULTS_DIR.iterdir()):
        if not problem_dir.is_dir() or problem_dir.name.startswith("."):
            continue
        problem = problem_dir.name
        for suite_dir in sorted(
            problem_dir.iterdir(),
            key=lambda p: (
                _SUITE_ORDER.index(p.name) if p.name in _SUITE_ORDER else 99,
                p.name,
            ),
        ):
            if not suite_dir.is_dir():
                continue
            suite = suite_dir.name

            # Suite-level PNGs (e.g. cost.png sitting directly in the suite dir)
            suite_pngs = _sort_pngs(
                [f for f in suite_dir.iterdir() if f.suffix == ".png"]
            )
            if suite_pngs:
                tree.setdefault(problem, {}).setdefault(suite, {})[""] = suite_pngs

            # Experiment-level PNGs
            for exp_dir in sorted(
                suite_dir.iterdir(),
                key=lambda p: (
                    _EXPERIMENT_ORDER.index(p.name)
                    if p.name in _EXPERIMENT_ORDER
                    else 99,
                    p.name,
                ),
            ):
                if not exp_dir.is_dir():
                    continue
                if "_debug" in exp_dir.name:
                    continue
                pngs = _sort_pngs([f for f in exp_dir.iterdir() if f.suffix == ".png"])
                if pngs:
                    tree.setdefault(problem, {}).setdefault(suite, {})[exp_dir.name] = (
                        pngs
                    )

                # Sub-experiment directories (one level deeper)
                for sub_dir in sorted(exp_dir.iterdir()):
                    if not sub_dir.is_dir():
                        continue
                    if "_debug" in sub_dir.name:
                        continue
                    sub_pngs = _sort_pngs(
                        [f for f in sub_dir.iterdir() if f.suffix == ".png"]
                    )
                    if not sub_pngs:
                        continue
                    sub_key = f"{exp_dir.name}/{sub_dir.name}"
                    tree.setdefault(problem, {}).setdefault(suite, {})[sub_key] = (
                        sub_pngs
                    )
    return tree


def _setup_suites(problem: str, suites: dict, setup: str) -> dict:
    """The part of a problem's result tree that belongs to ``setup``."""
    out: dict[str, dict[str, list[Path]]] = {}
    for suite, experiments in suites.items():
        for experiment, pngs in experiments.items():
            label = f"{suite}/{experiment}" if experiment else suite
            owner = views.iv.setup_of(problem, label)
            if owner is not None and owner.key == setup:
                out.setdefault(suite, {})[experiment] = pngs
    return out


def generate_qmd_for_setup(
    setup: dict, suites: dict, timestamp: str, scores: dict | None = None
) -> str:
    """Generate QMD content for the results page of one setup.

    ``setup`` is an entry of ``scoreboard.setups()``; ``suites`` the part of its
    problem's result tree that belongs to it (empty without results).
    """
    key, problem = setup["key"], setup["problem"]
    label = PROBLEM_LABELS.get(problem, problem.replace("-", " ").title())
    n_plots = sum(len(pngs) for exps in suites.values() for pngs in exps.values())

    pid = scoreboard.PROBLEM_IDS.get(problem)
    meta = (
        f"[{pid}]{{.domain-id .d-{pid}}} " if pid else ""
    ) + f"[{label}]{{.meta-chip}}"
    if suites:
        meta += (
            f" [Generated {timestamp}]{{.meta-chip}} "
            f"[{n_plots} plot{'s' if n_plots != 1 else ''}]{{.meta-chip}}"
        )
    lines: list[str] = [
        "---",
        f'title: "{setup["setup"]}"',
        "---",
        "",
        "::: {.page-meta}",
        meta,
        ":::",
        "",
    ]

    # Task schematic and tagline. The schematic is skipped if the figure is
    # missing, so a checkout without docs/figures/ still renders cleanly.
    illustration = SETUP_ILLUSTRATIONS.get(key)
    tagline = SETUP_TAGLINES.get(key)
    if illustration and (DOCS_DIR / illustration).exists():
        lines += [
            "::: {.domain-hero}",
            f"![](../{illustration}){{.domain-figure}}",
            "",
        ]
        if tagline:
            lines += [f"[{tagline}]{{.domain-tagline}}", ""]
        lines += [":::", ""]
    elif tagline:
        lines += [f"[{tagline}]{{.domain-tagline}}", ""]

    desc = _problem_description(problem)
    if desc:
        lines += [desc, ""]
    if key in SETUP_DESCRIPTIONS:
        lines += [SETUP_DESCRIPTIONS[key], ""]

    lines += [
        "::: {.callout-tip title='How these results were produced' collapse='true'}",
        "",
        RESULTS_PROVENANCE,
        "",
        ":::",
        "",
    ]

    bc = SETUP_BCS.get(key) or _bc_description(problem)
    if bc:
        lines += [
            "::: {.callout-note title='Boundary conditions' collapse='true'}",
            "",
            bc,
            "",
            ":::",
            "",
        ]

    if not suites:
        lines += [_NO_RESULTS]
        return "\n".join(lines)

    # The setup's row of the Results overview check tables, with each solver's
    # effort and deviations. Without a snapshot there are no checks, so the
    # effort table stands alone.
    checks = scoreboard.render_setup_page(scores, key) if scores else ""
    if checks:
        lines += [
            "## Checks {.unnumbered}",
            "",
            (
                "The [results overview](index.qmd#how-to-read-the-results) "
                "explains the columns and states."
            ),
            "",
            "::: {.column-page}",
            checks,
            ":::",
            "",
        ]
    else:
        lines += _effort_block(problem, key)

    for suite, experiments in suites.items():
        suite_label = SUITE_LABELS.get(suite, suite.title())
        suite_desc = SUITE_DESCRIPTIONS.get(suite, "")

        lines += [
            f'## {suite_label} {{#suite-{suite} .suite-heading data-suite="{suite}"}}',
            "",
        ]
        if suite_desc:
            lines += ["::: {.suite-intro}", suite_desc, ":::", ""]

        def _exp_sort_key(exp: str) -> tuple[int, str, str]:
            if exp == "":
                return (-1, "", "")
            # For sub-experiments like "horizon_sweep/tgv3d", sort after parent
            parts = exp.split("/", 1)
            parent = parts[0]
            child = parts[1] if len(parts) > 1 else ""
            idx = _EXPERIMENT_ORDER.index(parent) if parent in _EXPERIMENT_ORDER else 99
            return (idx, parent, child)

        for experiment in sorted(experiments.keys(), key=_exp_sort_key):
            pngs = experiments[experiment]
            is_sub = "/" in experiment

            lines += ["::: {.exp-card}", ""]
            if experiment == "":
                # Suite-level PNGs: no sub-heading
                exp_desc = _plot_description(problem, suite, "")
                if exp_desc:
                    lines += [exp_desc, ""]
                sub_params = sorted(
                    [
                        (d.name, d / "params.json")
                        for d in (RESULTS_DIR / problem / suite).iterdir()
                        if d.is_dir()
                    ],
                    key=lambda t: t[0],
                )
                lines += _params_block(None, sub_params=sub_params)
            else:
                # Build label: for sub-experiments use "Parent (sub)" format
                exp_label = _exp_label(experiment)
                exp_dir_name = experiment.split("/", 1)[0] if is_sub else experiment
                short_desc = _experiment_description(problem, suite, exp_dir_name)
                plot_desc = _plot_description(problem, suite, exp_dir_name)
                params_path = RESULTS_DIR / problem / suite / experiment / "params.json"

                lines += [f"### {exp_label} {{#{_exp_anchor(suite, experiment)}}}", ""]
                if short_desc:
                    lines += [short_desc, ""]
                if plot_desc:
                    lines += [plot_desc, ""]
                lines += _params_block(params_path)

            lines.append("")
            for png in pngs:
                chart = _chart_block(problem, suite, experiment, png)
                lines.append(chart or _img_tag(problem, suite, experiment, png))
            lines += ["", ":::", ""]

        # Results by solver, shown below the suite's plots.
        lines += _leaderboard_block(problem, suite, key, frozenset(experiments))

    return "\n".join(lines)


# ── Results overview includes (scoreboard + experiment explorer) ─────────────
#
# docs/results/index.qmd pulls these in with {{< include >}}. A checkout
# without benchmark results still renders (with a short placeholder) instead of
# failing on a missing include.

SCOREBOARD_INCLUDE = OUTPUT_DIR / "scoreboard.qmd"
EXPLORER_INCLUDE = OUTPUT_DIR / "explorer.qmd"
SCORES_JSON = OUTPUT_DIR / "scores.json"
GATES_INCLUDE = OUTPUT_DIR / "gates.qmd"

_NO_RESULTS = (
    "::: {.callout-note appearance='simple'}\n"
    "No benchmark results are available in this build. [Run the "
    "benchmark](../getting-started.qmd) or download the published results into "
    "`mosaic-results/` to populate this section.\n"
    ":::\n"
)


@functools.cache
def _solver_meta() -> dict[str, dict[str, dict]]:
    """``{problem: {solver_name: {display, anchor, gpu}}}`` from the problem configs.

    ``display`` is the card name from each container's tesseract_config.yaml
    (falling back to the SolverSpec name) and ``anchor`` is the solver's id on
    the Solver Reference page (``<tesseract dir>-<solver dir>``).
    """
    import yaml

    out: dict[str, dict[str, dict]] = {}
    for problem in scoreboard.PROBLEM_ORDER:
        try:
            cfg = _get_config(problem)
        except Exception:  # noqa: S112 — problem not importable in this env
            continue
        tdir = cfg.tesseract_dir
        for spec in cfg.solvers:
            display = spec.name
            cfg_path = Path(tdir) / spec.dir / "tesseract_config.yaml"
            try:
                meta = yaml.safe_load(cfg_path.read_text()) or {}
                display = ((meta.get("metadata") or {}).get("mosaic") or {}).get(
                    "name"
                ) or display
            except Exception:  # noqa: S110 — keep the SolverSpec name
                pass
            out.setdefault(problem, {})[spec.name] = {
                "display": display,
                "anchor": f"{Path(tdir).name}-{spec.dir}",
                "gpu": bool(getattr(spec, "uses_gpu", True)),
            }
    return out


def compute_scores() -> dict | None:
    try:
        return scoreboard.compute(RESULTS_DIR, _solver_meta())
    except Exception as exc:  # never fail the docs build on scoring
        print(f"Scoreboard skipped: {exc}")
        return None


def _md_attr(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")


def _plain(text: str) -> str:
    """Strip the markdown/math sigils that would otherwise pollute search text."""
    for ch in "*`$\\{}_^":
        text = text.replace(ch, " ")
    return " ".join(text.split())


def render_explorer(tree: dict) -> str:
    """Card grid of every experiment across setups, with a details dialog."""
    if not tree:
        return _NO_RESULTS
    setups = {s["key"]: s for s in scoreboard.setups()}
    tiles = [
        (problem, suite, experiment, pngs, owner.key)
        for problem in [p for p in scoreboard.PROBLEM_ORDER if p in tree]
        + sorted(p for p in tree if p not in scoreboard.PROBLEM_ORDER)
        for suite, experiments in tree[problem].items()
        for experiment, pngs in experiments.items()
        if (
            owner := views.iv.setup_of(
                problem, f"{suite}/{experiment}" if experiment else suite
            )
        )
        is not None
        and owner.key in setups
    ]
    present = {t[4] for t in tiles}
    lines = [
        "```{=html}",
        '<div class="explorer-toolbar" data-target="exp-grid">'
        '<label class="search-box"><span class="search-icon" aria-hidden="true"></span>'
        '<input type="search" class="explorer-search" placeholder="Search experiments, '
        'e.g. horizon, cost, recovery…" aria-label="Search experiments"></label>'
        '<div class="chip-filters" data-filter="setup"><button type="button" '
        'class="chip-btn active" data-value="">All setups</button>'
        + "".join(
            f'<button type="button" class="chip-btn" data-value="{s["key"]}">'
            f"{s['setup']}</button>"
            for s in scoreboard.setups()
            if s["key"] in present
        )
        + '</div><div class="chip-filters" data-filter="suite"><button type="button" '
        'class="chip-btn active" data-value="">All suites</button>'
        + "".join(
            f'<button type="button" class="chip-btn" data-value="{s}">{SUITE_LABELS[s]}</button>'
            for s in _SUITE_ORDER
            if any(s in suites for suites in tree.values())
        )
        + '</div><span class="explorer-count" aria-live="polite"></span></div>',
        "```",
        "",
        "::: {#exp-grid .card-grid .exp-grid}",
        "",
    ]
    for problem, suite, experiment, pngs, key in tiles:
        plabel = PROBLEM_LABELS.get(problem, problem)
        pid = scoreboard.PROBLEM_IDS.get(problem, problem)
        setup_label = setups[key]["setup"]
        page = setup_page(key)
        slabel = SUITE_LABELS.get(suite, suite.title())
        exp_dir = experiment.split("/", 1)[0]
        if experiment:
            title = _exp_label(experiment)
            short = _experiment_description(problem, suite, exp_dir)
            anchor = _exp_anchor(suite, experiment)
        else:
            title = f"{slabel} overview"
            short = ""
            anchor = f"suite-{suite}"
        plot_desc = _plot_description(problem, suite, exp_dir)
        search = _plain(
            f"{title} {setup_label} {plabel} {pid} {slabel} {short}"
        ).lower()
        thumb = _img_src(problem, suite, experiment, pngs[0])
        lines += [
            (
                f'::: {{.exp-tile data-setup="{key}" data-suite="{suite}" '
                f'data-search="{_md_attr(search)}" tabindex="0" role="button"}}'
            ),
            "",
            f'![]({thumb}){{.exp-thumb .nolightbox loading="lazy" alt=""}}',
            "",
            "::: {.exp-tile-body}",
            (
                f"[{pid}]{{.chip .chip-domain}} "
                f'[{slabel}]{{.chip .chip-suite data-suite="{suite}"}} '
                f"[{len(pngs)} plot{'s' if len(pngs) != 1 else ''}]"
                "{.chip .chip-quiet}"
            ),
            "",
            f"[{title}]{{.exp-tile-title}}",
            "",
        ]
        if short:
            lines += ["::: {.exp-tile-desc}", short, ":::", ""]
        lines += ["::: {.exp-tile-detail}", ""]
        lines += [f"*{setup_label}, {slabel.lower()}*", ""]
        if plot_desc:
            lines += [plot_desc, ""]
        for png in pngs:
            src = _img_src(problem, suite, experiment, png)
            lines += [
                f'[![]({src}){{.nolightbox loading="lazy"}}]({src}){{.exp-plot target="_blank"}}',
                "",
            ]
        lines += [
            f"[Open on the results page of this setup →]({page}#{anchor}){{.exp-open}}",
            "",
            ":::",
            ":::",
            ":::",
            "",
        ]
    lines += [":::", ""]
    return "\n".join(lines)


def copy_plots() -> int:
    """Copy the plots under mosaic-results/ into the Quarto project."""
    import shutil

    dest = OUTPUT_DIR / "plots"
    shutil.rmtree(dest, ignore_errors=True)
    if not RESULTS_DIR.exists():
        return 0
    n = 0
    for src in RESULTS_DIR.rglob("*"):
        if src.is_file() and src.name.endswith(_PLOT_SUFFIXES):
            target = dest / src.relative_to(RESULTS_DIR)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            n += 1
    return n


def write_overview(tree: dict, scores: dict | None) -> None:
    """Write the overview includes and the per-solver check JSON."""
    if scores:
        pages = {s["key"]: setup_page(s["key"]) for s in scoreboard.setups()}
        SCOREBOARD_INCLUDE.write_text(
            scoreboard.render_overview(scores, pages), encoding="utf-8"
        )
        SCORES_JSON.write_text(
            json.dumps(scoreboard.per_solver_summary(scores), indent=1),
            encoding="utf-8",
        )
    else:
        SCOREBOARD_INCLUDE.write_text(_NO_RESULTS, encoding="utf-8")
        SCORES_JSON.unlink(missing_ok=True)
    EXPLORER_INCLUDE.write_text(render_explorer(tree), encoding="utf-8")
    GATES_INCLUDE.write_text(scoreboard.render_gates(), encoding="utf-8")


def main() -> None:
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    tree = _scan_results()
    scores = compute_scores() if tree else None
    write_overview(tree, scores)
    print(f"Wrote results overview includes ({len(tree)} problems)")
    print(f"Copied {copy_plots()} plot files into {OUTPUT_DIR / 'plots'}")

    # Every setup gets a page, so the navbar links always resolve; setups
    # without results get a placeholder.
    for setup in scoreboard.setups():
        suites = _setup_suites(
            setup["problem"], tree.get(setup["problem"], {}), setup["key"]
        )
        out_path = OUTPUT_DIR / "results" / setup_page(setup["key"])
        out_path.write_text(
            generate_qmd_for_setup(setup, suites, timestamp, scores), encoding="utf-8"
        )
        n_plots = sum(len(pngs) for exps in suites.values() for pngs in exps.values())
        print(f"Generated {out_path} ({n_plots} plots)")


if __name__ == "__main__":
    main()
