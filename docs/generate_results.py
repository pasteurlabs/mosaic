#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate per-domain docs/results_<slug>.qmd files from benchmark result plots.

Usage:
    python docs/generate_results.py           # writes docs/results_*.qmd
    python docs/generate_results.py --check   # exit 1 if any file is stale
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "mosaic-results"
OUTPUT_DIR = Path(__file__).resolve().parent  # docs/

# Relative path from docs/results_*.qmd to mosaic-results/
_IMG_BASE = "../mosaic-results"

# Suite traversal order
_SUITE_ORDER = ["ics", "forward", "cost", "gradient", "optimization"]

# Experiment traversal order within each suite
_EXPERIMENT_ORDER = [
    "agreement",
    "convergence",
    "diagnostics",
    "stability",
    "fd_check",
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

# Schematic of each benchmark task (control variable, physical process,
# optimization objective), shared with the paper. Path is relative to the
# generated docs/results_*.qmd files. Optional \u2014 omitted if the file is absent.
PROBLEM_ILLUSTRATIONS = {
    "ns-grid": "figures/domain_ns_grid.png",
    "ns-3d-grid": "figures/domain_ns_3d_grid.png",
    "structural-mesh": "figures/domain_structural_mesh.png",
    "thermal-mesh": "figures/domain_thermal_mesh.png",
}

# One-line "what is this task" caption shown under the schematic, aimed at a
# reader who knows ML but not the specific solver community.
PROBLEM_TAGLINES = {
    "ns-grid": (
        "Optimize the inflow of a 2D channel flow to minimize drag on a "
        "cylinder, differentiating through an incompressible fluid solver."
    ),
    "ns-3d-grid": (
        "Recover the initial velocity field of a 3D turbulent flow from a "
        "later snapshot, differentiating through the Navier\u2013Stokes rollout."
    ),
    "structural-mesh": (
        "Place a fixed budget of material in a clamped beam to make it as "
        "stiff as possible, differentiating through a finite-element solve."
    ),
    "thermal-mesh": (
        "Invert for the conductivity field of a slab from its temperature, "
        "differentiating through a steady heat-conduction solve."
    ),
}

# Shared "how the results were produced" callout, surfaced near the top of
# every results page (hardware + reliability \u2014 issue 6). The benchmark runs on
# GitHub Actions runners: GPU solvers on a Tesla T4 node, CPU-only solvers
# (OpenFOAM, deal.II, FEniCS, Firedrake) on a CPU node. Wall times therefore
# reflect commodity cloud hardware, not a tuned workstation.
RESULTS_PROVENANCE = (
    "These are **example results**, produced automatically on GitHub Actions "
    "runners and refreshed on every release. Each solver runs on its intended "
    "device: GPU-capable solvers on a Tesla T4 GPU node, CPU-only solvers "
    "(OpenFOAM, deal.II, FEniCS, Firedrake) on a CPU node. Accuracy and "
    "gradient metrics are hardware-independent and reproducible. Wall-clock "
    "numbers reflect commodity cloud hardware and can vary by 10\u201315% between "
    "runs, so read them for relative scaling between solvers rather than as "
    "absolute timings. For numbers that reflect *your* setup, "
    "[run the benchmarks yourself](getting-started.qmd) on your target hardware."
)

PROBLEM_DESCRIPTIONS = {
    "ns-grid": (
        "JAX-CFD, PhiFlow, INS.jl, XLB, PICT, Warp-NS, and OpenFOAM on 2D "
        "incompressible Navier\u2013Stokes, sweeping viscosity \u03bd. "
        "Solvers span spectral, finite-difference, finite-volume, and LBM schemes."
    ),
    "ns-3d-grid": (
        "PhiFlow, XLB, PICT, Warp-NS, Exponax, INS.jl, and OpenFOAM on 3D "
        "incompressible Navier\u2013Stokes (triply periodic TGV). "
        "Initial condition recovery with 12k degrees of freedom."
    ),
    "structural-mesh": (
        "deal.II, FEniCS, Firedrake, JAX-FEM, and TopOpt.jl on 2D cantilever "
        "compliance minimization with SIMP penalization."
    ),
    "thermal-mesh": (
        "deal.II, FEniCS, Firedrake, JAX-FEM, and torch-fem on 2D steady-state "
        "heat conduction. Conductivity inversion from temperature observations."
    ),
}

PROBLEM_BC_DESCRIPTIONS = {
    "ns-grid": (
        "Channel domain [0,8]\u00d7[0,2] with cylinder obstacle at (2,1), radius 0.25. "
        "Inflow on left, advective outflow on right, no-slip walls top/bottom. "
        "Periodic TGV experiments use [0,2\u03c0]\u00b2."
    ),
    "ns-3d-grid": (
        "Triply periodic box [0,2\u03c0]\u00b3. Incompressibility enforced via "
        "pressure projection at each time step."
    ),
    "structural-mesh": (
        "2D cantilever beam on domain [0,2]\u00d7[0,1]. "
        "Dirichlet: all nodes at x=0 have zero displacement (clamped). "
        "Neumann: point load at right-center."
    ),
    "thermal-mesh": (
        "2D square domain [0,1]\u00b2 with Dirichlet boundary conditions. "
        "Spatially varying conductivity field as control variable."
    ),
}

SUITE_LABELS = {
    "ics": "Initial Conditions",
    "forward": "Forward",
    "gradient": "Gradient",
    "optimization": "Optimization",
    "cost": "Cost",
}

SUITE_DESCRIPTIONS = {
    "ics": (
        "Visualisation of each initial condition (the starting field a run is "
        "launched from) available for this problem. "
        "IC plots are generated without running any solver."
    ),
    "forward": (
        "**Is the prediction right?** Forward-pass benchmarks check each solver's "
        "output against a trusted reference (and an analytic solution where one "
        "exists): inter-solver agreement, field-level diagnostics, and long-run "
        "stability."
    ),
    "gradient": (
        "**Is the gradient right?** Gradient benchmarks compare each solver's "
        "AD/adjoint gradient against a finite-difference ground truth. We report "
        "magnitude error (relative $L^2$) and direction agreement (cosine "
        "similarity) across parameter, resolution, and horizon sweeps. The horizon "
        "sweep in particular exposes how gradients degrade as the rollout lengthens."
    ),
    "optimization": (
        "**Can you optimize through it?** End-to-end optimization benchmarks run a "
        "gradient-based optimizer using each solver's own gradients: recovery of "
        "initial conditions or physical parameters, topology optimization, and drag "
        "minimization. This is the ultimate test, since a gradient can pass the "
        "finite-difference check yet still fail to drive a full optimization loop."
    ),
    "cost": (
        "**What does it cost?** Wall-clock scaling of the forward and VJP passes "
        "with problem size $N$ and the number of integration steps. Timings come "
        "from dedicated runners with no concurrent workloads; see the reliability "
        "note at the top of the page before reading absolute numbers."
    ),
}

EXPERIMENT_LABELS = {
    "agreement": "Agreement",
    "convergence": "Convergence",
    "diagnostics": "Diagnostics",
    "stability": "Stability",
    "fd_check": "Finite-Difference Check",
    "param_sweep": "Parameter Sweep",
    "resolution_sweep": "Resolution Sweep",
    "horizon_sweep": "Horizon Sweep",
    "viscosity_recovery": "Viscosity Recovery",
    "ic_recovery": "IC Recovery",
    "param_recovery": "Parameter Recovery",
    "topopt": "Topology Optimisation",
    "recovery": "Recovery",
    "recovery3d": "Recovery 3D",
    "agreement3d": "Agreement 3D",
    "horizon_sweep3d": "Horizon Sweep 3D",
    "fd_check3d": "Finite-Difference Check 3D",
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
    (jax, jaxlib, matplotlib, scipy, ...), which is hundreds of MB resident.
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
        desc = _get_config(problem).description
        if desc:
            return desc
    except Exception:
        pass
    return PROBLEM_DESCRIPTIONS.get(problem, "")


def _bc_description(problem: str) -> str:
    """Return boundary-condition description from Problem."""
    try:
        bc = _get_config(problem).bc_description
        if bc:
            return bc
    except Exception:
        pass
    return PROBLEM_BC_DESCRIPTIONS.get(problem, "")


# ── Per-suite "best solver" leaderboard (issue 8) ──────────────────────────────
#
# Each suite is scored on its headline metric, computed directly from the
# result.json files so the table can't drift from the plots. Rules are
# deliberately conservative: a solver appears only when its metric is finite,
# and a suite's table is emitted only when at least two solvers can be ranked
# (a one-row "leaderboard" says nothing). All metrics here are
# hardware-independent, so the ranking is reproducible across runs.


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


def _rank_forward(problem: str) -> tuple[list[str], list[tuple]] | None:
    """Rank solvers by mean relative error vs the reference (lower is better)."""
    # Prefer the viscosity/parameter sweep; fall back to the first forward
    # experiment dir that carries an ``error`` metric.
    candidates = ["tgv_nu_sweep", "agreement", "baseline", "source_baseline"]
    by_solver: dict[str, list[float]] = {}
    for exp in candidates:
        for r in _load_suite_results(problem, "forward", exp):
            m = r.get("metrics") or {}
            if "error" in m and m.get("valid", True):
                by_solver.setdefault(r["solver"], []).append(m["error"])
        if by_solver:
            break
    rows = sorted(
        ((s, _mean_finite(v)) for s, v in by_solver.items()),
        key=lambda t: (math.isnan(t[1]), t[1]),
    )
    rows = [(s, e) for s, e in rows if math.isfinite(e)]
    if len(rows) < 2:
        return None
    header = ["Solver", "Mean rel. error"]
    return header, [(s, _fmt_sci(e)) for s, e in rows]


def _rank_gradient(problem: str) -> tuple[list[str], list[tuple]] | None:
    """Rank by best-ε FD error (lower better); show direction cosine alongside."""
    by_solver: dict[str, tuple[float, float]] = {}
    for r in _load_suite_results(problem, "gradient", "fd_check"):
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
    rows = sorted(by_solver.items(), key=lambda t: t[1][0])
    if len(rows) < 2:
        return None
    # Cosines sit so close to 1 that a fixed-decimal column reads as a wall of
    # "1.00000"; report the defect 1 − cos instead, where smaller is better.
    header = ["Solver", "Best-ε FD error", "1 − cosine"]
    return header, [
        (s, _fmt_sci(err), _fmt_sci(1.0 - cos) if math.isfinite(cos) else "—")
        for s, (err, cos) in rows
    ]


def _cost_at_max_n(results: list[dict]) -> dict[str, tuple[float, float]]:
    """For each solver, return ``(N, mean_s)`` at the largest N it completed.

    Picking the time at the max successful size avoids flattering a solver that
    OOMs early. ``results`` is a flat cost ``results`` list (sweep_value = N).
    """
    out: dict[str, tuple[float, float]] = {}
    for r in results:
        m = r.get("metrics") or {}
        if m.get("status") == "failed":
            continue
        mean_s = m.get("mean")
        n = r.get("sweep_value")
        if not isinstance(mean_s, int | float) or not math.isfinite(mean_s):
            continue
        nf = float(n) if isinstance(n, int | float | str) and str(n) else float("nan")
        prev = out.get(r["solver"])
        if prev is None or nf > prev[0]:
            out[r["solver"]] = (nf, float(mean_s))
    return out


def _rank_cost(problem: str) -> tuple[list[str], list[tuple]] | None:
    """Rank by forward wall-clock, reporting both forward and VJP time.

    Forward time comes from the ``spatial_cost`` sweep and VJP time from
    ``vjp_cost/by_N``; each is taken at the largest N that solver completed
    (shown in the "at N" column). Forward-only solvers have no VJP entry, which
    shows as "—". Ranked by forward time (faster wins).
    """
    fwd = _cost_at_max_n(_load_suite_results(problem, "cost", "spatial_cost"))
    vjp = _cost_at_max_n(_load_suite_results(problem, "cost", "vjp_cost/by_N"))
    if len(fwd) < 2:
        return None

    def _time(v: tuple[float, float] | None) -> str:
        # Annotate each time with its own N: forward and VJP can top out at
        # different sizes (a solver may OOM on the heavier backward pass first).
        if not v:
            return "—"
        n, t = v
        n_str = f" @ N={int(n)}" if math.isfinite(n) else ""
        return f"{t:.3g} s{n_str}"

    rows = sorted(fwd.items(), key=lambda t: t[1][1])
    header = ["Solver", "Forward time", "VJP time"]
    return header, [(s, _time(fv), _time(vjp.get(s))) for s, fv in rows]


# final-objective metric name by domain optimization experiment
_OPT_FINAL_KEYS = ("final_error", "final_drag", "final_compliance")


def _rank_optimization(problem: str) -> tuple[list[str], list[tuple]] | None:
    """Rank by final optimization objective (lower is better)."""
    # Find the first optimization experiment dir with a recognised final metric.
    opt_dir = RESULTS_DIR / problem / "optimization"
    if not opt_dir.exists():
        return None
    rows: list[tuple[str, float, bool]] = []
    metric_label = None
    for exp in sorted(p.name for p in opt_dir.iterdir() if p.is_dir()):
        for r in _load_suite_results(problem, "optimization", exp):
            m = r.get("metrics") or {}
            key = next((k for k in _OPT_FINAL_KEYS if k in m), None)
            if key is None:
                continue
            val = m.get(key)
            if isinstance(val, int | float) and math.isfinite(val):
                metric_label = key.replace("final_", "final ").replace("_", " ")
                rows.append((r["solver"], float(val), bool(m.get("converged", False))))
        if rows:
            break
    if len(rows) < 2:
        return None
    rows.sort(key=lambda t: t[1])
    header = ["Solver", metric_label.capitalize(), "Converged"]
    return header, [(s, _fmt_sci(v), "yes" if conv else "no") for s, v, conv in rows]


_SUITE_RANKERS = {
    "forward": _rank_forward,
    "gradient": _rank_gradient,
    "cost": _rank_cost,
    "optimization": _rank_optimization,
}

_SUITE_RANK_CAPTIONS = {
    "forward": "Ranked by mean relative error against the reference solution "
    "(lower is more accurate).",
    "gradient": "Ranked by the best-ε finite-difference error of the gradient "
    "(lower is more trustworthy); direction cosine near 1 confirms the gradient "
    "points the right way.",
    "cost": "Forward and VJP (backward) wall-clock time, each shown at the "
    "largest problem size N the solver completed for that pass; ranked by "
    "forward time (faster is better). Forward-only solvers have no VJP entry. "
    "See the reliability note above before comparing across devices.",
    "optimization": "Ranked by the final objective reached within the iteration "
    "budget (lower is better).",
}


def _leaderboard_block(problem: str, suite: str) -> list[str]:
    """Return QMD lines for the per-suite best-solver table, or [] if none.

    Rendered as a plain, always-visible table (no collapse, no emoji) under a
    bold "Solver ranking" lead-in, suited to a scientific report.
    """
    ranker = _SUITE_RANKERS.get(suite)
    if ranker is None:
        return []
    try:
        ranked = ranker(problem)
    except Exception:
        return []
    if not ranked:
        return []
    header, rows = ranked
    # Wrap the table in a .sortable-table div so docs/sortable-tables.html
    # (injected via include-after-body) can attach click-to-sort handlers
    # (numeric-aware). Static and correctly rank-ordered without JS, so it
    # degrades gracefully.
    # Rows are already in rank order, so a Rank column adds nothing; rely on row
    # order (and click-to-sort) instead.
    lines = [
        "**Solver ranking**",
        "",
        "::: {.sortable-table}",
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines += [":::"]
    caption = _SUITE_RANK_CAPTIONS.get(suite, "")
    if caption:
        lines += ["", f"*{caption}*"]
    lines += [""]
    return lines


# ── Helpers ───────────────────────────────────────────────────────────────────


def _slug(problem: str) -> str:
    return problem.replace("-", "_")


def _output_path(problem: str) -> Path:
    return OUTPUT_DIR / f"results_{_slug(problem)}.qmd"


def _sort_pngs(paths: list[Path]) -> list[Path]:
    def key(p: Path) -> tuple[int, str]:
        stem = p.stem
        for i, prefix in enumerate(_PNG_ORDER):
            if stem == prefix or stem.startswith(prefix + "_"):
                return (i, stem)
        return (len(_PNG_ORDER), stem)

    return sorted(paths, key=key)


def _img_tag(problem: str, suite: str, experiment: str, png: Path) -> str:
    if experiment:
        return (
            f"![]({_IMG_BASE}/{problem}/{suite}/{experiment}/{png.name}){{.lightbox}}"
        )
    return f"![]({_IMG_BASE}/{problem}/{suite}/{png.name}){{.lightbox}}"


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
        except Exception:
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
            lines += ["", f"**{name.replace('_', ' ').title()}**"]
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

            # Suite-level PNGs/GIFs (e.g. cost.png sitting directly in the suite dir)
            suite_pngs = _sort_pngs(
                [f for f in suite_dir.iterdir() if f.suffix in (".png", ".gif")]
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
                pngs = _sort_pngs(
                    [f for f in exp_dir.iterdir() if f.suffix in (".png", ".gif")]
                )
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
                        [f for f in sub_dir.iterdir() if f.suffix in (".png", ".gif")]
                    )
                    if not sub_pngs:
                        continue
                    sub_key = f"{exp_dir.name}/{sub_dir.name}"
                    tree.setdefault(problem, {}).setdefault(suite, {})[sub_key] = (
                        sub_pngs
                    )
    return tree


def generate_qmd_for_problem(problem: str, suites: dict, timestamp: str) -> str:
    """Generate QMD content for a single problem domain."""
    label = PROBLEM_LABELS.get(problem, problem.replace("-", " ").title())
    desc = _problem_description(problem)

    n_plots = sum(len(pngs) for exps in suites.values() for pngs in exps.values())

    lines: list[str] = [
        "---",
        f"title: {label}",
        "---",
        "",
        f"> Auto-generated {timestamp} &nbsp;·&nbsp; {n_plots} plots",
        "",
    ]

    # Task schematic + one-line tagline (skipped if the figure is missing, so a
    # checkout without docs/figures/ still renders cleanly).
    illustration = PROBLEM_ILLUSTRATIONS.get(problem)
    if illustration and (OUTPUT_DIR / illustration).exists():
        lines += [
            f"![]({illustration}){{width=100% "
            'style="max-width:560px; display:block; margin:0 auto 0.5rem;"}',
            "",
        ]
        tagline = PROBLEM_TAGLINES.get(problem)
        if tagline:
            lines += [f"*{tagline}*", ""]

    if desc:
        lines += [desc, ""]

    # How the results were produced — hardware + reliability (issue 6).
    lines += [
        "::: {.callout-tip title='How these results were produced' collapse='true'}",
        "",
        RESULTS_PROVENANCE,
        "",
        ":::",
        "",
    ]

    bc = _bc_description(problem)
    if bc:
        lines += [
            "::: {.callout-note title='Boundary conditions'}",
            "",
            bc,
            "",
            ":::",
            "",
        ]

    for suite, experiments in suites.items():
        suite_label = SUITE_LABELS.get(suite, suite.title())
        suite_desc = SUITE_DESCRIPTIONS.get(suite, "")

        lines += [f"## {suite_label}", ""]
        if suite_desc:
            lines += [suite_desc, ""]

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
                # Build label: for sub-experiments use "Parent (Sub)" format
                if is_sub:
                    parent_exp, sub_exp = experiment.split("/", 1)
                    parent_label = EXPERIMENT_LABELS.get(
                        parent_exp, parent_exp.replace("_", " ").title()
                    )
                    sub_label = sub_exp.replace("_", " ").title()
                    exp_label = f"{parent_label} ({sub_label})"
                    exp_dir_name = parent_exp
                else:
                    exp_label = EXPERIMENT_LABELS.get(
                        experiment, experiment.replace("_", " ").title()
                    )
                    exp_dir_name = experiment
                short_desc = _experiment_description(problem, suite, exp_dir_name)
                plot_desc = _plot_description(problem, suite, exp_dir_name)
                params_path = RESULTS_DIR / problem / suite / experiment / "params.json"

                lines += [f"### {exp_label}", ""]
                if short_desc:
                    lines += [short_desc, ""]
                if plot_desc:
                    lines += [plot_desc, ""]
                lines += _params_block(params_path)

            lines.append("")
            for png in pngs:
                lines.append(_img_tag(problem, suite, experiment, png))
            lines.append("")

        # Best-solver leaderboard, shown below the suite's plots (issue 8).
        lines += _leaderboard_block(problem, suite)

    return "\n".join(lines)


def main() -> None:
    from datetime import datetime, timezone

    check_mode = "--check" in sys.argv
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    tree = _scan_results()
    if not tree:
        # No plottable results. In --check mode this is a failure (someone
        # expected generated pages); in normal generation (e.g. a docs build
        # where benchmark artifacts are absent or contain only metadata like
        # snapshot.json) it is a clean no-op — exit 0 so the docs build
        # doesn't fail.
        msg = f"No benchmark results found under {RESULTS_DIR}"
        if check_mode:
            sys.exit(msg)
        print(f"{msg} — nothing to generate.")
        return

    stale: list[str] = []
    for problem, suites in tree.items():
        new_qmd = generate_qmd_for_problem(problem, suites, timestamp)
        out_path = _output_path(problem)

        if check_mode:
            if not out_path.exists() or out_path.read_text(encoding="utf-8") != new_qmd:
                stale.append(out_path.name)
        else:
            out_path.write_text(new_qmd, encoding="utf-8")
            n_plots = sum(
                len(pngs) for exps in suites.values() for pngs in exps.values()
            )
            print(f"Generated {out_path} ({n_plots} plots)")

    if check_mode:
        if stale:
            sys.exit(
                f"Stale files: {', '.join(stale)}. "
                "Run `python docs/generate_results.py` to regenerate."
            )
        print("All per-domain results QMD files are up to date.")


if __name__ == "__main__":
    main()
