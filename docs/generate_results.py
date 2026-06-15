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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "mosaic-results"
OUTPUT_DIR = Path(__file__).resolve().parent  # docs/

# Relative path from docs/results_*.qmd to mosaic-results/
_IMG_BASE = "../mosaic-results"

# Suite traversal order
_SUITE_ORDER = ["ics", "forward", "gradient", "optimization", "cost"]

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
    "structural-mesh": "Structural Mechanics",
    "thermal-mesh": "Heat Conduction",
}

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
        "similarity) across parameter, resolution, and horizon sweeps — the latter "
        "exposing how gradients degrade as the rollout lengthens."
    ),
    "optimization": (
        "**Can you optimise through it?** End-to-end optimization benchmarks run a "
        "gradient-based optimiser using each solver's own gradients: recovery of "
        "initial conditions or physical parameters, topology optimization, and drag "
        "minimization. This is the ultimate test — a gradient can pass the "
        "finite-difference check yet still fail to drive a full optimization loop."
    ),
    "cost": (
        "**What does it cost?** Wall-clock scaling of the forward and VJP passes "
        "with problem size $N$ and the number of integration steps.\n"
        "\n"
        "::: {.callout-note title='Note on wall-clock measurements'}\n"
        "Cost-suite timings are collected on dedicated CI runners "
        "with no concurrent benchmark workloads. Relative solver rankings within a "
        "single run are reliable; absolute wall times may vary ±10–15% across runs "
        "due to cloud VM variability.\n"
        ":::"
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

    if desc:
        lines += [desc, ""]

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
