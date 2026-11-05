# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml==6.0.3"]
# ///
"""Regenerate the paper's data-driven LaTeX tables from a Mosaic results directory.

Usage:
    uv run paper/make_tables.py [--out-dir DIR] [--status FILE]
        [--v100-results DIR] [--t4-results DIR] [--extra-results DIR]

Results are read from ``$MOSAIC_RESULTS_DIR`` if set, else ``./mosaic-results``.
``outcomes16gb.tex`` needs the two 16 GB runs (``--v100-results``,
``--t4-results``) and is skipped without them.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import count_loc
from plots import (
    by_solver,
    device_mib,
    failure_type,
    is_ok,
    load_result,
    results_dir,
    solver_metadata,
    solver_wrappers,
)

PROBLEMS = ["ns-grid", "ns-3d-grid", "structural-mesh", "thermal-mesh"]
PROBLEM_LABEL = {
    "ns-grid": "F2",
    "ns-3d-grid": "F3",
    "structural-mesh": "S",
    "thermal-mesh": "H",
}
# Grid size of each problem's row in Table 3 (merged.tex).
TABLE_SIZE = {
    "ns-grid": "64",
    "ns-3d-grid": "32",
    "structural-mesh": "128",
    "thermal-mesh": "256",
}

# Display label and citation key per solver name in the results.
SOLVERS = {
    "Exponax": ("Exponax", "koehler2024apebench"),
    "INS.jl": ("INS.jl", "agdestein2024ins"),
    "jax-cfd": ("JAX-CFD", "kochkov2021machine"),
    "OpenFOAM": ("OpenFOAM", "weller1998tensorial"),
    "PhiFlow": ("PhiFlow", "holl2024phiflow"),
    "PICT": ("PICT", "franz2025pict"),
    "Warp-NS": ("Warp-NS", "macklin2024warp"),
    "XLB": ("XLB", "ataei2024xlb"),
    "deal.II": ("deal.II", "bangerth2007dealii"),
    "FEniCS": ("FEniCS", "farrell2013automated"),
    "Firedrake": ("Firedrake", "rathgeber2016firedrake"),
    "JAX-FEM": ("JAX-FEM", "xue2023jaxfem"),
    "TopOpt.jl": ("TopOpt.jl", "huang2021topoptjl"),
    "torch-fem": ("torch-fem", "meyer2024torchfem"),
}

# Solver names of the V100 submission run, which used snake_case keys.
V100_NAMES = {
    "ins_jl": "INS.jl",
    "exponax": "Exponax",
    "phiflow": "PhiFlow",
    "xlb": "XLB",
    "warp_ns": "Warp-NS",
    "pict": "PICT",
    "jax_cfd": "jax-cfd",
    "openfoam": "OpenFOAM",
    "jax_fem": "JAX-FEM",
    "topopt_jl": "TopOpt.jl",
    "dealii_structural": "deal.II",
    "fenics_structural": "FEniCS",
    "firedrake_structural": "Firedrake",
    "dealii_heat": "deal.II",
    "fenics_heat": "FEniCS",
    "firedrake_heat": "Firedrake",
    "torch_fem_thermal": "torch-fem",
    "fenics_ns": "FEniCS-NS",
    "su2": "SU2",
}


# ── Formatting ──────────────────────────────────────────────────────────────


def fmt_sig(x: float) -> str:
    """Two significant figures, plain notation (for times and ratios)."""
    digits = 1 - math.floor(math.log10(x))
    if digits <= 0:
        return f"{round(x, digits):.0f}"
    return f"{x:.{digits}f}"


def fmt_err(x: float) -> str:
    """Two significant figures in compact scientific notation, e.g. ``1.9e-3``."""
    if x == 0:
        return "0"
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant}e{int(exp)}"


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}\\%" if x < 0.1 else f"{100 * x:.0f}\\%"


def bold_min(
    cells: dict[str, str | None], values: dict[str, float]
) -> dict[str, str | None]:
    """Bold the cells showing the smallest value, unless every shown value is the same."""
    if not values or len({cells[s] for s in values}) == 1:
        return cells
    best = cells[min(values, key=values.get)]
    return {
        s: (f"\\textbf{{{c}}}" if s in values and c == best else c)
        for s, c in cells.items()
    }


def ragged(width: float) -> str:
    """Column spec of a ragged-right paragraph column, *width* in units of \\linewidth."""
    return rf">{{\raggedright\arraybackslash}}p{{{width}\linewidth}}"


def size_label(key: str, value: str) -> str:
    """Sweep point as text, e.g. ``$N=64$`` or ``500 steps``."""
    if key == "steps":
        return f"{value} step{'s' if value != '1' else ''}"
    return f"${'n_x' if key == 'nx' else key}={value}$"


def solver_cell(name: str) -> str:
    label, key = SOLVERS[name]
    return f"{label}~\\cite{{{key}}}"


def write(out_dir: Path, name: str, text: str) -> None:
    path = out_dir / name
    path.write_text(text)
    print(f"Wrote {path}")


# ── Result helpers ──────────────────────────────────────────────────────────


def load_status(path: Path) -> dict[tuple[str, str, str], dict]:
    """``mosaic status`` cells keyed by (problem, "suite/experiment", solver)."""
    data = json.loads(path.read_text())
    return {
        (prob, row["label"], solver): cell
        for prob, pd in data["problems"].items()
        for row in pd["rows"]
        for solver, cell in row["cells"].items()
    }


def mean_times(problem: str, experiment: str) -> dict[tuple[str, str], float]:
    """Mean wall time of every successful (solver, sweep value)."""
    return {
        (s, v): m["mean"]
        for s, pts in by_solver(load_result(problem, experiment)).items()
        for v, m in pts.items()
        if is_ok(m) and m.get("mean") is not None
    }


def best_fd_error(metrics: dict) -> float:
    """Min over ε of the median relative error across FD directions (as in ``mosaic status``)."""
    meds = []
    for entry in metrics["eps_sweep"].values():
        errs = [e for e in entry["rel_error"] if e is not None and math.isfinite(e)]
        if errs:
            meds.append(statistics.median(errs))
    return min(meds)


def fd_errors(problem: str, experiment: str = "gradient/fd_check") -> dict[str, float]:
    return {
        r["solver"]: best_fd_error(r["metrics"])
        for r in load_result(problem, experiment)["results"]
        if is_ok(r["metrics"])
    }


def errors_at(
    problem: str, experiment: str, sweep_value: str | None
) -> dict[str, float]:
    """``error`` per solver at one sweep value, or the maximum over the sweep if None."""
    out: dict[str, float] = {}
    for s, pts in by_solver(load_result(problem, experiment)).items():
        vals = [
            m["error"]
            for v, m in pts.items()
            if (sweep_value is None or v == sweep_value)
            and is_ok(m)
            and m.get("error") is not None
        ]
        if vals:
            out[s] = max(vals)
    return out


def solver_order(problem: str) -> list[str]:
    """Differentiable solvers alphabetically, then forward-only ones."""
    meta = solver_metadata(problem)
    return sorted(meta, key=lambda s: (meta[s]["ad_strategy"] is None, s.lower()))


# ── Table 3: merged.tex ─────────────────────────────────────────────────────

BLOCKS = [
    {
        "problem": "ns-grid",
        "title": "Navier--Stokes grid (F2), wall time in ms",
        "scale": 1e3,
        "ref": ("forward/agreement/multimode", "0.01"),
        "an": ("forward/agreement/tgv", "0.01"),
        "task": "optimization/drag_opt",
        "grads": [
            ("$v_0$", "gradient/fd_check"),
            ("$\\nu$", "gradient/analytic_check_nu"),
            ("inflow", "optimization/drag_opt"),
        ],
    },
    {
        "problem": "ns-3d-grid",
        "title": "Navier--Stokes grid (F3), wall time in ms",
        "scale": 1e3,
        "ref": ("forward/agreement", "0.01"),
        "an": None,
        "task": "optimization/recovery_constant_ic_bfgs_proj",
        "grads": [("$v_0$", "gradient/fd_check")],
    },
    {
        "problem": "structural-mesh",
        "title": "Structural mechanics (S), wall time in s",
        "scale": 1.0,
        "ref": ("forward/agreement", None),
        "an": None,
        "task": "optimization/topopt",
        "grads": [("$\\rho$", "gradient/fd_check")],
    },
    {
        "problem": "thermal-mesh",
        "title": "Heat transfer (H), wall time in s",
        "scale": 1.0,
        "ref": ("forward/agreement", None),
        "an": None,
        "task": "optimization/conductivity_recovery_bfgs",
        "grads": [
            ("$\\rho$", "gradient/fd_check"),
            ("src", "gradient/source_fd_check"),
        ],
    },
]

LOC_COLUMNS = ("IO", "Num.", "Conf.")


def line_counts(problem: str) -> dict[str, tuple[int, int, int]]:
    """Bespoke lines (IO, Num., Conf.) per solver wrapper of *problem*.

    The Navier--Stokes wrappers serve both grid problems, so F2 and F3 share
    their counts.
    """
    counts = count_loc.line_counts()
    return {
        solver: tuple(counts[key][col] for col in ("IO", "Num", "Conf"))
        for solver, key in solver_wrappers(problem).items()
    }


def grad_symbol(cell: dict | None, ad_strategy: str | None) -> str:
    """Gradient-support symbol from a status cell.

    ● for ``ad_strategy: autodiff`` (gradients by AD through the solver),
    ○ for ``adjoint`` and ``hybrid`` (an adjoint or a hand-written VJP rule
    supplies at least part of the gradient).
    """
    if ad_strategy is None:
        return "$-$"
    status = (cell or {}).get("status")
    if status == "ok":
        return "$\\bullet$" if ad_strategy == "autodiff" else "$\\circ$"
    if status == "excluded":
        return "$\\bot$"
    if status in (None, "not_run"):
        return "$?$"
    return "$\\times$"


def conv_symbol(cell: dict | None, ad_strategy: str | None) -> str:
    if ad_strategy is None:
        return "$-$"
    status = (cell or {}).get("status")
    return {
        "ok": "$\\checkmark$",
        "anomaly": "$\\triangle$",
        "excluded": "$\\bot$",
        "not_run": "$?$",
        None: "$?$",
    }.get(status, "$\\times$")


def merged_table(status: dict) -> tuple[str, dict]:
    n_grad = max(len(b["grads"]) for b in BLOCKS)
    trials = {
        p: load_result(p, "cost/spatial_cost")["params"]["cost"]["n_trials"]
        for p in PROBLEMS
    }
    rows = []
    summary = {}
    for b in BLOCKS:
        p, size = b["problem"], TABLE_SIZE[b["problem"]]
        meta = solver_metadata(p)
        solvers = solver_order(p)
        fwd = {
            s: t * b["scale"]
            for (s, v), t in mean_times(p, "cost/spatial_cost").items()
            if v == size
        }
        vjp = {
            s: t * b["scale"]
            for (s, v), t in mean_times(p, "cost/vjp_cost/by_N").items()
            if v == size
        }
        ref = errors_at(p, *b["ref"])
        an = errors_at(p, *b["an"]) if b["an"] else {}
        fd = fd_errors(p)
        loc = line_counts(p)
        summary[PROBLEM_LABEL[p]] = {"fwd": fwd, "vjp": vjp, "fd": fd}

        cols = {}
        for key, vals, fmt in (
            ("fwd", fwd, fmt_sig),
            ("vjp", vjp, fmt_sig),
            ("ref", ref, fmt_err),
            ("an", an, fmt_err),
            ("fd", fd, fmt_err),
        ):
            cells = {s: (fmt(vals[s]) if s in vals else None) for s in solvers}
            cells = bold_min(cells, vals)
            missing = "$-$" if key == "an" and not b["an"] else None
            for s in solvers:
                if cells[s] is None:
                    no_grad = meta[s]["ad_strategy"] is None and key in ("vjp", "fd")
                    cells[s] = missing or ("$-$" if no_grad else "$?$")
            cols[key] = cells
        for i, col in enumerate(LOC_COLUMNS):
            vals = {s: loc[s][i] for s in solvers}
            cells = {s: str(vals[s]) for s in solvers}
            cols[col] = bold_min(cells, vals)

        heads = [h for h, _ in b["grads"]] + [""] * (n_grad - len(b["grads"]))
        g0 = 8
        rows.append("\\midrule")
        rows.append(
            f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{b['title']}}}}}\n      & "
            + " & ".join(f"\\textit{{{h}}}" if h else "" for h in heads)
            + " & & & \\\\"
        )
        rows.append(f"\\cmidrule(lr){{{g0}-{g0 + len(b['grads']) - 1}}}")
        for s in solvers:
            ad = meta[s]["ad_strategy"]
            grads = [grad_symbol(status.get((p, exp, s)), ad) for _, exp in b["grads"]]
            grads += [""] * (n_grad - len(grads))
            cells = [
                solver_cell(s),
                cols["fwd"][s],
                cols["vjp"][s],
                cols["ref"][s],
                cols["an"][s],
                cols["fd"][s],
                conv_symbol(status.get((p, b["task"], s)), ad),
                *grads,
                *(cols[col][s] for col in LOC_COLUMNS),
            ]
            rows.append("  " + " & ".join(cells) + " \\\\")

    ns_trials, mesh_trials = trials["ns-grid"], trials["structural-mesh"]
    body = "\n    ".join(rows)
    text = rf"""\begin{{table}}
  \centering\footnotesize
  \setlength{{\tabcolsep}}{{2.5pt}}
  \caption{{Benchmark results, gradient support, and integration effort per solver.
    Wall-clock times are means of {ns_trials} (F2, F3) or {mesh_trials} (S, H) timed runs after
    1 warmup, at $N={TABLE_SIZE["ns-grid"]}$ (F2), $N={TABLE_SIZE["ns-3d-grid"]}$ (F3),
    $n_x={TABLE_SIZE["structural-mesh"]}$ (S), and $n_x={TABLE_SIZE["thermal-mesh"]}$ (H).
    Full specifications in \cref{{app:effort}}; initial conditions in
    \cref{{app:ics}}. Legend below table.}}
  \label{{tab:merged}}
  \vspace{{2pt}}
  \begin{{tabular}}{{@{{}}l r r c c c c | {"c" * n_grad} | r r r@{{}}}}
    \toprule
    \textbf{{Solver}}
      & \multicolumn{{2}}{{c}}{{\textbf{{Wall time}}}}
      & \multicolumn{{3}}{{c}}{{\textbf{{Error}}}}
      & \textbf{{Conv.}}
      & \multicolumn{{{n_grad}}}{{c|}}{{\textbf{{Grad.\ support}}}}
      & \multicolumn{{3}}{{c@{{}}}}{{\textbf{{Integ.\ effort}}}} \\
    \cmidrule(lr){{2-3}}\cmidrule(lr){{4-6}}\cmidrule(lr){{8-{7 + n_grad}}}\cmidrule(lr){{{8 + n_grad}-{10 + n_grad}}}
    & \textit{{Fwd}} & \textit{{VJP}} & \textit{{Ref.}} & \textit{{An.}} & \textit{{FD}} &
      & \multicolumn{{{n_grad}}}{{c|}}{{\textit{{input variables}}}}
      & \textit{{IO}} & \textit{{Num.}} & \textit{{Conf.}} \\
    {body}
    \bottomrule
  \end{{tabular}}
  \par\vspace{{2pt}}
  \begin{{minipage}}{{\linewidth}}\scriptsize\raggedright
    \textit{{Error}} (relative): Ref.\,=\,vs.\ reference at $\nu=0.01$ (F2: solver consensus,
    multimode IC; F3: Exponax at $N=64$), or the largest over the density sweep vs.\ the solver
    consensus (S, H); An.\,=\,vs.\ analytical Taylor--Green solution at $\nu=0.01$;
    FD\,=\,vs.\ central finite differences (median over directions, best~$\varepsilon$).
    \textit{{Conv.}}\ (F2 drag, F3 L-BFGS IC recovery, S topology opt., H conductivity recovery):
    $\checkmark$\,=\,passes the checks; $\triangle$\,=\,flagged; $\times$\,=\,failed.
    \textit{{Grad.\ support}}: $\bullet$\,=\,AD through the solver; $\circ$\,=\,adjoint or
    hand-written VJP for at least part of it; $\times$\,=\,incorrect; $?$\,=\,not obtained;
    inflow evidenced only by a converged drag optimization.
    $\bot$\,=\,excluded (\cref{{app:solver-issues}}); $-$\,=\,not applicable.
    \textit{{Integ.\ effort}} (bespoke LOC): IO\,=\,input/output marshalling;
    Num.\,=\,numerics \& gradient code; Conf.\,=\,solver configuration.
  \end{{minipage}}
\end{{table}}
"""
    return text, summary


# ── VJP/forward ratios: vjp_ratios.tex ──────────────────────────────────────


def ratios(problem: str, fwd_exp: str, vjp_exp: str) -> dict[str, dict[str, float]]:
    fwd, vjp = mean_times(problem, fwd_exp), mean_times(problem, vjp_exp)
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (s, v), t in vjp.items():
        if (s, v) in fwd:
            out[s][v] = t / fwd[(s, v)]
    return out


def ratio_range(d: dict[str, float]) -> str:
    return f"{fmt_sig(min(d.values()))}--{fmt_sig(max(d.values()))}"


def ratio_table() -> tuple[str, dict]:
    rows = []
    summary = {}
    for p in PROBLEMS:
        by_n = ratios(p, "cost/spatial_cost", "cost/vjp_cost/by_N")
        ns = p.startswith("ns")
        by_steps = (
            ratios(p, "cost/temporal_cost", "cost/vjp_cost/by_steps") if ns else {}
        )
        size = TABLE_SIZE[p]
        n_key = "N" if ns else "n_x"
        steps_n = (
            load_result(p, "cost/temporal_cost")["params"]["physics"]["N"]
            if ns
            else None
        )
        sizes = f"table size ${n_key}={size}$" + (
            f"; rollout sweep at $N={steps_n}$" if ns else ""
        )
        rows += [
            "\\midrule",
            f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{PROBLEM_LABEL[p]}: {sizes}}}}} \\\\",
        ]
        summary[PROBLEM_LABEL[p]] = {}
        for s in solver_order(p):
            if s not in by_n:
                continue
            r = by_n[s]
            at = fmt_sig(r[size]) if size in r else "$?$"
            steps = ratio_range(by_steps[s]) if s in by_steps else "$-$"
            rows.append(f"  {SOLVERS[s][0]} & {at} & {ratio_range(r)} & {steps} \\\\")
            summary[PROBLEM_LABEL[p]][s] = (
                r.get(size),
                min(r.values()),
                max(r.values()),
            )
    body = "\n    ".join(rows)
    text = rf"""\begin{{table}}[t]
  \centering\small
  \caption{{Ratio of VJP to forward wall time per solver, from mean times: at the
    \cref{{tab:merged}} size, and the range over the grid-size sweep and, for the fluid
    domains, the rollout-length sweep. The VJP request includes a forward pass, so about~2 is
    the expected floor for a solver whose gradient costs one adjoint solve.}}
  \label{{tab:app:ratios}}
  \begin{{tabular}}{{@{{}}l c c c@{{}}}}
    \toprule
    \textbf{{Solver}} & \textbf{{Table size}} & \textbf{{Grid sweep}} & \textbf{{Rollout sweep}} \\
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
    return text, summary


# ── Timing variance: variance.tex ───────────────────────────────────────────

EXP_LABEL = {
    "cost/spatial_cost": "fwd",
    "cost/temporal_cost": "fwd",
    "cost/vjp_cost/by_N": "VJP",
    "cost/vjp_cost/by_steps": "VJP",
    "gradient/horizon_sweep": "horizon",
    "gradient/horizon_sweep_limits": "horizon",
}
COST_EXPERIMENTS = [
    "cost/spatial_cost",
    "cost/temporal_cost",
    "cost/vjp_cost/by_N",
    "cost/vjp_cost/by_steps",
]


def variance_table(n_worst: int = 3) -> tuple[str, dict]:
    rows = []
    summary = {}
    all_cvs = []
    for p in PROBLEMS:
        cfgs = []
        for exp in COST_EXPERIMENTS:
            res = load_result(p, exp)
            key = res["sweep"]["key"]
            for s, pts in by_solver(res).items():
                for v, m in pts.items():
                    t = m.get("trials_s")
                    if is_ok(m) and t and len(t) >= 2:
                        cfgs.append(
                            (statistics.stdev(t) / statistics.fmean(t), s, exp, key, v)
                        )
        cvs = [c[0] for c in cfgs]
        all_cvs += cvs
        worst = sorted(cfgs, reverse=True)[:n_worst]
        worst_txt = "; ".join(
            f"{SOLVERS[s][0]} {EXP_LABEL[e]} {size_label(k, v)} ({fmt_pct(cv)})"
            for cv, s, e, k, v in worst
        )
        n_hi = sum(c > 0.15 for c in cvs)
        med = statistics.median(cvs)
        rows.append(
            f"  {PROBLEM_LABEL[p]} & {len(cvs)} & {fmt_pct(med)} & {n_hi} & {worst_txt} \\\\"
        )
        summary[PROBLEM_LABEL[p]] = (len(cvs), med, n_hi)
    summary["all"] = (
        len(all_cvs),
        statistics.median(all_cvs),
        sum(c > 0.15 for c in all_cvs),
    )
    n, med, n_hi = summary["all"]
    rows.append("\\midrule")
    rows.append(f"  All & {n} & {fmt_pct(med)} & {n_hi} & \\\\")
    body = "\n    ".join(rows)
    text = rf"""\begin{{table}}[t]
  \centering\footnotesize
  \caption{{Run-to-run variation of the timed cost measurements: coefficient of variation (CV,
    sample standard deviation with $n-1$ over the mean of the timed trials) per configuration,
    i.e.\ per solver, cost experiment and sweep value, with the {n_worst} configurations of
    highest CV per problem.}}
  \label{{tab:app:variance}}
  \begin{{tabular}}{{@{{}}l r r r {ragged(0.48)}@{{}}}}
    \toprule
    \textbf{{Problem}} & \textbf{{Configs}} & \textbf{{Median CV}} & \textbf{{CV\,>\,15\%}} & \textbf{{Highest CV}} \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
    return text, summary


# ── Gradient checks: gradient_checks.tex ────────────────────────────────────


def gradient_checks_table() -> tuple[str, dict]:
    p = "ns-grid"
    fd = fd_errors(p)
    an = errors_at(p, "gradient/analytic_check", "0.01")
    nu = errors_at(p, "gradient/analytic_check_nu", "0.01")
    solvers = [s for s in solver_order(p) if s in fd]
    cols = [
        bold_min({s: fmt_err(d[s]) for s in solvers}, {s: d[s] for s in solvers})
        for d in (fd, an, nu)
    ]
    rows = [
        f"  {SOLVERS[s][0]} & " + " & ".join(c[s] for c in cols) + " \\\\"
        for s in solvers
    ]
    body = "\n    ".join(rows)
    text = rf"""\begin{{table}}[t]
  \centering\small
  \caption{{Gradient checks on F2. FD: relative error of the initial-condition VJP against
    central finite differences (median over directions at the best~$\varepsilon$).
    Analytic: relative error of the initial-condition gradient against the exact Taylor--Green
    gradient; $\mathrm{{d}}J/\mathrm{{d}}\nu$: relative error of the viscosity gradient against
    the exact one; both at $\nu=0.01$.}}
  \label{{tab:gradchecks}}
  \begin{{tabular}}{{@{{}}l c c c@{{}}}}
    \toprule
    \textbf{{Solver}} & \textbf{{FD}} & \textbf{{Analytic}} & $\mathrm{{d}}J/\mathrm{{d}}\nu$ \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
    return text, {"fd": fd, "analytic": an, "nu": nu}


# ── 16 GB outcomes: outcomes16gb.tex ────────────────────────────────────────

# Experiments compared, as stored in the current results.
OUTCOME_EXPERIMENTS = {
    "ns-grid": COST_EXPERIMENTS + ["gradient/horizon_sweep"],
    "ns-3d-grid": COST_EXPERIMENTS + ["gradient/horizon_sweep_limits"],
    "structural-mesh": ["cost/spatial_cost", "cost/vjp_cost/by_N"],
    "thermal-mesh": ["cost/spatial_cost", "cost/vjp_cost/by_N"],
}
SWEEP_LABEL = {
    "cost/spatial_cost": "fwd, size",
    "cost/temporal_cost": "fwd, steps",
    "cost/vjp_cost/by_N": "VJP, size",
    "cost/vjp_cost/by_steps": "VJP, steps",
    "gradient/horizon_sweep": "horizon",
    "gradient/horizon_sweep_limits": "horizon",
}
# The V100 run stored each cost experiment once, with one block per sweep.
V100_LOCATION = {
    "cost/spatial_cost": ("cost/spatial_cost", "by_N"),
    "cost/temporal_cost": ("cost/temporal_cost", "by_steps"),
    "cost/vjp_cost/by_N": ("cost/vjp_cost", "by_N"),
    "cost/vjp_cost/by_steps": ("cost/vjp_cost", "by_steps"),
    "gradient/horizon_sweep": ("gradient/horizon_sweep", "by_solver"),
    "gradient/horizon_sweep_limits": ("gradient/horizon_sweep_limits", "by_solver"),
}
# Experiments of the extended mesh sweep (--extra-results).
EXTRA_EXPERIMENT = {
    "cost/spatial_cost": "cost/spatial_cost_large",
    "cost/vjp_cost/by_N": "cost/vjp_cost_large/by_N",
}
FAILURE_LABEL = {
    "OOM": "OOM",
    "nan": "NaN",
    "container_died": "died",
    "timeout": "timeout",
    "error": "error",
}


T4_DEVICE_MIB = 15360


def failure_label(m: dict | None, device: float | None = None) -> str | None:
    """Short outcome label of a failed entry, or None if it succeeded."""
    if m is None:
        return "no reason"
    if is_ok(m):
        return None
    if m.get("status") == "skipped":
        return "skipped"
    ft = failure_type(m, device)
    return FAILURE_LABEL.get(ft, ft or "error")


def v100_failures(root: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for p, exps in OUTCOME_EXPERIMENTS.items():
        for exp in exps:
            rel, block = V100_LOCATION[exp]
            path = root / p / rel / "result.json"
            if not path.exists():
                continue
            for s, pts in json.loads(path.read_text()).get(block, {}).items():
                for v, m in pts.items():
                    if (lab := failure_label(m)) is not None:
                        out[(p, exp, V100_NAMES.get(s, s))][v] = lab
    return out


def v1_failures(
    root: Path, device: float | None
) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for p, exps in OUTCOME_EXPERIMENTS.items():
        for exp in exps:
            path = root / p / exp / "result.json"
            if not path.exists():
                continue
            for r in json.loads(path.read_text())["results"]:
                if (lab := failure_label(r.get("metrics"), device)) is not None:
                    out[(p, exp, r["solver"])][str(r["sweep_value"])] = lab
    return out


def compress(pairs: list[tuple[str, str]]) -> str:
    """Render sorted (value, label) pairs, merging runs with the same label into ranges."""
    groups: list[list] = []
    for v, lab in pairs:
        if groups and groups[-1][2] == lab:
            groups[-1][1] = v
        else:
            groups.append([v, v, lab])
    return ", ".join(f"{a}{'--' + b if b != a else ''} {lab}" for a, b, lab in groups)


def gb(mib: float) -> str:
    return f"{mib / 1024:.2g}" if mib < 10 * 1024 else f"{mib / 1024:.0f}"


def a100_outcomes(
    p: str, exp: str, solver: str, values: list[str], status: dict, extra: Path | None
) -> str:
    """Summarize the A100 outcome of *solver* at the sweep *values* of *exp*."""
    meta = solver_metadata(p)
    if solver not in meta:
        return "solver not in suite"
    cell = status.get((p, exp, solver)) or {}
    if cell.get("status") == "excluded":
        return "excluded"
    pts = by_solver(load_result(p, exp)).get(solver, {})
    if extra is not None and exp in EXTRA_EXPERIMENT:
        path = extra / p / EXTRA_EXPERIMENT[exp] / "result.json"
        if path.exists():
            extra_pts = by_solver(json.loads(path.read_text())).get(solver, {})
            pts = {**extra_pts, **pts}
    mem_key = "vram_peak_mib" if meta[solver]["uses_gpu"] else "ram_peak_mib"
    parts: list[list] = []  # [first, last, label, max memory of the ok points]
    for v in values:
        m = pts.get(v)
        lab = (
            "not in sweep"
            if m is None
            else ("ok" if is_ok(m) else failure_label(m, device_mib(results_dir())))
        )
        mem = m.get(mem_key) if lab == "ok" else None
        if parts and parts[-1][2] == lab and lab in ("ok", "not in sweep", "skipped"):
            parts[-1][1] = v
            parts[-1][3] = max(parts[-1][3] or 0, mem or 0)
        else:
            parts.append([v, v, lab, mem])
    out = []
    for a, b, lab, mem in parts:
        txt = f"{lab} {a if a == b else f'{a}--{b}'}"
        if lab == "ok" and mem is not None:
            txt += f" ({gb(mem)}\\,GB)"
        out.append(txt)
    return "; ".join(out)


def outcomes_table(
    status: dict, v100_root: Path, t4_root: Path, extra: Path | None
) -> tuple[str, dict]:
    runs = {
        "V100": v100_failures(v100_root),
        "T4": v1_failures(t4_root, device_mib(t4_root, T4_DEVICE_MIB)),
    }
    keys = sorted(
        set(runs["V100"]) | set(runs["T4"]),
        key=lambda k: (
            PROBLEMS.index(k[0]),
            OUTCOME_EXPERIMENTS[k[0]].index(k[1]),
            k[2].lower(),
        ),
    )
    rows = []
    prev = None
    for p, exp, s in keys:
        if prev is not None and prev != p:
            rows.append("\\midrule")
        prev = p
        failed = []
        for run in ("V100", "T4"):
            pts = runs[run].get((p, exp, s))
            if pts:
                pairs = sorted(pts.items(), key=lambda kv: float(kv[0]))
                failed.append(f"{run}: {compress(pairs)}")
        values = sorted(
            {v for run in runs.values() for v in run.get((p, exp, s), {})}, key=float
        )
        label = SOLVERS.get(s, (s,))[0]
        rows.append(
            f"  {PROBLEM_LABEL[p]} & {SWEEP_LABEL[exp]} & {label} & {'; '.join(failed)} & "
            f"{a100_outcomes(p, exp, s, values, status, extra)} \\\\"
        )

    body = "\n    ".join(rows)
    died_note = (
        r" \emph{died}: the container exited otherwise;"
        if any(" died" in r for r in rows)
        else ""
    )
    text = rf"""\begin{{table}}[p]
  \centering\scriptsize
  \setlength{{\tabcolsep}}{{3pt}}
  \caption{{Configurations that failed on a 16\,GB GPU, in the V100 submission run or the T4
    CI run, and their outcome on the A100 80\,GB. Sizes are $N$ or $n_x$ for the size sweeps and
    rollout steps for the \emph{{steps}} and \emph{{horizon}} sweeps. The V100 run stored most
    failures without a reason (\emph{{no reason}}). \emph{{skipped}}: not attempted after an
    earlier failure in the same sweep; \emph{{OOM}} includes containers that exited with the GPU's
    memory full;{died_note} \emph{{not in sweep}}:
    outside the sweep range of the A100 run. Memory is the largest peak over the successful
    points, VRAM for GPU solvers and host RAM for CPU solvers, with 1\,GB = $2^{{30}}$ bytes.
    FEniCS-NS and SU2 are no longer part of the suite.}}
  \label{{tab:app:outcomes}}
  \begin{{tabular}}{{@{{}}l l l {ragged(0.33)} {ragged(0.33)}@{{}}}}
    \toprule
    & \textbf{{Sweep}} & \textbf{{Solver}} & \textbf{{16\,GB failures}} & \textbf{{A100 outcome}} \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
    return text, {"rows": len(rows)}


# ── Main ────────────────────────────────────────────────────────────────────


def print_summary(merged: dict, rat: dict, var: dict, grad: dict) -> None:
    print("\nKey numbers")
    for blk, d in merged.items():
        f = min(d["fwd"], key=d["fwd"].get)
        v = min(d["vjp"], key=d["vjp"].get)
        print(
            f"  {blk}: fastest fwd {f} {fmt_sig(d['fwd'][f])}, fastest VJP {v} {fmt_sig(d['vjp'][v])}"
            f" ({'ms' if blk.startswith('F') else 's'}); FD error "
            f"{fmt_err(min(d['fd'].values()))}--{fmt_err(max(d['fd'].values()))}"
        )
    for blk, d in rat.items():
        lo = min(r[1] for r in d.values())
        hi = max(r[2] for r in d.values())
        top = max(d, key=lambda s: d[s][2])
        print(
            f"  {blk} VJP/fwd over grid sweep: {fmt_sig(lo)}--{fmt_sig(hi)} (max {top})"
        )
    for k, (n, med, n_hi) in var.items():
        print(f"  CV {k}: {n} configs, median {100 * med:.1f}%, {n_hi} above 15%")
    for s in grad["fd"]:
        print(
            f"  F2 {s}: FD {fmt_err(grad['fd'][s])}, analytic {fmt_err(grad['analytic'][s])},"
            f" dJ/dnu {fmt_err(grad['nu'][s])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "tables",
        help="Output directory (default: paper/tables).",
    )
    parser.add_argument(
        "--status",
        type=Path,
        help="`mosaic status` JSON of the results (default: status.json next to the results).",
    )
    parser.add_argument(
        "--v100-results", type=Path, help="Results of the V100 submission run."
    )
    parser.add_argument("--t4-results", type=Path, help="Results of the T4 CI run.")
    parser.add_argument(
        "--extra-results",
        type=Path,
        help="Results of the extended mesh cost sweeps (cost/*_large experiments).",
    )
    args = parser.parse_args()

    print(f"Reading results from {results_dir()}")
    status = load_status(args.status or results_dir().parent / "status.json")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged, merged_sum = merged_table(status)
    write(args.out_dir, "merged.tex", merged)
    rat, rat_sum = ratio_table()
    write(args.out_dir, "vjp_ratios.tex", rat)
    var, var_sum = variance_table()
    write(args.out_dir, "variance.tex", var)
    grad, grad_sum = gradient_checks_table()
    write(args.out_dir, "gradient_checks.tex", grad)
    if args.v100_results and args.t4_results:
        out, out_sum = outcomes_table(
            status, args.v100_results, args.t4_results, args.extra_results
        )
        write(args.out_dir, "outcomes16gb.tex", out)
        print(f"  outcomes: {out_sum['rows']} 16 GB rows")
    else:
        print("Skipping outcomes16gb.tex (needs --v100-results and --t4-results)")

    print_summary(merged_sum, rat_sum, var_sum, grad_sum)


if __name__ == "__main__":
    main()
