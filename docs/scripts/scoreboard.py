# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-setup solver check tables for the Results pages.

The tables report each metric separately and never combine them into one
score. For each problem *setup* (one problem statement, e.g. periodic
Taylor–Green flow or flow past a cylinder; the setups and the experiments each
one owns are defined in ``mosaic/benchmarks/core/interventions.py``) they
report:

* **Checks** — forward, gradient and optimization, each one of pass,
  flagged, fail, blocked, not supported or not run. A check
  combines the status of the setup's gate experiments for that suite, using the
  pass criteria (``status_check``) each problem config declares for them; see
  :func:`_check`. Cells excluded because the method cannot express the
  experiment are "not supported", not failures; cells excluded for a recorded
  reason such as an instability or an upstream bug are "blocked".
* **Cost** — forward and VJP wall-clock time, and its multiple of the
  fastest solver *on the same device class* (GPU or CPU, from each solver's
  config), at one problem size per device class (see :func:`comparison_n`).
  Next to it, the peak memory of the VJP pass (the forward pass for
  forward-only solvers).
* **Effort** and **deviations** — from each solver's intervention record.

Below each table, notes list every experiment of the setup (diagnostics such
as Jacobian spectra and cost sweeps included) that a solver did not complete
within its criteria, with the reason, and the sweeps where a solver reached a
resource limit.

Everything is computed from ``mosaic-results/snapshot.json`` (the
``mosaic status`` snapshot), the cost ``result.json`` files and the
intervention records.
"""

from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path


def _load_views():
    """Load docs/scripts/intervention_views.py whether this runs as a script or a module."""
    import importlib.util

    name = "_mosaic_intervention_views"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent / "intervention_views.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


views = _load_views()

SUITES = ["forward", "gradient", "optimization"]
SUITE_LABELS = {
    "forward": "Forward",
    "gradient": "Gradient",
    "optimization": "Optimization",
}

PROBLEM_ORDER = ["thermal-mesh", "structural-mesh", "ns-grid", "ns-3d-grid"]
PROBLEM_IDS = {
    "thermal-mesh": "H",
    "structural-mesh": "S",
    "ns-grid": "F2",
    "ns-3d-grid": "F3",
}

# How each setup is shown, in display order. Which experiments belong to a
# setup comes from ``interventions.SETUPS``; an entry whose key is not defined
# there is skipped. ``gates`` are the experiments whose status decides each
# check (an empty list drops the column for that setup). ``accuracy_cost``
# adds a column with the cost of reaching an error (see :func:`cost_to_error`).
SETUPS = [
    {
        "key": "heat",
        "short": "slab",
        "problem": "thermal-mesh",
        "label": "Heat transfer",
        "task": "conductivity recovery",
        "gates": {
            "forward": [
                "forward/agreement",
                "forward/baseline",
                "forward/physical_laws",
                "forward/source_baseline",
                "forward/source_linearity",
            ],
            "gradient": [
                "gradient/fd_check",
                "gradient/param_sweep",
                "gradient/source_fd_check",
                "gradient/source_width_sweep",
            ],
            "optimization": ["optimization/conductivity_recovery_bfgs"],
        },
    },
    {
        "key": "structural",
        "short": "cantilever",
        "problem": "structural-mesh",
        "label": "Structural mechanics",
        "task": "compliance topology optimization",
        "gates": {
            "forward": [
                "forward/agreement",
                "forward/baseline",
                "forward/physical_laws",
            ],
            "gradient": ["gradient/fd_check", "gradient/param_sweep"],
            "optimization": ["optimization/topopt"],
        },
    },
    {
        "key": "ns-2d-periodic",
        "short": "periodic",
        "problem": "ns-grid",
        "label": "Navier–Stokes (2D)",
        "task": "",
        "gates": {
            "forward": [
                "forward/agreement",
                "forward/baseline",
                "forward/physical_laws",
                "forward/tgv_nu_sweep",
            ],
            "gradient": [
                "gradient/fd_check",
                "gradient/analytic_check",
                "gradient/analytic_check_nu",
                "gradient/param_sweep",
            ],
            "optimization": [],
        },
    },
    {
        "key": "ns-2d-cylinder",
        "short": "cylinder",
        "problem": "ns-grid",
        "label": "Navier–Stokes (2D)",
        "task": "inflow optimization for drag",
        "gates": {
            "forward": ["forward/cylinder"],
            "gradient": [],
            "optimization": ["optimization/drag_opt"],
        },
    },
    {
        "key": "ns-2d-kolmogorov",
        "short": "Kolmogorov",
        "problem": "ns-grid",
        "label": "Navier–Stokes (2D)",
        "task": "",
        "gates": {
            "forward": ["forward/kolmogorov_laminar", "forward/kolmogorov"],
            "gradient": ["gradient/chain_fd_check"],
            "optimization": [],
        },
        # The flow is chaotic, so what separates the solvers is how much
        # compute they need to track the converged truth.
        "accuracy_cost": {"experiment": "forward/kolmogorov", "error": 0.1},
    },
    {
        "key": "ns-3d",
        "short": "periodic",
        "problem": "ns-3d-grid",
        "label": "Navier–Stokes (3D)",
        "task": "initial-condition recovery",
        "gates": {
            "forward": [
                "forward/agreement",
                "forward/baseline",
                "forward/physical_laws",
            ],
            "gradient": ["gradient/fd_check"],
            "optimization": ["optimization/recovery"],
        },
    },
]

# Check states, in order of severity (worst last).
STATES = ["pass", "flagged", "fail", "blocked", "na", "notrun"]
STATE_LABELS = {
    "pass": "Pass",
    "flagged": "Flagged",
    "fail": "Fail",
    "blocked": "Blocked",
    "na": "Not supported",
    "notrun": "Not run",
}
STATE_GLYPHS = {
    "pass": "✓",
    "flagged": "!",
    "fail": "✗",
    "blocked": "⊘",
    "na": "—",
    "notrun": "·",
}

# Cost passes: (column label, experiment dir under cost/).
COST_PASSES = [("Forward cost", "spatial_cost"), ("VJP cost", "vjp_cost/by_N")]

_PERMANENT = {"categorical"}
# Exclusions whose recorded reason is a finding about the solver on this setup
# (the ``ExclusionCategory`` values in mosaic/benchmarks/core/config.py). The
# remaining work-to-do categories (``wip``, ``unspecified``) read as not run.
BLOCKED_LABELS = {
    "unstable": "Unstable",
    "infeasible": "Infeasible",
    "upstream_bug": "Upstream bug",
    "not_implemented": "Not implemented",
}

# Snapshot cell metrics shown on hover, with their labels.
_METRIC_LABELS = {
    "fwd_error": "error",
    "rel_err": "error",
    "cosine": "cos",
    "final_ratio": "objective ratio",
    "final_loss": "final loss",
}


def _finite(v) -> bool:
    return isinstance(v, int | float) and math.isfinite(v)


def _matches(label: str, prefixes: list[str]) -> bool:
    return any(label == p or label.startswith((p + "/", p + "_")) for p in prefixes)


def setups() -> list[dict]:
    """The setups of :data:`SETUPS` that the intervention module defines, with
    their display name (``setup``) from it."""
    return [
        {**s, "setup": views.iv.SETUPS[s["key"]].label}
        for s in SETUPS
        if s["key"] in views.iv.SETUPS
    ]


def _setup_rows(setup: dict, rows: list[dict]) -> list[dict]:
    """Snapshot rows (one per experiment) belonging to ``setup``."""
    out = []
    for r in rows:
        if r["suite"] == "ics":
            continue
        owner = views.iv.setup_of(setup["problem"], r["label"])
        if owner is not None and owner.key == setup["key"]:
            out.append(r)
    return out


def _reason(cell: dict) -> str:
    """The snapshot's reason, trimmed to what a reader can act on."""
    why = (cell.get("reason") or "").strip()
    why = why.split("Traceback", 1)[0].rstrip(" :")
    why = why.removeprefix("error: ")
    why = {"nan": "result is NaN", "error": "errored (no message recorded)"}.get(
        why, why
    )
    # A documented explanation annotates the measured reason; it does not
    # change the cell's status (same rule as ``mosaic status``).
    note = (cell.get("explanation") or "").strip()
    return f"{why} (explained: {note})" if note else why


def _cell_kind(cell: dict) -> str:
    """ok | anomaly | failed | blocked | na | notrun for one snapshot cell."""
    st = cell.get("status")
    if st == "ok":
        return "ok"
    if st == "anomaly":
        return "anomaly"
    if st == "failed":
        return "failed"
    if st == "excluded" and cell.get("category") in _PERMANENT:
        return "na"
    if st == "excluded" and cell.get("category") in BLOCKED_LABELS:
        return "blocked"
    return "notrun"


def _blocked_label(cell: dict) -> str:
    return BLOCKED_LABELS.get(cell.get("category"), STATE_LABELS["blocked"])


def _check(cells: list[tuple[str, dict]]) -> dict:
    """Combine the gate cells of one suite into a check.

    Any failed gate fails the check; otherwise any anomaly (explained or not)
    flags it. A solver whose every gate is a permanent exclusion
    is "not supported"; one with no gate run is "blocked" if a gate is
    excluded for a recorded reason, and "not run" otherwise.
    """
    kinds = [(label, _cell_kind(c), c) for label, c in cells]
    run = [k for k in kinds if k[1] in ("ok", "anomaly", "failed")]
    blocked = [c for _, k, c in kinds if k == "blocked"]
    if not run:
        if kinds and all(k == "na" for _, k, _ in kinds):
            state = "na"
        elif blocked:
            state = "blocked"
        else:
            state = "notrun"
    elif any(k == "failed" for _, k, _ in run):
        state = "fail"
    elif any(k == "anomaly" for _, k, _ in run):
        state = "flagged"
    else:
        state = "pass"
    blocked_as = {_blocked_label(c) for c in blocked}
    return {
        "state": state,
        "n_run": len(run),
        "n_gates": len([k for k in kinds if k[1] != "na"]),
        "blocked_as": blocked_as.pop()
        if len(blocked_as) == 1
        else STATE_LABELS["blocked"],
        "metrics": [
            (label, c.get("metrics") or {})
            for label, k, c in kinds
            if k in ("ok", "anomaly")
        ],
    }


# ── Cost ─────────────────────────────────────────────────────────────────────


def cost_result(results_dir: Path, problem: str, exp: str) -> dict:
    path = results_dir / problem / "cost" / exp / "result.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _times_by_n(data: dict) -> dict[str, dict[float, dict]]:
    """Per solver and N: the mean, spread, trial count and first-call time."""
    out: dict[str, dict[float, dict]] = {}
    for r in data.get("results", []) or []:
        m = r.get("metrics") or {}
        if m.get("status") == "failed" or not _finite(m.get("mean")):
            continue
        try:
            n = float(r.get("sweep_value"))
        except (TypeError, ValueError):
            continue
        out.setdefault(r["solver"], {})[n] = {
            "mean": float(m["mean"]),
            "std": float(m["std"]) if _finite(m.get("std")) else math.nan,
            "n_trials": len(m.get("trials_s") or []),
            "first": float(m["first_call_s"])
            if _finite(m.get("first_call_s"))
            else math.nan,
            "vram": m.get("vram_peak_mib"),
            "ram": m.get("ram_peak_mib"),
        }
    return out


def hardware(data: dict) -> dict[str, dict]:
    """Per solver, the hardware its cost results were measured on.

    Merged results keep one record per solver, since the jobs ran on
    different machines; a single run has one record for all of them.
    """
    extras = data.get("extras") or {}
    if extras.get("hardware_by_solver"):
        return extras["hardware_by_solver"]
    single = extras.get("hardware")
    if not single:
        return {}
    return {r["solver"]: single for r in data.get("results", []) or []}


def _sweep_point(results_dir: Path, problem: str, label: str, value) -> str:
    """``N=64`` for a sweep value, or the bare value without a sweep key."""
    path = results_dir / problem / label / "result.json"
    try:
        key = (json.loads(path.read_text(encoding="utf-8")).get("sweep") or {})["key"]
    except Exception:
        return str(value)
    return f"{key}={value}"


def _gpu_names(record: dict) -> set[str]:
    return {str(g).split(",")[0].strip() for g in record.get("gpus") or []}


def comparison_n(
    passes: list[dict[str, dict[float, dict]]], group: list[str]
) -> float | None:
    """The one problem size N at which ``group``'s times are compared, in
    every pass.

    N is the largest size that every solver of the group completed in each of
    its passes, leaving out solvers that completed fewer than half the sizes
    the group reached in some pass, so that one solver failing early does not
    pull every comparison down to a small size. Solvers without a time at N
    are left out of the comparison.
    """
    have = [s for s in group if any(t.get(s) for t in passes)]
    reached = [{n for s in group for n in t.get(s, {})} for t in passes]

    def falls_short(s: str) -> bool:
        return any(
            t.get(s) and len(t[s]) < len(sizes) / 2
            for t, sizes in zip(passes, reached, strict=True)
        )

    core = [s for s in have if not falls_short(s)] or have
    done = [set(t[s]) for s in core for t in passes if t.get(s)]
    common = set.intersection(*done) if done else set()
    return max(common) if common else None


def _relative_cost(
    times: dict[str, dict[float, dict]], group: list[str], n: float | None
) -> dict[str, dict]:
    """``t / t_fastest`` within ``group`` at the size ``n`` from
    :func:`comparison_n`.

    A solver without a time at ``n`` gets only ``reach``, the largest N it
    completed. A solver is ``within_noise`` of the fastest when the gap
    between their means is at most twice the combined standard deviation of
    their trials.
    """
    compared = [s for s in group if n in times.get(s, {})]
    best = min((times[s][n] for s in compared), key=lambda t: t["mean"], default=None)
    out = {}
    for s in group:
        if not times.get(s):
            continue
        if s not in compared or best["mean"] <= 0:
            out[s] = {"ratio": None, "reach": max(times[s])}
            continue
        t = times[s][n]
        ratio = t["mean"] / best["mean"]
        spread = math.hypot(t["std"], best["std"])
        within_noise = (
            t is not best
            and math.isfinite(spread)
            and t["mean"] - best["mean"] <= 2 * spread
        )
        out[s] = {
            "ratio": ratio,
            "seconds": t["mean"],
            "std": t["std"],
            "n_trials": t["n_trials"],
            "first": t["first"],
            "vram": t["vram"],
            "ram": t["ram"],
            "n": n,
            "within_noise": within_noise,
        }
    return out


def plateau(result: dict) -> dict[str, dict]:
    """Per solver in an optimization ``result``, when the loss levelled off.

    ``iters`` is the harness's ``iters_to_plateau`` and ``seconds`` its
    ``seconds_to_plateau``. Results recorded before the harness kept the time
    get an estimate (``estimated``): the experiment's wall time for the solver
    times the share of its iterations run by then, which includes startup and
    compilation pro rata. Without ``n_iters`` (several trials whose lengths
    were not all recorded) or a wall time there is no estimate.
    """
    wall = (result.get("provenance") or {}).get("wall_time_s") or {}
    out = {}
    for r in result.get("results") or []:
        m = r.get("metrics") or {}
        iters = m.get("iters_to_plateau")
        if not _finite(iters):
            continue
        n, t = m.get("n_iters"), wall.get(r["solver"])
        seconds, estimated = m.get("seconds_to_plateau"), False
        if not _finite(seconds):
            estimated = True
            seconds = t * iters / n if _finite(n) and n > 0 and _finite(t) else None
        out[r["solver"]] = {
            "iters": float(iters),
            "n_iters": n,
            "seconds": seconds,
            "estimated": estimated,
        }
    return out


def _plateau_for(
    results_dir: Path, problem: str, gates: list[str] | None
) -> dict[str, dict]:
    """:func:`plateau` of the first optimization gate whose results carry it.

    A gate names an experiment or the prefix of its directory, as in
    :func:`_matches`.
    """
    suite_dir = results_dir / problem / "optimization"
    dirs = sorted(p.name for p in suite_dir.iterdir()) if suite_dir.is_dir() else []
    for label in [
        f"optimization/{d}" for d in dirs if _matches(f"optimization/{d}", gates or [])
    ]:
        path = results_dir / problem / label / "result.json"
        try:
            out = plateau(json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: S112 — no or malformed result file
            continue
        if out:
            return out
    return {}


def cost_to_error(result: dict, target: float) -> dict[str, dict]:
    """Per solver in a ``forward/kolmogorov`` result, the wall time per
    simulated time unit needed to reach the relative error ``target`` at the
    last scored time.

    The runs are a ladder of grids. Between the last grid above the target
    and the first one within it, the cost is interpolated linearly in
    log(error) and log(cost). ``state`` is ``reached``, ``coarsest`` when the
    coarsest grid that ran is already within the target, so that ``seconds``
    is an upper bound, ``not_reached`` with the ``best`` error, or
    ``diverged`` when no grid stayed finite.
    """
    out: dict[str, dict] = {}
    for r in result.get("results") or []:
        runs = sorted(
            (r.get("metrics") or {}).get("runs") or [], key=lambda x: x["grid"]
        )
        points = [
            (run["grid"], run["median_errors"][-1], run["seconds_per_time"])
            for run in runs
            if not run.get("diverged")
            and _finite(run.get("seconds_per_time"))
            and run.get("median_errors")
            and _finite(run["median_errors"][-1])
        ]
        if not points:
            if runs:
                out[r["solver"]] = {"state": "diverged"}
            continue
        entry = {"state": "not_reached", "best": min(e for _, e, _ in points)}
        prev = None
        for grid, err, cost in points:
            if err > target:
                prev = (grid, err, cost)
                continue
            if prev is None:
                entry = {"state": "coarsest", "seconds": cost, "grids": (grid,)}
            else:
                g0, e0, c0 = prev
                frac = (
                    (math.log(e0) - math.log(target)) / (math.log(e0) - math.log(err))
                    if err > 0
                    else 1.0
                )
                entry = {
                    "state": "reached",
                    "seconds": math.exp(
                        math.log(c0) + frac * (math.log(cost) - math.log(c0))
                    ),
                    "grids": (g0, grid),
                }
            break
        out[r["solver"]] = entry
    return out


def duration(t: float) -> str:
    """``42 s``, ``11 min`` or ``1.5 h``."""
    if t < 120:
        return f"{t:.2g} s" if t < 10 else f"{t:.0f} s"
    if t < 3600:
        return f"{t / 60:.2g} min" if t < 600 else f"{t / 60:.0f} min"
    return f"{t / 3600:.2g} h"


# ── Assembly ─────────────────────────────────────────────────────────────────


def _sort_key(row: dict, suites: list[str]):
    sev = [STATES.index(row["checks"][s]["state"]) for s in suites]
    blocking = sum(1 for v in sev if v >= STATES.index("fail"))
    soft = sum(1 for v in sev if v == STATES.index("flagged"))
    fwd = (row["cost"].get("Forward cost") or {}).get("ratio")
    fwd = math.inf if fwd is None else fwd
    return (blocking, soft, fwd, row["name"].lower())


def compute(results_dir: Path, solver_meta: dict[str, dict[str, dict]]) -> dict | None:
    """Return the per-setup tables, or None without a snapshot.

    ``solver_meta[problem][solver_name]`` supplies ``display`` (card name),
    ``anchor`` (Solver Reference id) and ``gpu`` (uses a GPU) per solver.
    """
    snap_path = results_dir / "snapshot.json"
    if not snap_path.exists():
        return None
    snapshot = json.loads(snap_path.read_text(encoding="utf-8")).get("problems", {})

    out = []
    for setup in setups():
        problem = setup["problem"]
        snap = snapshot.get(problem)
        if not snap:
            continue
        rows = _setup_rows(setup, snap.get("rows", []))
        if not rows:
            continue
        suites = [s for s in SUITES if setup["gates"].get(s)]
        meta = solver_meta.get(problem, {})
        names = list(snap.get("solvers", []))

        has_cost = any(r["suite"] == "cost" for r in rows)
        cost: dict[str, dict[str, dict]] = {}
        # Solvers are compared within the device class their config targets.
        # The hardware records only give the GPU model: a record lists the
        # machine's GPUs, whether or not the solver used them.
        gpu = {n: bool(meta.get(n, {}).get("gpu", True)) for n in names}
        gpu_models: set[str] = set()
        cost_n: dict[bool, float | None] = {}
        if has_cost:
            times: dict[str, dict[str, dict[float, dict]]] = {}
            for label, exp in COST_PASSES:
                data = cost_result(results_dir, problem, exp)
                for n, record in hardware(data).items():
                    if gpu.get(n):
                        gpu_models |= _gpu_names(record)
                times[label] = _times_by_n(data)
            for on_gpu in (True, False):
                group = [n for n in names if gpu[n] is on_gpu]
                n = comparison_n(list(times.values()), group)
                cost_n[on_gpu] = n
                for label, by_n in times.items():
                    for s, v in _relative_cost(by_n, group, n).items():
                        cost.setdefault(s, {})[label] = v

        spec = setup.get("accuracy_cost")
        accuracy_cost = {}
        if spec:
            path = results_dir / problem / spec["experiment"] / "result.json"
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # no or malformed result file
                result = {}
            accuracy_cost = cost_to_error(result, spec["error"])
        plateaus = _plateau_for(
            results_dir, problem, setup["gates"].get("optimization")
        )
        records = views.records_for(setup["key"])
        out_rows = []
        for name in names:
            m = meta.get(name, {})
            cells = [
                (r["label"], r["cells"][name])
                for r in rows
                if name in r.get("cells", {})
            ]
            if not cells:
                continue
            checks = {
                s: _check(
                    [(lab, c) for lab, c in cells if _matches(lab, setup["gates"][s])]
                )
                for s in suites
            }
            # A cost cell flagged only for being slow ran fine: the cost
            # columns already show its speed.
            kinds = [
                (
                    lab,
                    "ok"
                    if lab.startswith("cost/") and _cell_kind(c) == "anomaly"
                    else _cell_kind(c),
                    c,
                )
                for lab, c in cells
            ]
            flags = [
                (lab, _blocked_label(c) if k == "blocked" else k, _reason(c))
                for lab, k, c in kinds
                if k in ("anomaly", "failed", "blocked", "notrun")
            ]
            limits = [
                (lab, _sweep_point(results_dir, problem, lab, c["resource_frontier"]))
                for lab, _, c in kinds
                if c.get("resource_frontier")
            ]
            unsupported = [(lab, _reason(c)) for lab, k, c in kinds if k == "na"]
            out_rows.append(
                {
                    "name": m.get("display") or name,
                    "anchor": m.get("anchor", ""),
                    "gpu": gpu[name],
                    "checks": checks,
                    "plateau": plateaus.get(name),
                    "accuracy_cost": accuracy_cost.get(name),
                    "cost": cost.get(name, {}),
                    "flags": flags,
                    "limits": limits,
                    "unsupported": unsupported,
                    "effort": views.effort_parts(records.get(name), setup["key"]),
                    "deviations": views.deviation_items(
                        records.get(name), setup["key"]
                    ),
                }
            )
        out_rows.sort(key=lambda r: _sort_key(r, suites))
        out.append(
            {
                **{
                    k: setup[k]
                    for k in ("key", "short", "problem", "label", "setup", "task")
                },
                "id": PROBLEM_IDS.get(problem, problem),
                "suites": suites,
                "has_cost": has_cost and bool(cost),
                "cost_n": cost_n,
                "accuracy_target": spec["error"] if accuracy_cost else None,
                "gpu_model": gpu_models.pop() if len(gpu_models) == 1 else "",
                "has_effort": bool(records),
                "rows": out_rows,
            }
        )
    _mark_setups_elsewhere(out)
    return {"setups": out}


def _mark_setups_elsewhere(setups: list[dict]) -> None:
    """Give each setup ``elsewhere``: per solver of its table, the other
    setups of the same domain that the solver does not run, meaning that
    none of its checks there passed or was flagged.

    Solvers specialized to one kind of problem can top a table on it and run
    nothing else, so the check tables say so next to the numbers.
    """
    ran = {
        (st["key"], r["name"]): any(
            c["state"] in ("pass", "flagged") for c in r["checks"].values()
        )
        for st in setups
        for r in st["rows"]
    }
    for st in setups:
        domain = views.iv.domain_of(st["key"])
        others = [
            o for o in setups if o is not st and views.iv.domain_of(o["key"]) == domain
        ]
        st["elsewhere"] = {
            r["name"]: missing
            for r in st["rows"]
            if (
                missing := [
                    o for o in others if ran.get((o["key"], r["name"])) is False
                ]
            )
        }


# ── HTML rendering ───────────────────────────────────────────────────────────


def _join(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return " and ".join([", ".join(items[:-1]), items[-1]] if len(items) > 1 else items)


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _short(label: str) -> str:
    """``forward/agreement/tgv`` → ``agreement/tgv``."""
    return label.split("/", 1)[1] if "/" in label else label


def _metric_text(metrics: dict) -> str:
    parts = []
    for key, label in _METRIC_LABELS.items():
        v = metrics.get(key)
        if not _finite(v):
            continue
        parts.append(f"{label} {v:.5f}" if key == "cosine" else f"{label} {v:.2g}")
    return ", ".join(parts)


def _check_td(chk: dict) -> str:
    """One check cell, sorted by severity."""
    st = chk["state"]
    ran = st not in ("na", "notrun", "blocked")
    label = chk["blocked_as"] if st == "blocked" else STATE_LABELS[st]
    tip = [label]
    if chk["n_gates"] and ran:
        tip.append(
            f"Ran {chk['n_run']} of the {chk['n_gates']} experiments behind this check"
        )
    # Every gate's headline numbers, so that e.g. the finite-difference and
    # analytic gradient checks read side by side.
    tip += [
        f"{_short(lab)}: {txt}" for lab, m in chk["metrics"] if (txt := _metric_text(m))
    ]
    extra = (
        f'<span class="ck-part">{chk["n_run"]} of {chk["n_gates"]} run</span>'
        if ran and chk["n_run"] < chk["n_gates"]
        else ""
    )
    return (
        f'<td class="ck-cell" data-sort="{STATES.index(st)}">'
        f'<span class="ck ck-{st}" title="{_esc(chr(10).join(tip))}">'
        f'<span class="ck-glyph" aria-hidden="true">{STATE_GLYPHS[st]}</span>'
        f"{_esc(label)}</span>{extra}</td>"
    )


def _ratio_text(r: float) -> str:
    return "fastest" if r < 1.005 else (f"{r:.1f}×" if r < 10 else f"{r:.0f}×")


def _plateau_td(p: dict | None, timed: bool, best: float | None) -> str:
    """When the loss levelled off: the estimated time and its multiple of the
    fastest solver ``best`` in the group, or iterations where no solver of
    the setup has a time."""
    if not p:
        return '<td class="ck-cost ck-cost-na">—</td>'
    tip = f"Loss leveled off after {p['iters']:.0f}"
    if _finite(p["n_iters"]):
        tip += f" of {p['n_iters']:.0f}"
    tip += " iterations"
    if not timed:
        return (
            f'<td class="ck-cost" data-sort="{p["iters"]}" title="{_esc(tip)}">'
            f"{p['iters']:.0f}</td>"
        )
    t = p["seconds"]
    if t is None:
        return f'<td class="ck-cost ck-cost-na" title="{_esc(tip)}">—</td>'
    tip += (
        f", about {duration(t)} into the run (estimated from its wall time)"
        if p.get("estimated")
        else f", {duration(t)} into the run"
    )
    rel = f'<span class="ck-part">{_ratio_text(t / best)}</span>' if best else ""
    return (
        f'<td class="ck-cost" data-sort="{t}" title="{_esc(tip)}">'
        f"{duration(t)}{rel}</td>"
    )


def _accuracy_cost_td(v: dict | None, target: float, best: float | None) -> str:
    """The cost of reaching ``target`` error (:func:`cost_to_error`), and its
    multiple of the cheapest solver ``best`` in the group."""
    if not v:
        return '<td class="ck-cost ck-cost-na">—</td>'
    if v["state"] == "diverged":
        return '<td class="ck-cost ck-cost-na">diverges</td>'
    if v["state"] == "not_reached":
        return (
            '<td class="ck-cost ck-cost-na">not reached'
            f'<span class="ck-part">best {v["best"]:.0%}</span></td>'
        )
    t = v["seconds"]
    grids = " and ".join(f"{g}²" for g in v["grids"])
    tip = f"Wall time per simulated time unit to reach {target:.0%} error, "
    tip += (
        f"interpolated between {grids}"
        if v["state"] == "reached"
        else f"at most that of {grids}, the coarsest grid that ran"
    )
    text = ("≤ " if v["state"] == "coarsest" else "") + _seconds(t)
    rel = f'<span class="ck-part">{_ratio_text(t / best)}</span>' if best else ""
    return f'<td class="ck-cost" data-sort="{t}" title="{_esc(tip)}">{text}{rel}</td>'


def _seconds(t: float) -> str:
    return f"{t * 1000:.3g} ms" if t < 1 else f"{t:.3g} s"


def _cost_td(v: dict | None) -> str:
    if not v:
        return '<td class="ck-cost ck-cost-na">—</td>'
    r = v["ratio"]
    if r is None:
        return (
            f'<td class="ck-cost ck-cost-na" title="Left out: completed sizes only up '
            f'to N={int(v["reach"])}">—<span class="ck-part">stops at N='
            f"{int(v['reach'])}</span></td>"
        )
    rel = _ratio_text(r)
    tip = f"{v['seconds']:.3g} s"
    if math.isfinite(v["std"]) and v["n_trials"]:
        tip += f" ± {v['std']:.2g} s over {v['n_trials']} trials"
    tip += f" at N={int(v['n'])}"
    if math.isfinite(v["first"]):
        tip += f"\nFirst call {v['first']:.3g} s"
    if v["within_noise"]:
        tip += "\nWithin the spread of the fastest solver's trials"
    return (
        f'<td class="ck-cost" data-sort="{v["seconds"]}" title="{_esc(tip)}">'
        f'{_seconds(v["seconds"])}<span class="ck-part">{rel}</span></td>'
    )


def _memory_td(v: dict | None, gpu: bool) -> str:
    mib = (v or {}).get("vram" if gpu else "ram")
    if not _finite(mib):
        return '<td class="ck-cost ck-cost-na">—</td>'
    txt = f"{mib / 1024:.3g} GiB" if mib >= 1024 else f"{mib:.3g} MiB"
    tip = f"{'Device' if gpu else 'Container'} memory at N={int(v['n'])}"
    return f'<td class="ck-cost" data-sort="{mib}" title="{_esc(tip)}">{txt}</td>'


def effort_pill(level: str, detail: str = "") -> str:
    """An effort level as a pill shaded by level (L0 grey to L3 solid)."""
    cls = f"eff eff-{level}" if level in views.iv.LEVELS else "eff"
    title = f' title="{_esc(detail)}"' if detail else ""
    return f'<span class="{cls}"{title}>{_esc(level)}</span>'


def _effort_td(parts: tuple[str, str]) -> str:
    """The effort cell, sorted by level, with the required interventions on
    hover."""
    level, detail = parts
    key = level.removeprefix("L")
    sort = f' data-sort="{key}"' if key.isdigit() else ""
    return f'<td class="ck-effort"{sort}>{effort_pill(level, detail)}</td>'


# The pass whose peak memory the check tables show: the VJP, which holds the
# forward pass's state for the backward pass, else the forward pass.
_MEMORY_PASSES = ("VJP cost", "Forward cost")


def render_setup(setup: dict, *, link: str = "", heading: bool = True) -> str:
    """One setup: check table grouped by device class, plus the notes."""
    suites = setup["suites"]
    cost_cols = [c for c, _ in COST_PASSES] if setup["has_cost"] else []
    record_cols = [views.EFFORT_LABEL] if setup["has_effort"] else []
    deviations = f"deviations-{setup['key']}"
    plateaus = [r["plateau"] for r in setup["rows"] if r["plateau"]]
    # Seconds where any solver has them, else iterations (see :func:`plateau`).
    timed = any(p["seconds"] is not None for p in plateaus)
    plateau_col = (
        ["Time to plateau" if timed else "Iterations to plateau"] if plateaus else []
    )
    target = setup.get("accuracy_target")
    accuracy_col = [f"Cost to {target:.0%} error"] if target else []
    ncols = (
        1
        + len(suites)
        + len(cost_cols)
        + len(plateau_col)
        + len(accuracy_col)
        + bool(cost_cols)
        + len(record_cols)
    )
    head = (
        '<th scope="col" class="ck-solver">Solver</th>'
        + "".join(f'<th scope="col">{SUITE_LABELS[s]}</th>' for s in suites)
        + "".join(
            f'<th scope="col">{c}</th>' for c in cost_cols + accuracy_col + plateau_col
        )
        + ('<th scope="col">Peak memory</th>' if cost_cols else "")
        + "".join(f'<th scope="col">{c}</th>' for c in record_cols)
    )
    bodies = []
    for gpu, title in ((True, "GPU solvers"), (False, "CPU solvers")):
        rows = [r for r in setup["rows"] if r["gpu"] is gpu]
        if not rows:
            continue
        n = setup["cost_n"].get(gpu)
        device = setup["gpu_model"] if gpu else ""
        best = min(
            (
                r["plateau"]["seconds"]
                for r in rows
                if r["plateau"] and r["plateau"]["seconds"] is not None
            ),
            default=None,
        )
        cheapest = min(
            (
                r["accuracy_cost"]["seconds"]
                for r in rows
                if r["accuracy_cost"] and "seconds" in r["accuracy_cost"]
            ),
            default=None,
        )
        parts = []
        if cost_cols:
            parts.append(
                "Forward and VJP" + (f" at N={int(n)}" if n is not None else "")
            )
        if cost_cols or best is not None or cheapest is not None:
            parts.append(f"multiples of the fastest {'GPU' if gpu else 'CPU'} solver")
        if device and parts:
            parts.append(f"measured on {device}")
        note = ", ".join(parts)
        note = note[:1].upper() + note[1:]
        trs = [
            (
                f'<tr class="ck-group"><th colspan="{ncols}" scope="colgroup">'
                f"{title}<span>{_esc(note)}</span></th></tr>"
            )
        ]
        for r in rows:
            href = f"../solvers.qmd#{r['anchor']}" if r["anchor"] else "../solvers.qmd"
            memory = next(
                (
                    r["cost"][c]
                    for c in _MEMORY_PASSES
                    if r["cost"].get(c, {}).get("ratio") is not None
                ),
                None,
            )
            # The checks of a solver that runs a substitute for part of the
            # setup hold for that modified setup, which the notes describe.
            tag = (
                f'<a class="ck-dev-tag" href="#{deviations}" title="'
                + _esc("; ".join(t for t, _ in r["deviations"]))
                + '">modified setup</a>'
                if r["deviations"]
                else ""
            )
            trs.append(
                f'<tr><th scope="row" class="ck-solver"><a href="{href}">{_esc(r["name"])}</a>{tag}</th>'
                + "".join(_check_td(r["checks"][s]) for s in suites)
                + "".join(_cost_td(r["cost"].get(c)) for c in cost_cols)
                + (
                    _accuracy_cost_td(r["accuracy_cost"], target, cheapest)
                    if accuracy_col
                    else ""
                )
                + (_plateau_td(r["plateau"], timed, best) if plateau_col else "")
                + (_memory_td(memory, gpu) if cost_cols else "")
                + (_effort_td(r["effort"]) if record_cols else "")
                + "</tr>"
            )
        bodies.append("<tbody>" + "".join(trs) + "</tbody>")

    word = {"anomaly": "flagged", "failed": "failed", "notrun": "not run"}
    flag_items = []
    for r in setup["rows"]:
        for lab, kind, why in r["flags"]:
            cls = kind if kind in word else "blocked"
            flag_items.append(
                f"<li><strong>{_esc(r['name'])}</strong> on <code>{_esc(lab)}</code>: "
                f'<span class="fl-{cls}">{_esc(word.get(kind, kind.lower()))}</span>'
                + (f". {_esc(why)}" if why else "")
                + "</li>"
            )
    limit_items = [
        f"<li><strong>{_esc(r['name'])}</strong> on <code>{_esc(lab)}</code>, "
        f"from {_esc(frontier)}</li>"
        for r in setup["rows"]
        for lab, frontier in r["limits"]
    ]
    unsup = []
    for r in setup["rows"]:
        whys = list(dict.fromkeys(w for _, w in r["unsupported"] if w))
        if r["unsupported"]:
            n = len(r["unsupported"])
            unsup.append(
                f"<li><strong>{_esc(r['name'])}</strong>, {n} experiment"
                + ("s" if n != 1 else "")
                + (f". {_esc('; '.join(whys))}" if whys else "")
                + "</li>"
            )
    dev_items = [
        f"<li><strong>{_esc(r['name'])}</strong>: {_esc(what)}. "
        f"<em>Effect:</em> {_esc(effect)}</li>"
        for r in setup["rows"]
        for what, effect in r["deviations"]
    ]
    # Solvers of this table that do not run the domain's other setups.
    by_setup: dict[str, tuple[dict, list[str]]] = {}
    for name, missing in sorted((setup.get("elsewhere") or {}).items()):
        for o in missing:
            by_setup.setdefault(o["key"], (o, []))[1].append(name)
    elsewhere = (
        '<p class="ck-elsewhere">Solvers differ in how general they are. '
        + " ".join(
            f"{_esc(_join(names))} {'does' if len(names) == 1 else 'do'} not run "
            f'<a href="{_esc(o["key"])}.qmd">{_esc(o["setup"])}</a>.'
            for o, names in by_setup.values()
        )
        + "</p>"
        if by_setup
        else ""
    )
    details = elsewhere + (
        f'<details class="ck-flags" id="{deviations}" open><summary>Modified '
        f"setups ({len(dev_items)})</summary><p>These solvers run a substitute "
        "for part of the setup, so their checks hold for the modified setup.</p>"
        f"<ul>{''.join(dev_items)}</ul></details>"
        if dev_items
        else ""
    )
    if flag_items:
        details += (
            f'<details class="ck-flags"><summary>Flags and exclusions ({len(flag_items)})</summary>'
            f"<ul>{''.join(flag_items)}</ul></details>"
        )
    if limit_items:
        details += (
            f'<details class="ck-flags"><summary>Resource limits ({len(limit_items)})'
            "</summary><p>Sweeps a solver completed only up to some size or rollout "
            "length. It ran out of memory or time or, in sweeps that look for the "
            "limit, failed past the last value it completed. Unless that happened "
            "at the first value, the experiment counts as ok, and the limit shows "
            "how far the solver goes on this hardware.</p>"
            f"<ul>{''.join(limit_items)}</ul></details>"
        )
    if unsup:
        n = f"{len(unsup)} solver" + ("s" if len(unsup) != 1 else "")
        details += (
            f'<details class="ck-flags"><summary>Not supported ({n})</summary>'
            f"<ul>{''.join(unsup)}</ul></details>"
        )

    task = (
        f'<span class="ck-task">Optimization task: {_esc(setup["task"])}</span>'
        if setup["task"]
        else ""
    )
    header = (
        f'<header class="ck-head"><div class="ck-title">'
        f'<span class="domain-id d-{_esc(setup["id"])}">{_esc(setup["id"])}</span>'
        f'<div><h3>{_esc(setup["setup"])}</h3><span class="ck-domain">{_esc(setup["label"])}</span></div>'
        f'</div><div class="ck-meta">{task}'
        + (f'<a href="{link}">Results page</a>' if link else "")
        + "</div></header>"
        if heading
        else ""
    )
    return (
        f'<section class="checkboard" id="setup-{_esc(setup["key"])}">'
        + header
        + '<div class="table-scroll"><table class="ck-table sortable" data-quarto-disable-processing="true"><thead><tr>'
        + head
        + "</tr></thead>"
        + "".join(bodies)
        + "</table></div>"
        + (f'<div class="ck-foot">{details}</div>' if details else "")
        + "</section>"
    )


def render_overview(data: dict, pages: dict[str, str]) -> str:
    """All setups, for the Results overview page. ``pages`` maps setup keys to
    their results pages."""
    parts = ['```{=html}\n<div class="checkboards">']
    for setup in data["setups"]:
        parts.append(render_setup(setup, link=pages.get(setup["key"], "")))
    parts.append("</div>\n```\n")
    return "".join(parts)


def render_setup_page(data: dict, key: str) -> str:
    """The check table of one setup, for its results page."""
    setup = next((s for s in data["setups"] if s["key"] == key), None)
    if setup is None:
        return ""
    return (
        '```{=html}\n<div class="checkboards">'
        + render_setup(setup, heading=False)
        + "</div>\n```\n"
    )


def per_solver_summary(data: dict) -> dict[str, list[dict]]:
    """Map Solver Reference anchor → one entry per setup the solver runs."""
    out: dict[str, list[dict]] = {}
    for setup in data["setups"]:
        for r in setup["rows"]:
            if not r["anchor"]:
                continue
            states = {s: r["checks"][s]["state"] for s in setup["suites"]}
            supported = [v for v in states.values() if v != "na"]
            worst = max(supported, key=STATES.index) if supported else "na"
            out.setdefault(r["anchor"], []).append(
                {
                    "id": setup["id"],
                    "key": setup["key"],
                    "short": setup["short"],
                    "setup": setup["setup"],
                    "states": states,
                    "worst": worst,
                    "n_pass": sum(v == "pass" for v in supported),
                    "n_supported": len(supported),
                }
            )
    return out


def render_gates() -> str:
    """Markdown table of the experiments behind each check, per setup."""
    lines = [
        "| Setup | Forward | Gradient | Optimization |",
        "|---|---|---|---|",
    ]
    for setup in setups():
        cells = [
            ", ".join(f"`{_short(g)}`" for g in setup["gates"].get(s, [])) or "—"
            for s in SUITES
        ]
        lines.append(f"| {setup['setup']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
