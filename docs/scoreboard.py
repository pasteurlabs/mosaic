# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-setup solver check tables for the Results pages.

The benchmark does not rank solvers. For each problem *setup* (one problem
statement, e.g. periodic Taylor–Green flow or flow past a cylinder; the setups
and the experiments each one owns are defined in
``mosaic/benchmarks/core/interventions.py``) it reports:

* **Checks** — forward, gradient and optimization, each one of pass,
  borderline, fail, blocked, not supported or not run. A check
  combines the status of the setup's gate experiments for that suite, using the
  pass criteria (``status_check``) each problem config declares for them; see
  :func:`_check`. Diagnostic experiments (Jacobian spectra, horizon sweeps,
  cost sweeps) are not gates, but count towards reliability.
* **Cost** — forward and VJP wall-clock time as a multiple of the fastest
  solver *on the same device class* (GPU or CPU), at the largest problem size
  every solver of that class completed. A gap to the fastest solver that is
  within the spread of the timed trials is marked.
* **Reliability** — counts of ok, flagged (anomaly), failed, blocked and
  not-run experiment cells over every experiment of the setup, with the reason
  for each, and the experiments where a solver reached a resource limit
  partway through a sweep. Cells excluded because the method cannot express
  the experiment are "not supported", not failures; cells excluded for a
  recorded reason such as an instability or an upstream bug are "blocked".
* **Effort** and **deviations** — from each solver's intervention record.

Everything is computed from ``mosaic-results/snapshot.json`` (the
``mosaic status`` snapshot), the cost ``result.json`` files and the
intervention records.
"""

from __future__ import annotations

import html
import json
import math
import sys
from collections import Counter
from pathlib import Path


def _load_views():
    """Load docs/intervention_views.py whether this runs as a script or a module."""
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
# check (an empty list drops the column for that setup).
SETUPS = [
    {
        "key": "heat",
        "short": "slab",
        "problem": "thermal-mesh",
        "label": "Heat transfer",
        "setup": "Steady conduction in a slab",
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
        "setup": "Linear-elastic cantilever",
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
        "setup": "Periodic box (Taylor–Green, multimode)",
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
        "setup": "Channel flow past a cylinder",
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
        "setup": "Forced Kolmogorov flow",
        "task": "",
        "gates": {
            "forward": ["forward/kolmogorov_laminar", "forward/kolmogorov"],
            "gradient": ["gradient/chain_fd_check"],
            "optimization": [],
        },
    },
    {
        "key": "ns-3d",
        "short": "periodic",
        "problem": "ns-3d-grid",
        "label": "Navier–Stokes (3D)",
        "setup": "Triply periodic box (Taylor–Green)",
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
STATES = ["pass", "borderline", "fail", "blocked", "na", "notrun"]
STATE_LABELS = {
    "pass": "Pass",
    "borderline": "Borderline",
    "fail": "Fail",
    "blocked": "Blocked",
    "na": "Not supported",
    "notrun": "Not run",
}
STATE_GLYPHS = {
    "pass": "✓",
    "borderline": "!",
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


def _setups() -> list[dict]:
    """The setups of :data:`SETUPS` that the intervention module defines."""
    return [s for s in SETUPS if s["key"] in views.iv.SETUPS]


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
    makes it borderline, and the check is marked explained when every anomaly
    has a documented cause. A solver whose every gate is a permanent exclusion
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
        state = "borderline"
    else:
        state = "pass"
    reasons = [(label, k, _reason(c)) for label, k, c in kinds if k not in ("ok",)]
    anomalies = [c for _, k, c in kinds if k == "anomaly"]
    blocked_as = {_blocked_label(c) for c in blocked}
    return {
        "state": state,
        "n_run": len(run),
        "n_gates": len([k for k in kinds if k[1] != "na"]),
        "reasons": reasons,
        "explained": state == "borderline"
        and all((c.get("explanation") or "").strip() for c in anomalies),
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


def _relative_cost(
    times: dict[str, dict[float, dict]], group: list[str]
) -> dict[str, dict]:
    """``t / t_fastest`` within ``group`` at the largest N all of them completed.

    A solver is ``within_noise`` of the fastest when the gap between their
    means is at most twice the combined standard deviation of their trials.
    """
    have = [s for s in group if times.get(s)]
    if not have:
        return {}
    common = set.intersection(*(set(times[s]) for s in have))
    if not common:
        return {}
    n = max(common)
    fastest = min(have, key=lambda s: times[s][n]["mean"])
    best = times[fastest][n]
    if best["mean"] <= 0:
        return {}
    out = {}
    for s in have:
        t = times[s][n]
        spread = math.hypot(t["std"], best["std"])
        out[s] = {
            "ratio": t["mean"] / best["mean"],
            "seconds": t["mean"],
            "std": t["std"],
            "n_trials": t["n_trials"],
            "first": t["first"],
            "n": n,
            "within_noise": s != fastest
            and math.isfinite(spread)
            and t["mean"] - best["mean"] <= 2 * spread,
        }
    return out


# ── Assembly ─────────────────────────────────────────────────────────────────


def _sort_key(row: dict, suites: list[str]):
    sev = [STATES.index(row["checks"][s]["state"]) for s in suites]
    blocking = sum(1 for v in sev if v >= STATES.index("fail"))
    soft = sum(1 for v in sev if v == STATES.index("borderline"))
    fwd = (row["cost"].get("Forward cost") or {}).get("ratio", math.inf)
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

    setups = []
    for setup in _setups():
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
        # Each solver is compared within the device class its cost results
        # were measured on, which can differ from its config when the results
        # predate a move to the GPU. Without a hardware record the config
        # decides.
        configured = {n: bool(meta.get(n, {}).get("gpu", True)) for n in names}
        measured: dict[str, bool] = {}
        gpu_models: set[str] = set()
        if has_cost:
            passes = [
                (label, cost_result(results_dir, problem, exp))
                for label, exp in COST_PASSES
            ]
            for _, data in reversed(passes):  # the forward pass wins
                for n, record in hardware(data).items():
                    if n in configured:
                        measured[n] = bool(record.get("gpus"))
            for label, data in passes:
                for n, record in hardware(data).items():
                    if measured.get(n):
                        gpu_models |= _gpu_names(record)
                times = _times_by_n(data)
                for gpu in (True, False):
                    group = [n for n in names if measured.get(n, configured[n]) is gpu]
                    for s, v in _relative_cost(times, group).items():
                        cost.setdefault(s, {})[label] = v

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
            # columns already show its speed, so reliability counts it as ok.
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
            counted = [k for k in kinds if k[1] != "na"]
            reliability = {
                "ok": sum(1 for _, k, _ in counted if k == "ok"),
                "flagged": sum(1 for _, k, _ in counted if k == "anomaly"),
                "failed": sum(1 for _, k, _ in counted if k == "failed"),
                "blocked": dict(
                    Counter(_blocked_label(c) for _, k, c in counted if k == "blocked")
                ),
                "notrun": sum(1 for _, k, _ in counted if k == "notrun"),
                "limits": sum(1 for _, _, c in counted if c.get("resource_frontier")),
                "total": len(counted),
            }
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
                    "gpu": measured.get(name, configured[name]),
                    # Cost measured on a different device class than the
                    # solver's config targets, e.g. results that predate a
                    # move to the GPU.
                    "device_mismatch": name in measured
                    and measured[name] is not configured[name],
                    "checks": checks,
                    "cost": cost.get(name, {}),
                    "reliability": reliability,
                    "flags": flags,
                    "limits": limits,
                    "unsupported": unsupported,
                    "effort": views.effort_parts(records.get(name), setup["key"]),
                    "deviations": views.deviations_parts(
                        records.get(name), setup["key"]
                    ),
                }
            )
        out_rows.sort(key=lambda r: _sort_key(r, suites))
        setups.append(
            {
                **{
                    k: setup[k]
                    for k in ("key", "short", "problem", "label", "setup", "task")
                },
                "id": PROBLEM_IDS.get(problem, problem),
                "suites": suites,
                "has_cost": has_cost and bool(cost),
                "gpu_model": gpu_models.pop() if len(gpu_models) == 1 else "",
                "has_effort": bool(records),
                "rows": out_rows,
            }
        )
    return {"setups": setups}


# ── HTML rendering ───────────────────────────────────────────────────────────


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
    st = chk["state"]
    label = chk["blocked_as"] if st == "blocked" else STATE_LABELS[st]
    tip = [label]
    if chk["n_gates"] and st not in ("na", "notrun", "blocked"):
        tip.append(f"{chk['n_run']} of {chk['n_gates']} gate experiments run")
    # Every gate's headline numbers, so that e.g. the finite-difference and
    # analytic gradient checks read side by side.
    tip += [
        f"{_short(lab)}: {txt}" for lab, m in chk["metrics"] if (txt := _metric_text(m))
    ]
    tip += [f"{_short(lab)}: {why}" for lab, _, why in chk["reasons"] if why]
    partial = (
        f'<span class="ck-part">{chk["n_run"]}/{chk["n_gates"]}</span>'
        if st not in ("na", "notrun", "blocked") and chk["n_run"] < chk["n_gates"]
        else ""
    )
    explained = (
        '<sup class="ck-explained" aria-label="explained">i</sup>'
        if chk["explained"]
        else ""
    )
    return (
        f'<td class="ck-cell"><span class="ck ck-{st}" title="{_esc(" · ".join(tip))}">'
        f'<span class="ck-glyph" aria-hidden="true">{STATE_GLYPHS[st]}</span>'
        f"{_esc(label)}{explained}</span>{partial}</td>"
    )


def _cost_td(v: dict | None) -> str:
    if not v:
        return '<td class="ck-cost ck-cost-na">—</td>'
    r = v["ratio"]
    txt = "1× (fastest)" if r < 1.005 else (f"{r:.1f}×" if r < 10 else f"{r:.0f}×")
    tip = f"{v['seconds']:.3g} s"
    if math.isfinite(v["std"]) and v["n_trials"]:
        tip += f" ± {v['std']:.2g} s over {v['n_trials']} trials"
    tip += f" at N={int(v['n'])}"
    if math.isfinite(v["first"]):
        tip += f" · first call {v['first']:.3g} s"
    if v["within_noise"]:
        txt = f"≈ {txt}"
        tip += " · within the spread of the fastest solver's trials"
    return f'<td class="ck-cost" title="{_esc(tip)}">{txt}</td>'


def _rel_td(rel: dict) -> str:
    if not rel["total"]:
        return '<td class="ck-rel ck-cost-na">—</td>'
    parts = []
    if rel["flagged"]:
        parts.append(f'<span class="rl rl-flag">{rel["flagged"]} flagged</span>')
    if rel["failed"]:
        parts.append(f'<span class="rl rl-fail">{rel["failed"]} failed</span>')
    for label, n in rel["blocked"].items():
        parts.append(f'<span class="rl rl-blocked">{n} {_esc(label.lower())}</span>')
    if rel["notrun"]:
        parts.append(f'<span class="rl rl-notrun">{rel["notrun"]} not run</span>')
    if rel["limits"]:
        n = rel["limits"]
        parts.append(
            f'<span class="rl rl-limit">{n} resource limit{"s" if n != 1 else ""}</span>'
        )
    return (
        f'<td class="ck-rel"><span class="rl-ok">{rel["ok"]}/{rel["total"]} ok</span>'
        + "".join(parts)
        + "</td>"
    )


def _record_td(parts: tuple[str, str], cls: str) -> str:
    text, detail = parts
    inner = (
        f'<span class="m-tip" title="{_esc(detail)}">{_esc(text)}</span>'
        if detail
        else _esc(text)
    )
    return f'<td class="{cls}">{inner}</td>'


def _cost_n(setup: dict, label: str, gpu: bool) -> str:
    ns = {
        int(r["cost"][label]["n"])
        for r in setup["rows"]
        if r["gpu"] is gpu and label in r["cost"]
    }
    return f"N={ns.pop()}" if len(ns) == 1 else ""


def render_setup(setup: dict, *, link: str = "", heading: bool = True) -> str:
    """One setup: check table grouped by device class, plus the flag list."""
    suites = setup["suites"]
    cost_cols = [c for c, _ in COST_PASSES] if setup["has_cost"] else []
    record_cols = (
        [views.EFFORT_LABEL, views.DEVIATIONS_LABEL] if setup["has_effort"] else []
    )
    ncols = 1 + len(suites) + len(cost_cols) + 1 + len(record_cols)
    head = (
        '<th scope="col" class="ck-solver">Solver</th>'
        + "".join(f'<th scope="col">{SUITE_LABELS[s]}</th>' for s in suites)
        + "".join(f'<th scope="col">{c}</th>' for c in cost_cols)
        + '<th scope="col" class="ck-rel">Reliability</th>'
        + "".join(f'<th scope="col">{c}</th>' for c in record_cols)
    )
    bodies = []
    for gpu, title in ((True, "GPU solvers"), (False, "CPU solvers")):
        rows = [r for r in setup["rows"] if r["gpu"] is gpu]
        if not rows:
            continue
        ns = [n for n in (_cost_n(setup, c, gpu) for c in cost_cols) if n]
        device = setup["gpu_model"] if gpu else ""
        note = (
            f" · cost relative to the fastest {'GPU' if gpu else 'CPU'} solver"
            + (f" at {', '.join(dict.fromkeys(ns))}" if ns else "")
            + (f", measured on {device}" if device else "")
            if cost_cols
            else ""
        )
        trs = [
            (
                f'<tr class="ck-group"><th colspan="{ncols}" scope="colgroup">'
                f"{title}<span>{_esc(note)}</span></th></tr>"
            )
        ]
        for r in rows:
            href = f"solvers.qmd#{r['anchor']}" if r["anchor"] else "solvers.qmd"
            mismatch = (
                '<span class="ck-device" title="This solver targets the '
                f"{'CPU' if gpu else 'GPU'}, but its cost results were measured "
                f'on the {"GPU" if gpu else "CPU"}; they predate the change.">'
                f"{'GPU' if gpu else 'CPU'} run</span>"
                if r["device_mismatch"]
                else ""
            )
            trs.append(
                f'<tr><th scope="row" class="ck-solver"><a href="{href}">{_esc(r["name"])}</a>{mismatch}</th>'
                + "".join(_check_td(r["checks"][s]) for s in suites)
                + "".join(_cost_td(r["cost"].get(c)) for c in cost_cols)
                + _rel_td(r["reliability"])
                + (
                    _record_td(r["effort"], "ck-effort")
                    + _record_td(r["deviations"], "ck-dev")
                    if record_cols
                    else ""
                )
                + "</tr>"
            )
        bodies.append("<tbody>" + "".join(trs) + "</tbody>")

    passing = [
        r["name"]
        for r in setup["rows"]
        if all(r["checks"][s]["state"] == "pass" for s in suites)
    ]
    summary = (
        "Passes every check: " + ", ".join(_esc(n) for n in passing) + "."
        if passing
        else "No solver passes every check yet."
    )

    flag_items = []
    for r in setup["rows"]:
        for lab, kind, why in r["flags"]:
            word = {"anomaly": "flagged", "failed": "failed", "notrun": "not run"}
            cls = kind if kind in word else "blocked"
            flag_items.append(
                f"<li><strong>{_esc(r['name'])}</strong> · <code>{_esc(lab)}</code> · "
                f'<span class="fl-{cls}">{_esc(word.get(kind, kind.lower()))}</span>'
                + (f" — {_esc(why)}" if why else "")
                + "</li>"
            )
    limit_items = [
        f"<li><strong>{_esc(r['name'])}</strong> · <code>{_esc(lab)}</code> · "
        f"from {_esc(frontier)}</li>"
        for r in setup["rows"]
        for lab, frontier in r["limits"]
    ]
    unsup = []
    for r in setup["rows"]:
        whys = list(dict.fromkeys(w for _, w in r["unsupported"] if w))
        if r["unsupported"]:
            unsup.append(
                f"<li><strong>{_esc(r['name'])}</strong> · {len(r['unsupported'])} experiments"
                + (f" — {_esc('; '.join(whys))}" if whys else "")
                + "</li>"
            )
    details = ""
    if flag_items:
        details += (
            f'<details class="ck-flags"><summary>Flags and exclusions ({len(flag_items)})</summary>'
            f"<ul>{''.join(flag_items)}</ul></details>"
        )
    if limit_items:
        details += (
            f'<details class="ck-flags"><summary>Resource limits ({len(limit_items)})'
            "</summary><p>Sweeps a solver completed only up to some size or rollout "
            "length: it ran out of memory or time, or, in sweeps that look for the "
            "limit, failed past the last value it completed. Unless that happened "
            "at the first value, the experiment counts as ok; the limit shows how "
            "far the solver goes on this hardware.</p>"
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
        f'<section class="checkboard effort-table" id="setup-{_esc(setup["key"])}">'
        + header
        + '<div class="table-scroll"><table class="ck-table" data-quarto-disable-processing="true"><thead><tr>'
        + head
        + "</tr></thead>"
        + "".join(bodies)
        + "</table></div>"
        + f'<div class="ck-foot"><p class="ck-summary">{summary}</p>{details}</div>'
        + "</section>"
    )


def render_overview(data: dict, pages: dict[str, str]) -> str:
    """All setups, for the Results overview page."""
    parts = ['```{=html}\n<div class="checkboards">']
    for setup in data["setups"]:
        parts.append(render_setup(setup, link=pages.get(setup["problem"], "")))
    parts.append("</div>\n```\n")
    return "".join(parts)


def render_problem(data: dict, problem: str) -> str:
    """The setups of one problem, for its results page."""
    setups = [s for s in data["setups"] if s["problem"] == problem]
    if not setups:
        return ""
    many = len(setups) > 1
    return (
        '```{=html}\n<div class="checkboards">'
        + "".join(render_setup(s, heading=many) for s in setups)
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
    """Markdown table of each setup's gate experiments, for the method notes."""
    lines = [
        "| Setup | Forward | Gradient | Optimization |",
        "|---|---|---|---|",
    ]
    for setup in _setups():
        pid = PROBLEM_IDS.get(setup["problem"], setup["problem"])
        cells = [
            ", ".join(f"`{_short(g)}`" for g in setup["gates"].get(s, [])) or "—"
            for s in SUITES
        ]
        lines.append(f"| {pid} · {setup['setup']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
