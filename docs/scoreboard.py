# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-setup solver check tables for the Results pages.

The benchmark does not rank solvers. For each problem *setup* (one problem
statement, e.g. periodic Taylor–Green flow or flow past a cylinder) it reports:

* **Checks** — forward, gradient and optimization, each one of pass,
  borderline, fail, not supported or not run. A check
  combines the status of the setup's gate experiments for that suite, using the
  pass criteria (``status_check``) each problem config declares for them; see
  :func:`_check`. Diagnostic experiments (Jacobian spectra, horizon sweeps,
  cost sweeps) are not gates, but count towards reliability.
* **Cost** — forward and VJP wall-clock time as a multiple of the fastest
  solver *on the same device class* (GPU or CPU), at the largest problem size
  every solver of that class completed.
* **Reliability** — counts of ok, flagged (anomaly), failed and not-run
  experiment cells over every experiment of the setup, with the reason for
  each flag. Cells excluded because the method cannot express the experiment
  are "not supported", not failures.

Everything is computed from ``mosaic-results/snapshot.json`` (the
``mosaic status`` snapshot) and the cost ``result.json`` files.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

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

# One entry per problem setup. ``claims`` are experiment-label prefixes the
# setup owns; the setup without ``claims`` owns the rest of its problem.
# ``gates`` are the experiments whose status decides each check (an empty
# list drops the column for that setup).
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
            "gradient": ["gradient/fd_check", "gradient/param_sweep"],
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
        "claims": ["forward/cylinder", "optimization/drag_opt"],
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
        "setup": "Forced periodic box (Kolmogorov flow)",
        "task": "training a neural corrector through the solver",
        "claims": [
            "forward/kolmogorov",
            "forward/kolmogorov_laminar",
            "optimization/kolmogorov_sil",
        ],
        "gates": {
            "forward": ["forward/kolmogorov_laminar", "forward/kolmogorov"],
            "gradient": [],
            "optimization": ["optimization/kolmogorov_sil"],
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
STATES = ["pass", "borderline", "fail", "na", "notrun"]
STATE_LABELS = {
    "pass": "Pass",
    "borderline": "Borderline",
    "fail": "Fail",
    "na": "Not supported",
    "notrun": "Not run",
}
STATE_GLYPHS = {
    "pass": "✓",
    "borderline": "!",
    "fail": "✗",
    "na": "—",
    "notrun": "·",
}

# Cost passes: (column label, experiment dir under cost/).
COST_PASSES = [("Forward cost", "spatial_cost"), ("VJP cost", "vjp_cost/by_N")]

_PERMANENT = {"categorical"}


def _finite(v) -> bool:
    return isinstance(v, int | float) and math.isfinite(v)


def _matches(label: str, prefixes: list[str]) -> bool:
    return any(label == p or label.startswith((p + "/", p + "_")) for p in prefixes)


def _setup_rows(setup: dict, rows: list[dict]) -> list[dict]:
    """Snapshot rows (one per experiment) belonging to ``setup``."""
    if setup.get("claims"):
        return [r for r in rows if _matches(r["label"], setup["claims"])]
    claimed = [
        p
        for s in SETUPS
        if s["problem"] == setup["problem"] and s.get("claims")
        for p in s["claims"]
    ]
    return [
        r for r in rows if not _matches(r["label"], claimed) and r["suite"] != "ics"
    ]


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
    """ok | anomaly | failed | na | notrun for one snapshot cell."""
    st = cell.get("status")
    if st == "ok":
        return "ok"
    if st == "anomaly":
        return "anomaly"
    if st == "failed":
        return "failed"
    if st == "excluded" and cell.get("category") in _PERMANENT:
        return "na"
    return "notrun"


def _check(cells: list[tuple[str, dict]]) -> dict:
    """Combine the gate cells of one suite into a check.

    Any failed gate fails the check; otherwise any anomaly (explained or not)
    makes it borderline. A solver whose every
    gate is a permanent exclusion is "not supported"; one with no gate run is
    "not run".
    """
    kinds = [(label, _cell_kind(c), c) for label, c in cells]
    run = [k for k in kinds if k[1] not in ("na", "notrun")]
    if not run:
        if kinds and all(k == "na" for _, k, _ in kinds):
            state = "na"
        else:
            state = "notrun"
    elif any(k == "failed" for _, k, _ in run):
        state = "fail"
    elif any(k == "anomaly" for _, k, _ in run):
        state = "borderline"
    else:
        state = "pass"
    reasons = [(label, k, _reason(c)) for label, k, c in kinds if k not in ("ok",)]
    return {
        "state": state,
        "n_run": len(run),
        "n_gates": len([k for k in kinds if k[1] != "na"]),
        "reasons": reasons,
    }


# ── Cost ─────────────────────────────────────────────────────────────────────


def _times_by_n(
    results_dir: Path, problem: str, exp: str
) -> dict[str, dict[float, float]]:
    path = results_dir / problem / "cost" / exp / "result.json"
    if not path.exists():
        return {}
    try:
        results = json.loads(path.read_text(encoding="utf-8")).get("results", []) or []
    except Exception:
        return {}
    out: dict[str, dict[float, float]] = {}
    for r in results:
        m = r.get("metrics") or {}
        if m.get("status") == "failed" or not _finite(m.get("mean")):
            continue
        try:
            n = float(r.get("sweep_value"))
        except (TypeError, ValueError):
            continue
        out.setdefault(r["solver"], {})[n] = float(m["mean"])
    return out


def _relative_cost(
    times: dict[str, dict[float, float]], group: list[str]
) -> dict[str, dict]:
    """``t / t_fastest`` within ``group`` at the largest N all of them completed."""
    have = [s for s in group if times.get(s)]
    if not have:
        return {}
    common = set.intersection(*(set(times[s]) for s in have))
    if not common:
        return {}
    n = max(common)
    best = min(times[s][n] for s in have)
    if best <= 0:
        return {}
    return {
        s: {"ratio": times[s][n] / best, "seconds": times[s][n], "n": n} for s in have
    }


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
    for setup in SETUPS:
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
        if has_cost:
            for label, exp in COST_PASSES:
                times = _times_by_n(results_dir, problem, exp)
                for gpu in (True, False):
                    group = [
                        n
                        for n in names
                        if bool(meta.get(n, {}).get("gpu", True)) is gpu
                    ]
                    for s, v in _relative_cost(times, group).items():
                        cost.setdefault(s, {})[label] = v

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
                "notrun": sum(1 for _, k, _ in counted if k == "notrun"),
                "total": len(counted),
            }
            flags = [
                (lab, k, _reason(c))
                for lab, k, c in kinds
                if k in ("anomaly", "failed", "notrun")
            ]
            unsupported = [(lab, _reason(c)) for lab, k, c in kinds if k == "na"]
            out_rows.append(
                {
                    "name": m.get("display") or name,
                    "anchor": m.get("anchor", ""),
                    "gpu": bool(m.get("gpu", True)),
                    "checks": checks,
                    "cost": cost.get(name, {}),
                    "reliability": reliability,
                    "flags": flags,
                    "unsupported": unsupported,
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


def _check_td(chk: dict) -> str:
    st = chk["state"]
    tip = [STATE_LABELS[st]]
    if chk["n_gates"] and st not in ("na", "notrun"):
        tip.append(f"{chk['n_run']} of {chk['n_gates']} gate experiments run")
    tip += [f"{_short(lab)}: {why}" for lab, _, why in chk["reasons"] if why]
    partial = (
        f'<span class="ck-part">{chk["n_run"]}/{chk["n_gates"]}</span>'
        if st not in ("na", "notrun") and chk["n_run"] < chk["n_gates"]
        else ""
    )
    return (
        f'<td class="ck-cell"><span class="ck ck-{st}" title="{_esc(" · ".join(tip))}">'
        f'<span class="ck-glyph" aria-hidden="true">{STATE_GLYPHS[st]}</span>'
        f"{STATE_LABELS[st]}</span>{partial}</td>"
    )


def _cost_td(v: dict | None) -> str:
    if not v:
        return '<td class="ck-cost ck-cost-na">—</td>'
    r = v["ratio"]
    txt = "1× (fastest)" if r < 1.005 else (f"{r:.1f}×" if r < 10 else f"{r:.0f}×")
    tip = f"{v['seconds']:.3g} s at N={int(v['n'])}"
    return f'<td class="ck-cost" title="{_esc(tip)}">{txt}</td>'


def _rel_td(rel: dict) -> str:
    if not rel["total"]:
        return '<td class="ck-rel ck-cost-na">—</td>'
    parts = []
    if rel["flagged"]:
        parts.append(f'<span class="rl rl-flag">{rel["flagged"]} flagged</span>')
    if rel["failed"]:
        parts.append(f'<span class="rl rl-fail">{rel["failed"]} failed</span>')
    if rel["notrun"]:
        parts.append(f'<span class="rl rl-notrun">{rel["notrun"]} not run</span>')
    return (
        f'<td class="ck-rel"><span class="rl-ok">{rel["ok"]}/{rel["total"]} ok</span>'
        + "".join(parts)
        + "</td>"
    )


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
    ncols = 1 + len(suites) + len(cost_cols) + 1
    head = (
        '<th scope="col" class="ck-solver">Solver</th>'
        + "".join(f'<th scope="col">{SUITE_LABELS[s]}</th>' for s in suites)
        + "".join(f'<th scope="col">{c}</th>' for c in cost_cols)
        + '<th scope="col" class="ck-rel">Reliability</th>'
    )
    bodies = []
    for gpu, title in ((True, "GPU solvers"), (False, "CPU solvers")):
        rows = [r for r in setup["rows"] if r["gpu"] is gpu]
        if not rows:
            continue
        ns = [n for n in (_cost_n(setup, c, gpu) for c in cost_cols) if n]
        note = (
            f" · cost relative to the fastest {'GPU' if gpu else 'CPU'} solver"
            + (f" at {', '.join(dict.fromkeys(ns))}" if ns else "")
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
            trs.append(
                f'<tr><th scope="row" class="ck-solver"><a href="{href}">{_esc(r["name"])}</a></th>'
                + "".join(_check_td(r["checks"][s]) for s in suites)
                + "".join(_cost_td(r["cost"].get(c)) for c in cost_cols)
                + _rel_td(r["reliability"])
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
            word = {"anomaly": "flagged", "failed": "failed", "notrun": "not run"}[kind]
            flag_items.append(
                f"<li><strong>{_esc(r['name'])}</strong> · <code>{_esc(lab)}</code> · "
                f'<span class="fl-{kind}">{word}</span>'
                + (f" — {_esc(why)}" if why else "")
                + "</li>"
            )
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
            f'<details class="ck-flags"><summary>Why cells are flagged ({len(flag_items)})</summary>'
            f"<ul>{''.join(flag_items)}</ul></details>"
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
    for setup in SETUPS:
        pid = PROBLEM_IDS.get(setup["problem"], setup["problem"])
        cells = [
            ", ".join(f"`{_short(g)}`" for g in setup["gates"].get(s, [])) or "—"
            for s in SUITES
        ]
        lines.append(f"| {pid} · {setup['setup']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
