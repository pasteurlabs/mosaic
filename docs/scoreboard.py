# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-domain solver scoreboard for the Results overview page.

Each solver is scored per problem on five axes, each in [0, 1]:

* **Forward** — mean relative error vs. the reference, the same metric the
  per-domain forward ranking uses.
* **Gradient** — best-ε finite-difference error of the gradient (fd_check).
* **Cost** — forward and VJP wall-clock time compared at every problem size N
  separately, so a solver that also completes the larger sizes is not
  penalised for being timed there. A size the solver did not complete scores 0.
* **Optimization** — improvement achieved by the optimizer (final / initial
  objective, compared on a log scale against the best solver). Domains whose
  objective has no recorded initial value compare final objectives directly.
* **Reliability** — the campaign-health weights from ``mosaic status``
  (ok = 1, anomaly = 0.53, not run / work-to-do exclusion = 0.33, failed = 0;
  permanent exclusions are left out) averaged over every experiment cell.

Forward and gradient errors are relative errors, so they are scored on an
absolute scale: digits of accuracy out of six, ``−log10(err) / 6`` clipped to
[0, 1] (an error of 1e-6 or below scores 1, an error of 100 % scores 0). Cost
has no absolute meaning and is scored against the fastest solver,
``1 − log10(t / t_best) / 3``: matching it scores 1, being three decades slower
scores 0. An axis on which a solver has no result — typically gradient
and optimization for forward-only solvers — scores 0, since gradient quality
is a first-class criterion of the benchmark. The problem score is the
unweighted mean of the five axes. Solvers are ranked within each physics
domain (heat transfer, structural mechanics, Navier–Stokes 2D and 3D) on
their score there. A domain can group several problems, in which case its
score is the mean of the solver's problem scores within it. Everything is computed from the same result.json
files and status snapshot as the per-domain pages.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

AXES = ["forward", "gradient", "cost", "optimization", "reliability"]
AXIS_LABELS = {
    "overall": "Overall",
    "forward": "Forward",
    "gradient": "Gradient",
    "cost": "Cost",
    "optimization": "Optimization",
    "reliability": "Reliability",
}

# Short IDs match the "Benchmark domains" table on the Overview page.
PROBLEM_ORDER = ["thermal-mesh", "structural-mesh", "ns-grid", "ns-3d-grid"]
# Physics domains the scoreboard is organised by; each groups one or more
# benchmark problems and is ranked on its own.
DOMAINS = [
    {"key": "heat", "label": "Heat transfer", "problems": ["thermal-mesh"]},
    {
        "key": "structural",
        "label": "Structural mechanics",
        "problems": ["structural-mesh"],
    },
    {"key": "ns-2d", "label": "Navier–Stokes (2D)", "problems": ["ns-grid"]},
    {"key": "ns-3d", "label": "Navier–Stokes (3D)", "problems": ["ns-3d-grid"]},
]
PROBLEM_IDS = {
    "thermal-mesh": "H",
    "structural-mesh": "S",
    "ns-grid": "F2",
    "ns-3d-grid": "F3",
}

# Decades of slack before a relative (vs. best solver) axis bottoms out at 0.
_DECADES = {"cost": 3.0, "optimization": 2.0}
# Digits of accuracy that count as perfect on the absolute error scale.
_DIGITS = 6.0

# (final metric, initial metric) pairs, first match wins.
_OPT_PAIRS = (
    ("final_drag", "initial_drag"),
    ("final_ic_error", "ic_error_init"),
    ("final_error", "initial_error"),
    ("final_compliance", None),
)

_FALLBACK_WEIGHTS = {"ok": 1.0, "anomaly": 0.53, "not_run": 0.33, "excluded": 0.33}
_FALLBACK_PERMANENT = {"categorical"}


def _finite(v) -> bool:
    return isinstance(v, int | float) and math.isfinite(v)


def _log_score(x: float, best: float, decades: float) -> float:
    if not (_finite(x) and _finite(best)) or x <= 0 or best <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - math.log10(x / best) / decades))


def _median(xs) -> float:
    vals = sorted(v for v in xs if _finite(v))
    if not vals:
        return float("nan")
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def _load(results_dir: Path, problem: str, suite: str, exp: str) -> list[dict]:
    path = results_dir / problem / suite / exp / "result.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("results", []) or []
    except Exception:
        return []


# ── Raw per-axis metrics ─────────────────────────────────────────────────────


def _forward_errors(results_dir: Path, problem: str) -> dict[str, float]:
    for exp in ("tgv_nu_sweep", "agreement", "baseline", "source_baseline"):
        by_solver: dict[str, list[float]] = {}
        for r in _load(results_dir, problem, "forward", exp):
            m = r.get("metrics") or {}
            if "error" in m and m.get("valid", True) and _finite(m["error"]):
                by_solver.setdefault(r["solver"], []).append(m["error"])
        if by_solver:
            return {s: sum(v) / len(v) for s, v in by_solver.items()}
    return {}


def _gradient_errors(results_dir: Path, problem: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in _load(results_dir, problem, "gradient", "fd_check"):
        sweep = (r.get("metrics") or {}).get("eps_sweep") or {}
        best = float("inf")
        for entry in sweep.values():
            re = entry.get("rel_error")
            med = _median(re) if isinstance(re, list) else _median([re])
            if _finite(med) and med < best:
                best = med
        if math.isfinite(best):
            out[r["solver"]] = best
    return out


def _times_by_n(results: list[dict]) -> dict[str, dict[float, float]]:
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


def _cost_scores(results_dir: Path, problem: str, solvers: list[str]) -> dict:
    """Per-solver cost score: mean over passes (forward, VJP) of the mean over N."""
    passes = []
    for exp in ("spatial_cost", "vjp_cost/by_N"):
        t = _times_by_n(_load(results_dir, problem, "cost", exp))
        if t:
            passes.append(t)
    scores: dict[str, float] = {}
    for s in solvers:
        per_pass = []
        for t in passes:
            if s not in t:
                continue  # pass not applicable (e.g. no VJP for forward-only)
            sizes = sorted({n for d in t.values() for n in d})
            vals = []
            for n in sizes:
                best = min(d[n] for d in t.values() if n in d and d[n] > 0)
                vals.append(
                    _log_score(t[s].get(n, float("nan")), best, _DECADES["cost"])
                )
            per_pass.append(sum(vals) / len(vals) if vals else 0.0)
        if per_pass:
            scores[s] = sum(per_pass) / len(per_pass)
    return scores


def _optimization_scores(results_dir: Path, problem: str) -> dict[str, float]:
    opt_dir = results_dir / problem / "optimization"
    if not opt_dir.exists():
        return {}
    for exp in sorted(p.name for p in opt_dir.iterdir() if p.is_dir()):
        ratios: dict[str, list[float]] = {}
        finals: dict[str, list[float]] = {}
        for r in _load(results_dir, problem, "optimization", exp):
            m = r.get("metrics") or {}
            pair = next((p for p in _OPT_PAIRS if p[0] in m), None)
            if pair is None or not _finite(m.get(pair[0])):
                continue
            final, init = m[pair[0]], m.get(pair[1]) if pair[1] else None
            if _finite(init) and init != 0:
                ratios.setdefault(r["solver"], []).append(final / init)
            else:
                finals.setdefault(r["solver"], []).append(final)
        if ratios:
            # Score = log(r) / log(r_best): the fraction of the best solver's
            # improvement (in decades) this solver achieved. r >= 1 means the
            # optimizer made no progress.
            logs = {
                s: sum(math.log10(v) for v in vs if v > 0) / len(vs)
                for s, vs in ratios.items()
                if all(v > 0 for v in vs)
            }
            best = min(logs.values(), default=0.0)
            if best >= 0:
                return {s: 0.0 for s in ratios}
            return {
                s: max(0.0, min(1.0, logs[s] / best)) if s in logs else 0.0
                for s in ratios
            }
        if finals:
            means = {s: sum(v) / len(v) for s, v in finals.items()}
            best = min((v for v in means.values() if v > 0), default=float("nan"))
            return {
                s: _log_score(v, best, _DECADES["optimization"])
                for s, v in means.items()
            }
    return {}


def _reliability(snapshot_problem: dict) -> dict[str, float]:
    try:
        from mosaic.benchmarks.core.status import EXCL_PERMANENT_VALUES, SCORE_WEIGHTS

        weights = {
            "ok": SCORE_WEIGHTS["ok"],
            "anomaly": SCORE_WEIGHTS["anom"],
            "failed": SCORE_WEIGHTS["fail"],
            "not_run": SCORE_WEIGHTS["missing"],
            "excluded": SCORE_WEIGHTS["excl"],
        }
        permanent = set(EXCL_PERMANENT_VALUES)
    except Exception:
        weights, permanent = dict(_FALLBACK_WEIGHTS), set(_FALLBACK_PERMANENT)
    tot: dict[str, list[float]] = {}
    for row in snapshot_problem.get("rows", []):
        for s, cell in (row.get("cells") or {}).items():
            st = cell.get("status")
            if st == "excluded" and cell.get("category") in permanent:
                continue
            tot.setdefault(s, []).append(weights.get(st, 0.0))
    return {s: sum(v) / len(v) for s, v in tot.items() if v}


def _status_counts(snapshot_problem: dict) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in snapshot_problem.get("rows", []):
        for s, cell in (row.get("cells") or {}).items():
            c = out.setdefault(s, {})
            c[cell.get("status", "?")] = c.get(cell.get("status", "?"), 0) + 1
    return out


# ── Assembly ─────────────────────────────────────────────────────────────────


def _assign_ranks(rows: list[dict], key) -> None:
    """Competition ranking on the displayed (rounded) score: ties share a rank."""
    prev, rank = None, 0
    for i, e in enumerate(rows):
        shown = round(100 * key(e))
        if shown != prev:
            rank, prev = i + 1, shown
        e["rank"] = rank


def _digits_scores(raw: dict[str, float]) -> dict[str, float]:
    """Relative error → digits of accuracy out of ``_DIGITS``, in [0, 1]."""
    floor = 10.0**-_DIGITS
    return {s: _log_score(max(v, floor), floor, _DIGITS) for s, v in raw.items()}


def compute(results_dir: Path, solver_meta: dict[str, dict[str, dict]]) -> dict | None:
    """Return the scoreboard data structure, or None without a snapshot.

    ``solver_meta[problem][solver_name]`` supplies ``display`` (card name) and
    ``anchor`` (Solver Reference id) for each solver.
    """
    snap_path = results_dir / "snapshot.json"
    if not snap_path.exists():
        return None
    snapshot = json.loads(snap_path.read_text(encoding="utf-8")).get("problems", {})
    problems = [p for p in PROBLEM_ORDER if p in snapshot] + sorted(
        p for p in snapshot if p not in PROBLEM_ORDER
    )

    solvers: dict[str, dict] = {}
    for problem in problems:
        snap = snapshot[problem]
        names = list(snap.get("solvers", []))
        axis_scores = {
            "forward": _digits_scores(_forward_errors(results_dir, problem)),
            "gradient": _digits_scores(_gradient_errors(results_dir, problem)),
            "cost": _cost_scores(results_dir, problem, names),
            "optimization": _optimization_scores(results_dir, problem),
            "reliability": _reliability(snap),
        }
        counts = _status_counts(snap)
        meta = solver_meta.get(problem, {})
        for name in names:
            m = meta.get(name, {})
            display = m.get("display") or name
            entry = solvers.setdefault(
                display,
                {"name": display, "anchor": m.get("anchor", ""), "problems": {}},
            )
            if not entry["anchor"]:
                entry["anchor"] = m.get("anchor", "")
            axes = {a: axis_scores[a].get(name) for a in AXES}
            score = sum(v or 0.0 for v in axes.values()) / len(AXES)
            entry["problems"][problem] = {
                "score": score,
                "axes": axes,
                "counts": counts.get(name, {}),
                "anchor": m.get("anchor", ""),
            }

    domains = []
    for dom in DOMAINS:
        probs = [p for p in dom["problems"] if p in problems]
        if not probs:
            continue
        rows = []
        for entry in solvers.values():
            ps = {p: entry["problems"][p] for p in probs if p in entry["problems"]}
            if not ps:
                continue
            rows.append(
                {
                    "name": entry["name"],
                    "anchor": next(iter(ps.values()))["anchor"] or entry["anchor"],
                    # Domain score: mean over the domain's problems the solver runs.
                    "score": sum(p["score"] for p in ps.values()) / len(ps),
                    # Missing results count as 0 but display as "—" when the
                    # solver has no result for that axis anywhere in the domain.
                    "axes": {
                        a: (
                            None
                            if all(p["axes"][a] is None for p in ps.values())
                            else sum((p["axes"][a] or 0.0) for p in ps.values())
                            / len(ps)
                        )
                        for a in AXES
                    },
                    "problems": {p: d["score"] for p, d in ps.items()},
                }
            )
        rows.sort(key=lambda e: -e["score"])
        _assign_ranks(rows, lambda e: e["score"])
        domains.append({**dom, "problems": probs, "rows": rows})
    return {
        "problems": [{"key": p, "id": PROBLEM_IDS.get(p, p)} for p in problems],
        "axes": AXES,
        "solvers": list(solvers.values()),
        "domains": domains,
    }


# ── HTML rendering ───────────────────────────────────────────────────────────


def _overall_td(value: float, attrs: str = "") -> str:
    """Overall-score cell: a small bar plus the number."""
    return (
        f'<td class="sb-overall"{attrs}><div class="sb-bar">'
        f'<span style="width:{100 * value:.1f}%"></span></div>'
        f'<span class="sb-val">{_pct(value)}</span></td>'
    )


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{round(100 * v)}"


def _level(v: float | None) -> int:
    if v is None:
        return 0
    return max(1, min(5, 1 + int(v * 5 - 1e-9))) if v > 0 else 1


def _score_td(v: float | None, title: str = "") -> str:
    t = f' title="{html.escape(title)}"' if title else ""
    return f'<td class="sb-cell lvl-{_level(v)}"{t}><span class="sb-val">{_pct(v)}</span></td>'


def render_overview(data: dict, labels: dict[str, str], pages: dict[str, str]) -> str:
    """Raw-HTML scoreboard: one ranked table per physics domain.

    Columns are the domain's problems (when it has more than one), the five
    axes averaged over those problems, and the domain score. Headers are
    click-to-sort via docs/sortable-tables.html.
    """
    out = ['```{=html}\n<div class="scoreboard-domains">']
    for dom in data["domains"]:
        probs = dom["problems"]
        multi = len(probs) > 1
        ids = [PROBLEM_IDS.get(p, p) for p in probs]
        links = " · ".join(
            f'<a href="{pages.get(p, "#")}">{html.escape(labels.get(p, p))} results</a>'
            for p in probs
        )
        prob_head = (
            "".join(
                f'<th scope="col" class="sb-prob" title="{html.escape(labels.get(p, p))}">'
                f"{html.escape(i)}</th>"
                for p, i in zip(probs, ids, strict=True)
            )
            if multi
            else ""
        )
        axis_head = "".join(f'<th scope="col">{AXIS_LABELS[a]}</th>' for a in AXES)
        body = []
        for r in dom["rows"]:
            anchor = f"solvers.qmd#{r['anchor']}" if r["anchor"] else "solvers.qmd"
            missing = [
                i for p, i in zip(probs, ids, strict=True) if p not in r["problems"]
            ]
            note = (
                f'<span class="sb-cover">not run on {", ".join(missing)}</span>'
                if multi and missing
                else ""
            )
            prob_cells = (
                "".join(
                    _score_td(r["problems"].get(p), labels.get(p, p)) for p in probs
                )
                if multi
                else ""
            )
            body.append(
                f'<tr><td class="sb-rank"><span class="rank-pill r{min(r["rank"], 4)}">'
                f"{r['rank']}</span></td>"
                f'<th scope="row" class="sb-solver"><a href="{anchor}">'
                f"{html.escape(r['name'])}</a>{note}</th>"
                + prob_cells
                + "".join(_score_td(r["axes"][a]) for a in AXES)
                + _overall_td(r["score"])
                + "</tr>"
            )
        pid_tiles = "".join(
            f'<span class="domain-id d-{html.escape(i)}">{html.escape(i)}</span>'
            for i in ids
        )
        out.append(
            f'<section class="scoreboard sb-domain" id="scores-{dom["key"]}">'
            f'<header class="sb-domain-head"><div class="sb-domain-title">{pid_tiles}'
            f"<h3>{html.escape(dom['label'])}</h3>"
            f'<span class="sb-domain-meta">{len(dom["rows"])} solvers</span></div>'
            f'<div class="sb-domain-links">{links}</div></header>'
            '<div class="table-scroll sortable-table"><table class="sb-table"><thead><tr>'
            '<th class="sb-rank" scope="col">Rank</th>'
            '<th scope="col" class="sb-solver">Solver</th>'
            f"{prob_head}{axis_head}"
            '<th scope="col" class="sb-overall">Domain score</th></tr></thead><tbody>'
            + "".join(body)
            + "</tbody></table></div></section>"
        )
    out.append("</div>\n```\n")
    return "".join(out)


def render_domain(data: dict, problem: str) -> str:
    """Raw-HTML per-domain score table (one row per solver, axes as columns)."""
    rows = [
        (s, s["problems"][problem]) for s in data["solvers"] if problem in s["problems"]
    ]
    if not rows:
        return ""
    rows.sort(key=lambda t: -t[1]["score"])
    ranks = [{"score": p["score"]} for _, p in rows]
    _assign_ranks(ranks, lambda e: e["score"])
    head = "".join(f'<th scope="col">{AXIS_LABELS[a]}</th>' for a in AXES)
    body = []
    for (s, p), r in zip(rows, ranks, strict=True):
        anchor = f"solvers.qmd#{p['anchor'] or s['anchor']}"
        cells = "".join(
            f'<td class="sb-cell lvl-{_level(p["axes"][a])}"><span class="sb-val">{_pct(p["axes"][a])}</span></td>'
            for a in AXES
        )
        body.append(
            f'<tr><td class="sb-rank"><span class="rank-pill r{min(r["rank"], 4)}">{r["rank"]}</span></td>'
            f'<th scope="row" class="sb-solver"><a href="{anchor}">{html.escape(s["name"])}</a></th>'
            f"{cells}{_overall_td(p['score'])}</tr>"
        )
    return (
        '```{=html}\n<div class="scoreboard scoreboard-domain"><div class="table-scroll">'
        '<table class="sb-table"><thead><tr><th class="sb-rank" scope="col">Rank</th>'
        f'<th scope="col" class="sb-solver">Solver</th>{head}<th scope="col" class="sb-overall">Score</th>'
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div></div>\n```\n"
    )


def per_solver_summary(data: dict) -> dict[str, dict]:
    """Map Solver Reference anchor → {problem id: {score, rank, of, domain}}.

    One card can cover several problems (the Navier–Stokes containers run
    both 2D and 3D), so ranks are reported per problem.
    """
    out: dict[str, dict] = {}
    dom_of = {p: d for d in data["domains"] for p in d["problems"]}
    for s in data["solvers"]:
        for prob, p in s["problems"].items():
            if not p["anchor"]:
                continue
            dom = dom_of.get(prob)
            row = (
                next((r for r in dom["rows"] if r["name"] == s["name"]), None)
                if dom
                else None
            )
            out.setdefault(p["anchor"], {})[PROBLEM_IDS.get(prob, prob)] = {
                "score": p["score"],
                "rank": row["rank"] if row else None,
                "of": len(dom["rows"]) if dom else 0,
                "domain": dom["label"] if dom else "",
            }
    return out
