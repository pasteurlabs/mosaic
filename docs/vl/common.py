# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared building blocks for the Vega-Lite result charts.

Every chart module in this package defines plots once as Altair specs over
tidy data read from ``mosaic-results/``. The specs carry no visual theme: the
docs site applies its own (light/dark) config in the browser, and
:func:`paper` applies a print config for static SVG/PDF export.

Conventions for builders (see ``BUILDERS`` in each module):

* signature ``build(ctx: Ctx) -> alt.TopLevelMixin | None``;
* return ``None`` when the data a plot needs is missing;
* reproduce the data reduction of the matching matplotlib plot exactly, and
  take solver labels, colours, dashes and markers from :func:`solver_encodings`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd

# Charts embed their data, so keep it lean.
alt.data_transformers.disable_max_rows()

# Longest side of an exported field; larger fields are subsampled.
MAX_FIELD = 64


@dataclass(frozen=True)
class Ctx:
    """Where a plot lives: ``mosaic-results/<problem>/<suite>/<experiment>/<png>``.

    ``experiment`` is ``""`` for suite-level plots (e.g. ``cost/cost.png``) and
    may contain a ``/`` for sub-experiments.
    """

    problem: str
    suite: str
    experiment: str
    png: Path

    @property
    def dir(self) -> Path:
        return self.png.parent

    @cached_property
    def config(self) -> Any:
        from mosaic.benchmarks.problems import get_config

        return get_config(self.problem)

    def result(self, sub: str | None = None) -> dict | None:
        """Parsed ``result.json`` of this experiment (or a sibling ``sub`` dir)."""
        path = (self.dir / sub if sub else self.dir) / "result.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def legacy(self, sub: str | None = None) -> dict | None:
        """``result.json`` in the legacy ``by_solver`` layout the plots use."""
        from mosaic.benchmarks.core.io import v1_to_legacy

        res = self.result(sub)
        return v1_to_legacy(res) if res is not None else None

    def npz(self, name: str) -> Any:
        path = self.dir / name
        return np.load(path, allow_pickle=False) if path.exists() else None


# ── Solver styles (the registry the matplotlib plots use) ────────────────────

# matplotlib linestyle → Vega strokeDash; marker → Vega shape (SVG path in the
# unit square for shapes Vega lacks).
_DASH = {"-": [1, 0], "--": [6, 3], "-.": [6, 3, 1, 3], ":": [1, 3]}
_SHAPE = {
    "o": "circle",
    "s": "square",
    "^": "triangle-up",
    "v": "triangle-down",
    "<": "triangle-left",
    ">": "triangle-right",
    "D": "diamond",
    "d": "diamond",
    "P": "cross",
    "+": "cross",
    "X": "M-.8,-1L0,-.2L.8,-1L1,-.8L.2,0L1,.8L.8,1L0,.2L-.8,1L-1,.8L-.2,0L-1,-.8Z",
    "h": "M0,-1L.87,-.5L.87,.5L0,1L-.87,.5L-.87,-.5Z",
    "H": "M-1,0L-.5,-.87L.5,-.87L1,0L.5,.87L-.5,.87Z",
}


def _dash(ls: Any) -> list[float]:
    if isinstance(ls, str):
        return _DASH.get(ls, [1, 0])
    try:
        return [2 * x for x in ls[1]]
    except (TypeError, IndexError):
        return [1, 0]


def solver_style(solver: str) -> tuple[str, str, list[float], str]:
    """``(label, color, strokeDash, shape)`` for a solver name/alias."""
    from mosaic.benchmarks.problems.shared.plots.style import (
        resolve_solver_alias,
        solver_props,
    )

    label, color, ls, mk = solver_props(resolve_solver_alias(solver) or solver)
    return label, color, _dash(ls), _SHAPE.get(mk, "circle")


def registry_order(solvers) -> list[str]:
    """Solvers in plot-registry order (NS_ORDER, then FEM_ORDER), others last."""
    from mosaic.benchmarks.problems.shared.plots.style import (
        FEM_ORDER,
        NS_ORDER,
        resolve_solver_alias,
    )

    rank = {a: i for i, a in enumerate(NS_ORDER + FEM_ORDER)}
    return sorted(
        set(solvers), key=lambda s: (rank.get(resolve_solver_alias(s) or s, 99), s)
    )


@dataclass
class SolverEnc:
    """Display labels plus the colour / dash / shape scales for a solver set."""

    keys: list[str]
    labels: list[str]
    color: alt.Scale
    dash: alt.Scale
    shape: alt.Scale

    def label(self, key: str) -> str:
        return dict(zip(self.keys, self.labels, strict=True)).get(key, key)

    def relabel(self, df: pd.DataFrame, col: str = "solver") -> pd.DataFrame:
        """Replace raw solver names in ``df[col]`` by display labels."""
        return df.assign(
            **{col: df[col].map(dict(zip(self.keys, self.labels, strict=True)))}
        )


def solver_encodings(solvers) -> SolverEnc:
    keys = registry_order(solvers)
    styles = [solver_style(s) for s in keys]
    labels = [s[0] for s in styles]
    return SolverEnc(
        keys=keys,
        labels=labels,
        color=alt.Scale(domain=labels, range=[s[1] for s in styles]),
        dash=alt.Scale(domain=labels, range=[s[2] for s in styles]),
        shape=alt.Scale(domain=labels, range=[s[3] for s in styles]),
    )


# ── Axes, legends, interactions ──────────────────────────────────────────────

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def log_axis(
    title: str | None = None, lo: int = -16, hi: int = 8, **kw: Any
) -> alt.Axis:
    """Ticks, grid and labels on decades only, labelled ``10⁻⁴``."""
    exps = list(range(lo, hi + 1))
    expr = " : ".join(
        f"datum.value == {10.0**e!r} ? '10{str(e).translate(_SUPERSCRIPT)}'"
        for e in exps
    )
    return alt.Axis(
        values=[10.0**e for e in exps], labelExpr=f"{expr} : ''", title=title, **kw
    )


def log_scale(**kw: Any) -> alt.Scale:
    return alt.Scale(type="log", **kw)


def solver_legend(enc: SolverEnc, columns: int = 6) -> alt.Legend:
    return alt.Legend(
        orient="bottom", direction="horizontal", columns=columns, title=None
    )


def legend_pick() -> alt.Parameter:
    """Click a legend entry to fade the other solvers."""
    return alt.selection_point(fields=["solver"], bind="legend")


def faded(pick: alt.Parameter) -> Any:
    return alt.condition(pick, alt.value(1.0), alt.value(0.12))


def solver_lines(
    data: pd.DataFrame | None,
    enc: SolverEnc,
    *,
    x: alt.X,
    y: alt.Y,
    tooltip: list,
    pick: alt.Parameter,
    points: bool = True,
    legend: alt.Legend | None = None,
) -> alt.LayerChart:
    """Per-solver lines (+ markers) with the registry styles and a legend pick."""
    base = alt.Chart(data) if data is not None else alt.Chart()
    base = base.encode(
        x=x,
        y=y,
        color=alt.Color(
            "solver:N",
            scale=enc.color,
            sort=enc.labels,
            legend=legend or solver_legend(enc),
        ),
        opacity=faded(pick),
        tooltip=tooltip,
    )
    layers = [
        base.mark_line(strokeWidth=2).encode(
            strokeDash=alt.StrokeDash("solver:N", scale=enc.dash, legend=None)
        )
    ]
    if points:
        layers.append(
            base.mark_point(filled=True, size=45).encode(
                shape=alt.Shape("solver:N", scale=enc.shape, legend=None)
            )
        )
    return alt.layer(*layers)


# ── Fields (heatmaps) ────────────────────────────────────────────────────────

# Heatmap cells: no band padding, so neighbouring cells don't show seams.
CELLS = alt.Scale(paddingInner=0, paddingOuter=0)


def field_cells(arr: np.ndarray, **cols: Any) -> pd.DataFrame:
    """2-D array → tidy cells (i, j, value) in ``field_grid`` orientation.

    ``field_grid`` draws ``imshow(arr.T, origin="lower")``: the first index runs
    along x and the second up the y axis. Fields larger than :data:`MAX_FIELD`
    are subsampled.
    """
    a = np.asarray(arr, dtype=float)
    step = max(1, math.ceil(max(a.shape) / MAX_FIELD))
    a = a[::step, ::step]
    ii, jj = np.indices(a.shape)
    df = pd.DataFrame({"i": ii.ravel(), "j": jj.ravel(), "value": a.ravel()})
    for k, v in cols.items():
        df[k] = v
    return df


def field_xy() -> dict:
    """x/y encodings for :func:`field_cells` tables (field_grid orientation)."""
    return {
        "x": alt.X("i:O", axis=None, scale=CELLS),
        "y": alt.Y("j:O", axis=None, sort="descending", scale=CELLS),
    }


def linked_cell() -> alt.Parameter:
    """Hover a cell to outline the same cell in every panel."""
    return alt.selection_point(
        fields=["i", "j"], on="pointerover", clear="pointerout", empty=False
    )


def cell_outline(cell: alt.Parameter, color: str = "#ffffff") -> dict:
    return {
        "stroke": alt.condition(cell, alt.value(color), alt.value(None)),
        "strokeWidth": alt.condition(cell, alt.value(2), alt.value(0)),
    }


def cell_px(n: int, total: int = 150) -> int:
    """Panel size in px for an n×n field (about ``total`` px wide)."""
    return max(4, round(total / max(n, 1))) * n


# ── Themes and export ────────────────────────────────────────────────────────

PAPER_CONFIG = {
    "background": "white",
    "font": "Helvetica, Arial, sans-serif",
    "view": {"stroke": None},
    "title": {"fontSize": 11, "fontWeight": "bold", "anchor": "start"},
    "axis": {
        "labelFontSize": 9,
        "titleFontSize": 10,
        "titleFontWeight": "normal",
        "gridColor": "#e6e6e6",
        "domainColor": "#333333",
        "tickColor": "#333333",
    },
    "header": {"labelFontSize": 10, "labelFontWeight": "bold"},
    "legend": {"labelFontSize": 9, "symbolSize": 60},
}


def paper(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    return chart.configure(**PAPER_CONFIG)


def web_spec(chart: alt.TopLevelMixin, *, renderer: str = "svg") -> dict:
    """Spec for the site; ``usermeta.renderer`` picks svg or canvas in the browser."""
    spec = chart.to_dict()
    spec.setdefault("usermeta", {})["renderer"] = renderer
    return spec
