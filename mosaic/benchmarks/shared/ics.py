"""IC suite primitives: generate one IC, save a 2-D projection.

Unlike the solver suites (forward, gradient, optimization, cost), this suite
does not invoke any Docker containers — it only calls the IC generator
functions a problem registers. The harness functions take ``make_ic``
explicitly so each problem owns its own registration in its
``experiments.py`` / ``plots.py`` files.

Results land at ``<results>/{problem}/ics/{ic_name}/``:

  * ``ic.png`` / ``ic.pdf`` — 2-D projection of the IC
  * ``params.json``         — IC generation kwargs used for the plot
"""

from __future__ import annotations

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import results_dir, save_json

_SUITE = "ics"


def run_ic(
    cfg: Problem,
    ic_name: str,
    *,
    make_ic,
    params: dict,
) -> dict:
    """Generate one IC, save a visualisation plot and params.json."""
    from mosaic.benchmarks.shared.plots.ics import plot_ic

    out_dir = results_dir() / cfg.name / _SUITE / ic_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ic = make_ic[ic_name](**params)
    plot_ic(cfg, ic_name, ic, out_dir, make_ic=make_ic)

    save_json(params, out_dir / "params.json")
    return {"ic_name": ic_name, "shape": list(ic.shape)}


def plot_ic_only(cfg: Problem, ic_name: str, *, make_ic, params: dict) -> None:
    """Regenerate the IC plot without saving params.json (re-plot flow)."""
    from mosaic.benchmarks.shared.plots.ics import plot_ic

    out_dir = results_dir() / cfg.name / _SUITE / ic_name
    ic = make_ic[ic_name](**params)
    plot_ic(cfg, ic_name, ic, out_dir, make_ic=make_ic)
