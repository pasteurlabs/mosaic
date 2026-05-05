"""IC suite: generate and visualise all initial conditions for a problem config.

Unlike the solver suites (forward, gradient, recovery, cost), this suite does
not invoke any Docker containers — it only calls the IC generator functions
defined in each ProblemConfig.  This makes it fast and dependency-free.

Results are saved to results/{problem}/ics/{ic_name}/:
  ic.png / ic.pdf  — 2-D projection of the IC
  params.json      — IC generation kwargs used for the plot
"""

from __future__ import annotations

import json
from pathlib import Path

from mosaic.benchmarks.core.config import ProblemConfig

_RESULTS_DIR = Path(__file__).parent.parent / "results"
_SUITE = "ics"


def _run_ic(
    cfg: ProblemConfig,
    ic_name: str,
    params: dict,
) -> dict:
    """Generate one IC, save a visualisation plot and params.json."""
    from mosaic.benchmarks.plots.ics import plot_ic

    out_dir = _RESULTS_DIR / cfg.name / _SUITE / ic_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ic = cfg.make_ic[ic_name](**params)
    plot_ic(cfg, ic_name, ic, out_dir)

    (out_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return {"ic_name": ic_name, "shape": list(ic.shape)}


def get_experiments(cfg: ProblemConfig) -> dict[str, object]:
    """Return ``{ic_name: exp_fn}`` for each IC registered in *cfg*.

    Each ``exp_fn(cfg, tags)`` generates the IC, saves its visualisation, and
    returns ``{"ic_name": ..., "shape": ...}``.  The *tags* argument is
    accepted but unused (no solver containers are started).
    """
    experiments: dict[str, object] = {}
    for ic_name in cfg.make_ic:
        params = dict(cfg.get_ic_plot_params(ic_name))

        def _make(name: str = ic_name, p: dict = params):
            def exp(cfg: ProblemConfig, _tags: dict) -> dict:
                return _run_ic(cfg, name, p)

            exp.__name__ = name
            return exp

        experiments[ic_name] = _make()
    return experiments


def get_plot_fns(cfg: ProblemConfig) -> dict[str, object]:
    """Return ``{ic_name: fn(cfg)}`` for regenerating IC plots from scratch."""
    fns: dict[str, object] = {}
    for ic_name in cfg.make_ic:
        params = dict(cfg.get_ic_plot_params(ic_name))

        def _make(name: str = ic_name, p: dict = params):
            def fn(cfg: ProblemConfig, **_kwargs) -> None:
                out_dir = _RESULTS_DIR / cfg.name / _SUITE / name
                ic = cfg.make_ic[name](**p)
                from mosaic.benchmarks.plots.ics import plot_ic

                plot_ic(cfg, name, ic, out_dir)

            fn.__name__ = name
            return fn

        fns[ic_name] = _make()
    return fns
