# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward suite plots: solver agreement, raw fields, physical laws."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import results_dir
from mosaic.benchmarks.problems.shared.plots.charts import forward as charts
from mosaic.benchmarks.problems.shared.plots.charts.common import render

_SUITE = "forward"


def plot_agreement(
    cfg: Problem,
    *,
    agreement_xlabel: str = "x",
    agreement_ylabel: str = "value",
    save: bool = True,
    suffix: str = "",
    exp_key: str = "agreement",
    **_kw: Any,
) -> Any:
    """Inter-solver agreement for one forward experiment.

    The chart depends on the shape of the stored outputs (``fields.npz``):
    scalar outputs give ``curves.png`` (output vs sweep value), 1-D outputs
    give ``curves.png`` as profiles with residuals, and fields give
    ``agreement.png`` (error vs sweep value).
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    ndim = None
    fields = out_dir / "fields.npz"
    if fields.exists():
        with np.load(fields, allow_pickle=False) as npz:
            if "sweep_values" in npz.files and "consensus_0" in npz.files:
                ndim = npz["consensus_0"].ndim
    if ndim == 0:
        builders = {"curves": charts.curves}
    elif ndim == 1:
        builders = {
            "curves": functools.partial(
                charts.profiles, xlabel=agreement_xlabel, ylabel=agreement_ylabel
            )
        }
    else:
        builders = {"agreement": charts.agreement}
    out = render(
        cfg, out_dir, builders, suite=_SUITE, experiment=f"{exp_key}{suffix}", save=save
    )
    return next(iter(out.values()), None)


def plot_forward_fields(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "cylinder",
    **_kw: Any,
) -> Any:
    """``fields_raw.png``: per-solver vorticity fields across the sweep."""
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    out = render(
        cfg,
        out_dir,
        {"fields_raw": charts.fields_raw},
        suite=_SUITE,
        experiment=f"{exp_key}{suffix}",
        save=save,
    )
    return out.get("fields_raw")


def plot_physical_laws(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "physical_laws",
    **_kw: Any,
) -> Any:
    """``physical_accuracy.png``: physical-law checks (e.g. compliance scaling)."""
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    out = render(
        cfg,
        out_dir,
        {"physical_accuracy": charts.physical_accuracy},
        suite=_SUITE,
        experiment=f"{exp_key}{suffix}",
        save=save,
    )
    return out.get("physical_accuracy")
