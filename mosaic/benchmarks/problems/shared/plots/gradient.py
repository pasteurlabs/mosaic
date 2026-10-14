# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradient suite plots: FD check, parameter / horizon sweeps, Jacobian SVD."""

from __future__ import annotations

from typing import Any

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import results_dir
from mosaic.benchmarks.problems.shared.plots.charts import gradient, sweeps
from mosaic.benchmarks.problems.shared.plots.charts.common import render

_SUITE = "gradient"


def _render(cfg: Problem, exp: str, builders: dict, save: bool) -> dict:
    return render(
        cfg,
        results_dir() / cfg.name / _SUITE / exp,
        builders,
        suite=_SUITE,
        experiment=exp,
        save=save,
    )


def plot_fd_check(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "fd_check",
    **_kw: Any,
) -> Any:
    """``fd_check.png`` (FD error / 1 − cos vs ε) and ``gradient_fields.png``."""
    out = _render(
        cfg,
        f"{exp_key}{suffix}",
        {"fd_check": gradient.fd_check, "gradient_fields": gradient.gradient_fields},
        save,
    )
    return out.get("fd_check")


def plot_param_sweep(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "param_sweep",
    **_kw: Any,
) -> Any:
    """``best_eps_vs_param.png``: best-ε FD error vs the swept parameter."""
    out = _render(
        cfg, f"{exp_key}{suffix}", {"best_eps_vs_param": sweeps.best_eps_vs_param}, save
    )
    return out.get("best_eps_vs_param")


def plot_horizon_sweep(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "horizon_sweep",
    **_kw: Any,
) -> Any:
    """``horizon_sweep.png``: gradient norm, FD error and cosine vs rollout T."""
    out = _render(
        cfg, f"{exp_key}{suffix}", {"horizon_sweep": sweeps.horizon_sweep}, save
    )
    return out.get("horizon_sweep")


def plot_horizon_sweep_limits(
    cfg: Problem,
    *,
    save: bool = True,
    exp_key: str = "horizon_sweep_limits",
    **_kw: Any,
) -> Any:
    """``horizon_sweep_limits.png``: memory, wall time and gradient norm vs T."""
    out = _render(
        cfg, exp_key, {"horizon_sweep_limits": sweeps.horizon_sweep_limits}, save
    )
    return out.get("horizon_sweep_limits")


def plot_jacobian_svd_comparison(
    cfg: Problem,
    exp_keys: list[str] | None = None,
    *,
    save: bool = True,
    out_name: str = "jacobian_svd_comparison",
    **_kw: Any,
) -> Any:
    """Suite-level spectra of all ``jacobian_svd*`` variants, one panel per solver."""
    out = render(
        cfg,
        results_dir() / cfg.name / _SUITE,
        {out_name: sweeps.jacobian_svd_comparison},
        suite=_SUITE,
        save=save,
    )
    return out.get(out_name)
