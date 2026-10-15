# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Altair chart builders behind the result plots."""

from __future__ import annotations

import json

import pytest

from mosaic.benchmarks.problems.shared.plots.charts import builders
from mosaic.benchmarks.problems.shared.plots.charts.common import (
    Ctx,
    field_cells,
    log_axis,
)

# Every PNG stem the results pages show (docs/generate_results.py).
EXPECTED = {
    "agreement",
    "best_eps_vs_param",
    "conductivity_recovery_convergence",
    "convergence_curves",
    "cost",
    "curves",
    "drag_opt",
    "drag_opt_fields",
    "fd_check",
    "fields_raw",
    "gradient_fields",
    "horizon_sweep",
    "horizon_sweep_limits",
    "ics",
    "jacobian_svd_comparison",
    "physical_accuracy",
    "topopt",
    "topopt_3d",
    "topopt_fields",
    # Animations (last frame as PNG, slider on the web)
    "drag_opt_evolution",
    "recovery_evolution",
    "topopt_evolution_FEniCS",
}


def test_every_plot_has_a_builder():
    assert set(builders()) >= EXPECTED


def test_log_axis_labels_decades():
    axis = log_axis("ε", lo=-2, hi=1).to_dict()
    assert axis["values"] == [0.01, 0.1, 1.0, 10.0]
    assert "'10⁻²'" in axis["labelExpr"]


def test_field_cells_orientation_and_subsampling():
    import numpy as np

    arr = np.arange(6.0).reshape(2, 3)  # first index → x (i), second → y (j)
    df = field_cells(arr)
    assert df.loc[(df.i == 1) & (df.j == 2), "value"].item() == 5.0
    big = field_cells(np.zeros((256, 128)))
    assert big["i"].max() + 1 <= 64


@pytest.mark.parametrize("sweep_key", ["steps", "perturb_sigma"])
def test_convergence_curves_reads_stored_sweep(tmp_path, sweep_key):
    from mosaic.benchmarks.core.experiment import (
        _build_result_envelope,
        _flatten_by_solver,
    )
    from mosaic.benchmarks.problems import get_config
    from mosaic.benchmarks.problems.shared.plots.charts.optimization import (
        convergence_curves,
    )

    cfg = get_config("ns-3d-grid")
    metrics = {"errors": [1.0, 0.1, 0.01], "final_ic_error": 0.2, "converged": True}
    result = _build_result_envelope(
        cfg=cfg,
        suite="optimization",
        exp_key="recovery",
        run={},
        sweep_key=sweep_key,
        sweep_values=[100],
        results=_flatten_by_solver(
            {"PhiFlow": {100: metrics}, "XLB": {100: metrics}}, sweep_key
        ),
    )
    (tmp_path / "result.json").write_text(json.dumps(result))
    ctx = Ctx("ns-3d-grid", "optimization", "recovery", tmp_path / "x.png", cfg)

    chart = convergence_curves(ctx)

    assert chart is not None
    rows = [r for ds in chart.to_dict()["datasets"].values() for r in ds]
    assert {r["solver"] for r in rows if "loss" in r} == {"PhiFlow", "XLB"}


def _cost_result(hw_key: str, hw: dict, *solvers: str) -> dict:
    entries = [
        {"solver": s, "sweep_value": "32", "metrics": {"mean": 1.0}} for s in solvers
    ]
    return {"schema_version": 1, "results": entries, "extras": {hw_key: hw}}


def test_cost_subtitle_names_each_solvers_device():
    from types import SimpleNamespace

    from mosaic.benchmarks.problems.shared.plots.charts.cost import _hardware_str

    cfg = SimpleNamespace(
        solvers=[
            SimpleNamespace(name="JAX-FEM", uses_gpu=True),
            SimpleNamespace(name="FEniCS", uses_gpu=False),
        ]
    )
    gpu_node = {
        "gpus": ["Tesla T4, 16384 MiB"],
        "cpu": "AMD EPYC 7V12 64-Core Processor",
    }
    cpu_node = {"cpu": "INTEL(R) XEON(R) PLATINUM 8573C"}

    # One machine ran everything: the CPU-only solver used the GPU node's CPU.
    single = _cost_result("hardware", gpu_node, "JAX-FEM", "FEniCS")
    assert _hardware_str([single], cfg) == "GPU Tesla T4 · CPU AMD EPYC 7V12"

    # CI: the CPU-only solver ran on a separate CPU node.
    split = _cost_result(
        "hardware_by_solver", {"JAX-FEM": gpu_node, "FEniCS": cpu_node}
    )
    assert _hardware_str([split], cfg) == "GPU Tesla T4 · CPU INTEL XEON PLATINUM 8573C"
