# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The solver-in-the-loop plots render from a run's result."""

from __future__ import annotations

import json
from types import SimpleNamespace

from mosaic.benchmarks.problems.navier_stokes_grid import kolmogorov as ko
from mosaic.benchmarks.problems.navier_stokes_grid import sil

from .test_sil import _truth, grid_dependent_step


def test_plots_render(tmp_path, monkeypatch):
    from mosaic.benchmarks.problems import get_config
    from mosaic.benchmarks.problems.navier_stokes_grid.plots import (
        plot_kolmogorov_sil,
    )

    monkeypatch.setattr(ko, "load_ground_truth", lambda domain: _truth())
    optim = {
        "unrolls": [1, 2],
        "updates": 2,
        "model_seeds": [0],
        "learning_rates": [1e-3],
        "validation_interval": 1,
        "seeds": {"train": [0], "validation": [1], "test": [2, 3]},
        "train_frames": 3,
        "fd_unroll": 2,
        "eval_frames": 3,
        "audit_frames": 2,
    }
    ctx = SimpleNamespace(
        name="fake",
        cfg=SimpleNamespace(name="ns-grid"),
        run={"optim": optim},
        phys={"N": 4, "dt": 0.05, "reference_factor": 2, "nu": 0.01},
        make_inputs=None,
        output_key="result",
    )
    metrics = sil.run(None, ctx, step_factory=grid_dependent_step)["metrics"]
    out = tmp_path / "ns-grid" / "optimization" / "kolmogorov_sil"
    out.mkdir(parents=True)
    result = {
        "results": [{"solver": "jax-cfd", "sweep_value": None, "metrics": metrics}]
    }
    (out / "result.json").write_text(json.dumps(result, default=float))
    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))

    charts = plot_kolmogorov_sil(get_config("ns-grid"), exp_key="kolmogorov_sil")
    assert len(charts) == 3
    for stem in ("kolmogorov_sil", "kolmogorov_sil_paired", "kolmogorov_sil_cost"):
        assert (out / f"{stem}.png").exists()
