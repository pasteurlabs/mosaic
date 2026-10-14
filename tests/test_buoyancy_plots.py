# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Buoyancy chart rendering and shared solver-style integration."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mosaic.benchmarks.problems.buoyancy_grid import plots
from mosaic.benchmarks.problems.shared.plots.style import (
    resolve_solver_alias,
    solver_order_for_problem,
)


@pytest.mark.parametrize("selected_update", [0, 64])
def test_buoyancy_charts_write_web_specs_and_pngs(
    tmp_path, monkeypatch, selected_update
):
    monkeypatch.setattr(plots, "results_dir", lambda: tmp_path)
    reference = np.zeros((3, 4, 4, 3), dtype=np.float32)
    reference[..., 2] = np.arange(16).reshape(4, 4) / 16
    baseline = reference.copy()
    baseline[1:, ..., 2] += 0.1
    corrected = baseline.copy() if selected_update == 0 else reference.copy()
    snapshots = {
        "solver_names": np.array(["WaterLily"]),
        "reference_0": reference,
        "uncorrected_0": baseline,
        "solver_in_loop_0": corrected,
    }
    checkpoints = [
        {"update": 0, "loss": 1.0},
        {"update": 64, "loss": 1.0 if selected_update == 0 else 0.5},
    ]
    trained = {
        "selected_update": selected_update,
        "selected_candidate": 0,
        "velocity_mse": [0.0, 0.0, 0.0],
        "scalar_mse": [0.0, 0.1, 0.2],
        "total_training_seconds": 2.0,
        "total_validation_seconds": 1.0,
        "total_seconds": 3.0,
        "candidates": [
            {
                "learning_rate": 0.003,
                "validation_checkpoints": checkpoints,
                "validation_loss": checkpoints[-1]["loss"],
            }
        ],
    }
    result = {
        "params": {"physics": {"dt": 0.005, "steps": 8, "domain_extent": 2.0}},
        "results": [
            {
                "solver": "WaterLily",
                "metrics": {
                    "status": "ok",
                    "updates": 64,
                    "uncorrected": {
                        "velocity_mse": [0.0, 0.1, 0.2],
                        "scalar_mse": [0.0, 0.2, 0.3],
                    },
                    "solver_in_loop": trained,
                },
            }
        ],
    }
    monkeypatch.setattr(plots, "load_experiment_result", lambda _: result)
    monkeypatch.setattr(plots, "try_load_npz", lambda _: snapshots)
    charts = plots.plot_solver_in_loop(SimpleNamespace(name="buoyancy-grid"))
    assert len(charts) == 4
    folder = tmp_path / "buoyancy-grid/optimization/solver_in_loop"
    assert len(list(folder.glob("*.png"))) == len(list(folder.glob("*.vl.json"))) == 4
    for png in folder.glob("*.png"):
        assert png.read_bytes().startswith(b"\x89PNG")
    field = json.loads((folder / "fields_WaterLily.vl.json").read_text())
    assert len(field["vconcat"]) == 3
    for row in field["vconcat"]:
        # One scale per field row is shared across all three method panels.
        assert len(row["spec"]["encoding"]["color"]["scale"]["domain"]) == 2
        assert row["spec"]["encoding"]["x"]["scale"]["domain"] == [0, 2.0]
    text = json.dumps(field)
    assert (
        "Selected: no correction"
        if selected_update == 0
        else "Solver in the loop (update 64)"
    ) in text
    rollout = json.loads((folder / "rollout_error_WaterLily.vl.json").read_text())
    assert rollout["spec"]["encoding"]["y"]["scale"]["zero"] is True


def test_buoyancy_shared_styles_cover_every_backend():
    order = solver_order_for_problem("buoyancy-grid")
    assert order == ["phiflow", "jax_cfd", "ins_jl", "waterlily", "oceananigans"]
    assert resolve_solver_alias("WaterLily") == "waterlily"
    assert resolve_solver_alias("Oceananigans") == "oceananigans"


def test_buoyancy_missing_results_skip_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "results_dir", lambda: tmp_path)
    monkeypatch.setattr(plots, "load_experiment_result", lambda _: None)
    assert plots.plot_solver_in_loop(SimpleNamespace(name="buoyancy-grid")) == []


def test_forward_diagnostics_render_signed_values(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "results_dir", lambda: tmp_path)
    result = {
        "sweep": {"key": "steps", "values": ["8", "32"]},
        "results": [
            {
                "solver": "PhiFlow",
                "sweep_value": str(steps),
                "metrics": {
                    "valid": True,
                    "scalar_mass_rel_drift": drift,
                    "divergence_rms": 0.0,
                    "scalar_min": -0.01,
                },
            }
            for steps, drift in [(8, -0.01), (32, -0.02)]
        ],
    }
    monkeypatch.setattr(plots, "load_experiment_result", lambda _: result)
    charts = plots.plot_forward_metrics(SimpleNamespace(name="buoyancy-grid"))
    assert len(charts) == 1
    folder = tmp_path / "buoyancy-grid/forward/physical_laws"
    assert (folder / "diagnostics.png").read_bytes().startswith(b"\x89PNG")
    spec = json.loads((folder / "diagnostics.vl.json").read_text())
    assert spec["spec"]["encoding"]["x"]["title"] == "steps"
    assert spec["resolve"]["scale"]["y"] == "independent"
    assert "log" not in json.dumps(spec["spec"]["encoding"]["y"])


def test_spectrum_label_uses_actual_buoyancy_horizon(tmp_path):
    from mosaic.benchmarks.problems.shared.plots.charts.sweeps import (
        jacobian_svd_comparison,
    )

    directory = tmp_path / "jacobian_svd"
    directory.mkdir()
    (directory / "result.json").write_text("{}")
    data = {
        "params": {"physics": {"nu": 0.001, "dt": 0.01, "steps": 4}},
        "per_solver_spectra": {"PhiFlow": [1.0, 0.5, 0.1]},
    }
    chart = jacobian_svd_comparison(
        SimpleNamespace(dir=tmp_path, legacy=lambda _: data)
    )
    spec = chart.to_dict()
    assert "T=0.04s" in json.dumps(spec)
    assert "T=0.5s" not in json.dumps(spec)
