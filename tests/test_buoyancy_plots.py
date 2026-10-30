# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Buoyancy chart rendering and shared solver-style integration."""

import json
from types import SimpleNamespace

import pytest

from mosaic.benchmarks.problems.buoyancy_grid import plots
from mosaic.benchmarks.problems.shared.plots.style import (
    resolve_solver_alias,
    solver_order_for_problem,
)


@pytest.mark.parametrize("selected_update", [0, 64])
def test_combined_rollout_and_validation_include_all_solvers(
    tmp_path, monkeypatch, selected_update
):
    monkeypatch.setattr(plots, "results_dir", lambda: tmp_path)
    names = ["PhiFlow", "JAX-CFD", "INS.jl", "WaterLily", "Oceananigans"]
    result = {"params": {"physics": {"dt": 0.005, "steps": 8}}, "results": []}
    for name in names:
        errors = {"velocity_mse": [0, 0.04, 0.09], "scalar_mse": [0, 0.01, 0.04]}
        trained = {
            **errors,
            "selected_candidate": 0,
            "selected_update": selected_update,
            "candidates": [
                {
                    "validation_checkpoints": [
                        {"update": 0, "loss": 1.0},
                        {"update": 64, "loss": 0.5},
                    ]
                }
            ],
        }
        result["results"].append(
            {
                "solver": name,
                "metrics": {
                    "status": "ok",
                    "uncorrected": errors,
                    "solver_in_loop": trained,
                },
            }
        )
    monkeypatch.setattr(plots, "load_experiment_result", lambda _: result)
    monkeypatch.setattr(plots, "try_load_npz", lambda _: {})
    from mosaic.benchmarks.problems.buoyancy_grid import animation

    monkeypatch.setattr(animation, "write_error_gif", lambda *args: None)
    folder = tmp_path / "buoyancy-grid/optimization/solver_in_loop"
    folder.mkdir(parents=True)
    (folder / "training_cost_PhiFlow.png").write_bytes(b"old")
    charts = plots.plot_solver_in_loop(SimpleNamespace(name="buoyancy-grid"))
    assert len(charts) == 2
    assert {p.stem for p in folder.glob("*.png")} == {
        "rollout_errors",
        "validation_loss",
    }
    for stem in ["rollout_errors", "validation_loss"]:
        assert (folder / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
        spec = json.loads((folder / f"{stem}.vl.json").read_text())
        rows = [row for dataset in spec["datasets"].values() for row in dataset]
        assert {row["solver"] for row in rows} == set(names)
    rollout = json.loads((folder / "rollout_errors.vl.json").read_text())
    assert rollout["resolve"]["scale"]["y"] == "independent"


def test_buoyancy_shared_styles_cover_every_backend():
    order = solver_order_for_problem("buoyancy-grid")
    assert order == [
        "phiflow",
        "jax_cfd",
        "ins_jl",
        "waterlily",
        "oceananigans",
        "openfoam",
    ]
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
    assert spec["spec"]["encoding"]["y"]["axis"]["format"] == ".3~g"
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


def test_horizon_sweep_includes_all_buoyancy_solvers():
    from mosaic.benchmarks.problems.shared.plots.charts.sweeps import horizon_sweep

    names = ["PhiFlow", "JAX-CFD", "INS.jl", "WaterLily", "Oceananigans"]
    data = {
        "by_solver": {
            name: {
                str(step): {
                    "grad_norm": 1.0,
                    "eps_sweep": {
                        "0.01": {"rel_error_mean": 0.001, "cosine_mean": 0.999}
                    },
                }
                for step in [1, 4, 16]
            }
            for name in names
        }
    }
    chart = horizon_sweep(
        SimpleNamespace(
            config=SimpleNamespace(name="buoyancy-grid"),
            legacy=lambda: data,
        )
    )
    spec = chart.to_dict()
    rows = [row for dataset in spec["datasets"].values() for row in dataset]
    assert {row["solver"] for row in rows} == set(names)
    for name in names:
        assert len([row for row in rows if row["solver"] == name]) == 3
