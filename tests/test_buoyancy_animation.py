# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiny error-animation fixtures; rendering runs on Slurm/CI."""

import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from mosaic.benchmarks.problems.buoyancy_grid.animation import (
    _center_velocity,
    _field_indices,
    _palette,
    _reduce,
    write_error_gif,
)


def test_centered_vector_error_is_not_difference_of_speeds():
    state = np.zeros((2, 4, 4, 3))
    state[:, :-1, :, 0] = 2
    opposite = -state
    a, b = _center_velocity(state), _center_velocity(opposite)
    np.testing.assert_array_equal(
        np.linalg.norm(a, axis=-1) - np.linalg.norm(b, axis=-1), 0
    )
    assert np.max(np.linalg.norm(a - b, axis=-1)) == 4


def test_spatial_mean_preserves_positive_error_without_cancellation():
    fields = np.arange(64 * 64, dtype=float).reshape(1, 64, 64)
    reduced = _reduce(fields)
    assert reduced.shape == (1, 32, 32)
    assert reduced.mean() == fields.mean()


def test_animation_all_solvers_frames_and_global_scales(tmp_path):
    names = ["PhiFlow", "JAX-CFD", "INS.jl", "WaterLily", "Oceananigans"]
    cfg = SimpleNamespace(
        solvers=[SimpleNamespace(name=n) for n in names], domain_extent=1.0
    )
    result = {"params": {"physics": {"dt": 0.005, "steps": 8}}, "results": []}
    snapshots = {"solver_names": np.array(names)}
    for index, name in enumerate(names):
        reference = np.zeros((3, 4, 4, 3), dtype=np.float32)
        baseline = reference.copy()
        baseline[1:, ..., 2] = index + 1
        baseline[1:, :-1, :, 0] = index + 1
        corrected = baseline / 2
        for key, array in (
            ("reference", reference),
            ("uncorrected", baseline),
            ("solver_in_loop", corrected),
        ):
            snapshots[f"{key}_{index}"] = array
        result["results"].append(
            {
                "solver": name,
                "metrics": {"status": "ok", "solver_in_loop": {"selected_update": 64}},
            }
        )
    path = write_error_gif(cfg, tmp_path, result, snapshots)
    with Image.open(path) as animation:
        assert animation.n_frames == 3
        assert animation.info["duration"] == 100
        assert animation.info["loop"] == 0
        zero = animation.convert("RGB")
        assert zero.getpixel((197, 188)) == zero.getpixel((509, 188)) == (0, 0, 0)
        animation.seek(2)
        frame = animation.convert("RGB")
        temperature, velocity = frame.getpixel((197, 188)), frame.getpixel((509, 188))
        assert temperature[0] > temperature[2]
        assert velocity[2] > velocity[0]
    assert path.with_suffix(".png").exists()
    receipt = json.loads(path.with_suffix(".json").read_text())
    assert receipt["solvers"] == names
    assert receipt["scalar_max"] == 5
    assert receipt["velocity_max"] == 5
    assert receipt["frames"] == 3 and receipt["frame_interval"] == 0.04


def test_docs_prefer_animation_over_poster_and_interactive_chart(tmp_path):
    from docs.generate_results import _chart_block, _img_tag

    png = tmp_path / "error_evolution.png"
    png.write_bytes(b"poster")
    png.with_suffix(".gif").write_bytes(b"animation")
    png.with_suffix(".vl.json").write_text("{}")
    assert "error_evolution.gif" in _img_tag(
        "buoyancy-grid", "optimization", "solver_in_loop", png
    )
    assert _chart_block("buoyancy-grid", "optimization", "solver_in_loop", png) is None


def test_field_palettes_distinguish_equal_errors_and_preserve_zero():
    palette = np.asarray(_palette(), dtype=np.uint8).reshape(256, 3)
    normalized = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    red_indices = _field_indices(normalized, 0)
    blue_indices = _field_indices(normalized, 1)
    assert red_indices.min() == 0 and red_indices.max() == 119
    assert blue_indices.min() == 120 and blue_indices.max() == 239
    red, blue = palette[red_indices], palette[blue_indices]
    np.testing.assert_array_equal(red[0], [0, 0, 0])
    np.testing.assert_array_equal(blue[0], red[0])
    assert np.all(red[1:, 0] > red[1:, 2])
    assert np.all(blue[1:, 2] > blue[1:, 0])
    assert np.all(np.diff(red.astype(int).sum(axis=1)) > 0)
    assert np.all(np.diff(blue.astype(int).sum(axis=1)) > 0)
    np.testing.assert_array_equal(palette[255], [255, 255, 255])
