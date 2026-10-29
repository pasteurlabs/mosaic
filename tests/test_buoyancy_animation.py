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
    _panel_indices,
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


def test_animation_all_solvers_frames_and_fixed_paired_row_scales(tmp_path):
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
        assert zero.getpixel((181, 175)) == zero.getpixel((296, 175)) == (0, 0, 0)
        animation.seek(2)
        frame = animation.convert("RGB")
        assert animation.width <= 1100
        for row in range(5):
            baseline = frame.getpixel((181, 175 + row * 174))
            corrected = frame.getpixel((418, 175 + row * 174))
            assert baseline[0] == baseline[1] == baseline[2] == 255
            assert corrected[0] == corrected[1] == corrected[2]
            assert 0 < corrected[0] < baseline[0]
            red = frame.getpixel((296, 175 + row * 174))
            corrected_red = frame.getpixel((533, 175 + row * 174))
            blue = frame.getpixel((780, 175 + row * 174))
            assert red[0] > red[2] and blue[2] > blue[0]
            assert sum(red) > sum(corrected_red) > 0
    assert path.with_suffix(".png").exists()
    receipt = json.loads(path.with_suffix(".json").read_text())
    assert receipt["solvers"] == names
    assert len(receipt["columns"]) == 8
    for index, name in enumerate(names):
        scale = receipt["row_scales"][name]
        assert scale["temperature_range"] == [0, index + 1]
        assert scale["speed_range"] == [0, index + 1]
        assert scale["temperature_error_max"] == index + 1
        assert scale["velocity_error_max"] == index + 1
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
    assert red_indices.min() == 0 and red_indices.max() == 95
    assert blue_indices.min() == 96 and blue_indices.max() == 191
    red, blue = palette[red_indices], palette[blue_indices]
    np.testing.assert_array_equal(red[0], [0, 0, 0])
    np.testing.assert_array_equal(blue[0], red[0])
    assert np.all(red[1:, 0] > red[1:, 2])
    assert np.all(blue[1:, 2] > blue[1:, 0])
    assert np.all(np.diff(red.astype(int).sum(axis=1)) > 0)
    assert np.all(np.diff(blue.astype(int).sum(axis=1)) > 0)
    np.testing.assert_array_equal(palette[255], [255, 255, 255])


def test_corrected_fields_remain_visible_when_errors_are_zero(tmp_path):
    cfg = SimpleNamespace(solvers=[SimpleNamespace(name="PhiFlow")], domain_extent=1.0)
    state = np.zeros((2, 4, 4, 3), dtype=np.float32)
    state[..., 2] = np.linspace(-1, 1, 4)[None, :, None]
    state[:, :-1, :, 0] = 1
    snapshots = {"solver_names": np.array(["PhiFlow"])}
    snapshots.update(
        {f"{arm}_0": state for arm in ("reference", "uncorrected", "solver_in_loop")}
    )
    result = {
        "params": {"physics": {"dt": 0.005, "steps": 8}},
        "results": [
            {
                "solver": "PhiFlow",
                "metrics": {"status": "ok", "solver_in_loop": {"selected_update": 0}},
            }
        ],
    }
    path = write_error_gif(cfg, tmp_path, result, snapshots)
    with Image.open(path) as animation:
        frame = animation.convert("RGB")
        for x in (181, 418, 665, 902):
            gray = frame.getpixel((x, 175))
            assert gray[0] == gray[1] == gray[2] and gray[0] > 0
        for x in (296, 533, 780, 1017):
            assert frame.getpixel((x, 175)) == (0, 0, 0)
    receipt = json.loads(path.with_suffix(".json").read_text())
    scale = receipt["row_scales"]["PhiFlow"]
    assert scale["temperature_range"] == [-1, 1]
    assert scale["temperature_error_max"] == scale["velocity_error_max"] == 0


def test_separate_panels_keep_uniform_errors_visible_and_independent():
    actual = np.full((32, 32), 0.6)
    error = np.full_like(actual, 0.5)
    baseline_field = _panel_indices(actual, (0, 1), 2)
    corrected_field = _panel_indices(actual / 2, (0, 1), 2)
    baseline_error = _panel_indices(error, (0, 1), 0)
    corrected_error = _panel_indices(error, (0, 1), 0)
    assert np.all(baseline_field > corrected_field)
    np.testing.assert_array_equal(baseline_error, corrected_error)
    palette = np.asarray(_palette()).reshape(256, 3)
    assert np.all(palette[baseline_error][..., 0] > palette[baseline_error][..., 2])
    # Tiny errors are legible under their own paired row scale, without blending.
    np.testing.assert_array_equal(
        _panel_indices(error * 0.001, (0, 0.001), 0), baseline_error
    )
    blue = palette[_panel_indices(error, (0, 1), 1)]
    assert np.all(blue[..., 2] > blue[..., 0])
