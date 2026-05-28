# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``mosaic_shared/utils/plotting`` helpers.

Covers:
- ``plotting.plot_mesh``: writes a non-empty image file at the requested path.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_plot_mesh_writes_image_file(tmp_path):
    """plot_mesh saves a PNG with non-zero size when save_path is given."""
    from mosaic.mosaic_shared.utils.plotting import plot_mesh

    # Minimal valid mesh: a single triangle in 3D.
    mesh = {
        "points": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        "faces": np.array([[0, 1, 2]]),
    }
    out = tmp_path / "mesh.png"
    plot_mesh(mesh, bounds=(2.0, 2.0, 2.0), save_path=str(out))
    assert out.exists(), "plot_mesh did not save the file"
    assert out.stat().st_size > 0, "saved image is empty"


def test_plot_mesh_no_save_path_does_not_write(tmp_path, monkeypatch):
    """When save_path is None, plot_mesh shouldn't write any files."""
    from mosaic.mosaic_shared.utils.plotting import plot_mesh

    mesh = {
        "points": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        "faces": np.array([[0, 1, 2]]),
    }
    before = set(tmp_path.iterdir())
    plot_mesh(mesh, bounds=(2.0, 2.0, 2.0), save_path=None)
    after = set(tmp_path.iterdir())
    assert before == after  # nothing written
    plt.close("all")  # clean up the figure plot_mesh left open


# Compatibility note: matplotlib leaves figures in memory unless explicitly
# closed; the test suite doesn't assert on figure-count, but close-all keeps
# memory usage bounded across the full run.
@pytest.fixture(autouse=True)
def _close_all_figures():
    yield
    plt.close("all")
