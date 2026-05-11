"""Tests for ``mosaic_shared/utils/`` helpers.

Covers:
- ``sdf.generate_regular_grid``: shape, spacing, bounds, dtype.
- ``comparisons.make_ensemble_deviation_plot`` and
  ``make_gradient_cosine_plot``: factory output draws onto the supplied axes
  without error and uses every solver in the dict.
- ``plotting.plot_mesh``: writes a non-empty image file at the requested path.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mosaic.mosaic_shared.utils.sdf import generate_regular_grid

# ── sdf.generate_regular_grid ─────────────────────────────────────────────────


def test_generate_regular_grid_shape():
    """Grid has shape (nx, ny, nz, 3) with nx/ny/nz determined by bounds/resolution."""
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    grid = generate_regular_grid(resolution=0.25, grid_bounds=bounds)
    # arange(0, 1, 0.25) gives 4 samples (0, 0.25, 0.5, 0.75)
    assert grid.shape == (4, 4, 4, 3)


def test_generate_regular_grid_corner_value():
    """The (0,0,0) entry is the lower-bound corner."""
    bounds = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]])
    grid = generate_regular_grid(resolution=0.5, grid_bounds=bounds)
    np.testing.assert_allclose(grid[0, 0, 0], [-1.0, -2.0, -3.0])


def test_generate_regular_grid_spacing():
    """Adjacent points along each axis differ by exactly resolution."""
    bounds = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    grid = generate_regular_grid(resolution=0.1, grid_bounds=bounds)
    dx = grid[1, 0, 0, 0] - grid[0, 0, 0, 0]
    dy = grid[0, 1, 0, 1] - grid[0, 0, 0, 1]
    dz = grid[0, 0, 1, 2] - grid[0, 0, 0, 2]
    np.testing.assert_allclose([dx, dy, dz], [0.1, 0.1, 0.1])


def test_generate_regular_grid_respects_upper_bound():
    """No grid point exceeds the upper bound (arange is half-open)."""
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    grid = generate_regular_grid(resolution=0.25, grid_bounds=bounds)
    assert grid[..., 0].max() < 1.0
    assert grid[..., 1].max() < 1.0
    assert grid[..., 2].max() < 1.0


def test_generate_regular_grid_anisotropic_bounds():
    """Different per-axis extents produce different per-axis sample counts."""
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 4.0]])
    grid = generate_regular_grid(resolution=1.0, grid_bounds=bounds)
    assert grid.shape == (1, 2, 4, 3)


# ── comparisons: ensemble deviation plot ──────────────────────────────────────


class _FakeOut:
    """Stand-in for an OutputSchema instance — exposes a ``.result`` array."""

    def __init__(self, arr):
        self.result = arr


def test_ensemble_deviation_plot_renders_one_axis_per_solver():
    """The factory's plot fn draws an imshow on every supplied axis."""
    from mosaic.mosaic_shared.utils.comparisons import make_ensemble_deviation_plot

    plot_fn = make_ensemble_deviation_plot(
        get_field=lambda out: out.result,
        title="test deviation",
    )

    rng = np.random.default_rng(0)
    results = {
        "solver_a": _FakeOut(rng.normal(size=(8, 8))),
        "solver_b": _FakeOut(rng.normal(size=(8, 8))),
        "solver_c": _FakeOut(rng.normal(size=(8, 8))),
    }

    fig, axes = plt.subplots(1, 3)
    try:
        plot_fn(inputs=None, results=results, axes=list(axes))
        # Every axis must have at least one image drawn on it.
        for ax in axes:
            assert ax.images, "axis is missing the imshow output"
        # Each title is the solver name with an RMS value.
        for ax, name in zip(axes, results.keys(), strict=False):
            assert name in ax.get_title()
            assert "RMS" in ax.get_title()
        assert fig._suptitle.get_text() == "test deviation"
    finally:
        plt.close(fig)


def test_ensemble_deviation_handles_3d_fields():
    """Fields with ndim > 2 are reduced via a per-cell norm before plotting."""
    from mosaic.mosaic_shared.utils.comparisons import make_ensemble_deviation_plot

    plot_fn = make_ensemble_deviation_plot(get_field=lambda out: out.result)

    # Shape (H, W, 3) — e.g. a 2D vector field with 3 components.
    rng = np.random.default_rng(1)
    results = {
        "a": _FakeOut(rng.normal(size=(6, 6, 3))),
        "b": _FakeOut(rng.normal(size=(6, 6, 3))),
    }

    fig, axes = plt.subplots(1, 2)
    try:
        plot_fn(None, results, list(axes))
        for ax in axes:
            assert ax.images
    finally:
        plt.close(fig)


# ── comparisons: gradient cosine plot ─────────────────────────────────────────


def test_gradient_cosine_plot_draws_n_plus_one_axes():
    """The factory draws one magnitude map per solver + one similarity matrix."""
    from mosaic.mosaic_shared.utils.comparisons import make_gradient_cosine_plot

    plot_fn = make_gradient_cosine_plot(get_field=lambda g: g.result)

    rng = np.random.default_rng(2)
    # Shape (H, W, C) so the magnitude path runs along axis=-1.
    grads = {
        "a": _FakeOut(rng.normal(size=(5, 5, 2))),
        "b": _FakeOut(rng.normal(size=(5, 5, 2))),
        "c": _FakeOut(rng.normal(size=(5, 5, 2))),
    }
    n = len(grads)

    fig, axes = plt.subplots(1, n + 1)
    try:
        plot_fn(None, grads, list(axes))
        # First n axes are the per-solver magnitude maps; axes[n] is the
        # similarity matrix.
        for ax in axes[:n]:
            assert ax.images
        sim_ax = axes[n]
        # Similarity matrix is also an imshow; plus we should see one text
        # label per cell (n x n).
        assert sim_ax.images
        assert len(sim_ax.texts) == n * n
    finally:
        plt.close(fig)


def test_gradient_cosine_similarity_diagonal_is_one():
    """A solver's gradient must have cosine similarity 1.0 with itself.

    We can't inspect cos_sim directly (it's local to the closure), but we can
    read the diagonal off the rendered cell text — each diagonal label should
    be '1.00'.
    """
    from mosaic.mosaic_shared.utils.comparisons import make_gradient_cosine_plot

    plot_fn = make_gradient_cosine_plot(get_field=lambda g: g.result)

    rng = np.random.default_rng(3)
    grads = {
        "a": _FakeOut(rng.normal(size=(4, 4, 1))),
        "b": _FakeOut(rng.normal(size=(4, 4, 1))),
    }
    fig, axes = plt.subplots(1, 3)
    try:
        plot_fn(None, grads, list(axes))
        sim_ax = axes[2]
        # Text objects are positioned at (j, i) for entry [i, j]. Diagonal
        # entries have x == y in data coordinates.
        diag_labels = []
        for txt in sim_ax.texts:
            x, y = txt.get_position()
            if x == y:
                diag_labels.append(txt.get_text())
        assert all(lbl == "1.00" for lbl in diag_labels), diag_labels
    finally:
        plt.close(fig)


# ── plotting.plot_mesh ────────────────────────────────────────────────────────


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
