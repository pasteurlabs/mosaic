"""Utilities for signed distance field (SDF) computations."""

import numpy as np


def generate_regular_grid(resolution: float, grid_bounds: np.ndarray) -> np.ndarray:
    """Create a grid of points for the SDF.

    Args:
        resolution: The spacing between grid points.
        grid_bounds: Array of shape (2, 3) defining [[x_min, y_min, z_min], [x_max, y_max, z_max]].

    Returns:
        Grid of points with shape (nx, ny, nz, 3).
    """
    # Generate a regular grid of points
    x = np.arange(
        grid_bounds[0, 0],
        grid_bounds[1, 0],
        resolution,
    )

    y = np.arange(
        grid_bounds[0, 1],
        grid_bounds[1, 1],
        resolution,
    )

    z = np.arange(
        grid_bounds[0, 2],
        grid_bounds[1, 2],
        resolution,
    )

    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
