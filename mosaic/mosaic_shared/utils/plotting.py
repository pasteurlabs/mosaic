"""Utilities for mesh visualization."""

from collections.abc import Sequence

import matplotlib.pyplot as plt


def plot_mesh(
    mesh: dict,
    bounds: Sequence[float],
    save_path: str | None = None,
    figsize: tuple = (10, 6),
) -> None:
    """Plot a 3D triangular mesh with boundary conditions visualization.

    Args:
        mesh: Dictionary containing 'points' and 'faces' arrays.
        bounds: Bounds of the 3D space [Lx, Ly, Lz].
        save_path: Optional path to save the plot as an image file.
        figsize: Size of the matplotlib figure.
    """
    Lx = bounds[0]
    Ly = bounds[1]
    Lz = bounds[2]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        mesh["points"][:, 0],
        mesh["points"][:, 1],
        mesh["points"][:, 2],
        triangles=mesh["faces"],
        alpha=0.7,
        antialiased=True,
        color="lightblue",
        edgecolor="black",
    )

    ax.set_xlim(-Lx / 2, Lx / 2)
    ax.set_ylim(-Ly / 2, Ly / 2)
    ax.set_zlim(-Lz / 2, Lz / 2)

    # set equal aspect ratio
    ax.set_box_aspect(
        (
            (Lx) / (Ly),
            1,
            (Lz) / (Ly),
        )
    )

    ax.set_zticks([])

    # x axis label
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Tighten layout to reduce whitespace
    plt.subplots_adjust(
        left=0.05,
        right=0.95,  # Adjust as needed
        bottom=0.05,
        top=0.95,  # Adjust as needed
        wspace=0.1,
        hspace=0.1,
    )

    if save_path:
        # avoid showing the plot in notebook
        plt.savefig(save_path)
        plt.close(fig)
