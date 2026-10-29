# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary-condition checks and the staircase cylinder for ns-grid tesseracts.

Every ns-grid solver must raise on boundary conditions it does not implement.
The two configurations the benchmark uses are a fully periodic box and the
2-D DFG channel: profile inlet at x_lo, outflow at x_hi, no-slip walls in y,
and a cylinder. Call one of the checks from the solver's ``InputSchema``::

    @model_validator(mode="after")
    def _check_supported_bcs(self) -> "InputSchema":
        check_periodic_or_channel(self, "my-solver")
        return self
"""

from typing import Any

import numpy as np
from mosaic_shared.schema_types import ObstacleShape

CHANNEL_BCS = {
    "x_lo": "dirichlet",
    "x_hi": "neumann",
    "y_lo": "no_slip",
    "y_hi": "no_slip",
}


def check_periodic(inputs: Any, solver: str) -> None:
    """Raise unless ``inputs`` is a fully periodic box without an obstacle."""
    if inputs.obstacle is not None or not inputs.boundary_conditions.is_fully_periodic:
        raise ValueError(
            f"{solver} supports only fully periodic boundary conditions without "
            "an obstacle"
        )


def check_periodic_or_channel(inputs: Any, solver: str) -> None:
    """Raise unless ``inputs`` is a periodic box or the 2-D channel with a cylinder.

    The periodic box takes no obstacle. The channel needs the faces in
    :data:`CHANNEL_BCS`, a 2-D ``v0``, an ``inflow_profile`` and a cylinder.
    """
    bc = inputs.boundary_conditions
    if bc.is_fully_periodic and inputs.obstacle is None:
        return
    faces = {face: getattr(bc, face).type.value for face in CHANNEL_BCS}
    ndim = inputs.v0.shape[-1]
    if (
        faces != CHANNEL_BCS
        or ndim != 2
        or inputs.inflow_profile is None
        or inputs.obstacle is None
        or inputs.obstacle.shape != ObstacleShape.CYLINDER
    ):
        raise ValueError(
            f"{solver} supports a fully periodic box without an obstacle, or the "
            f"2-D channel {CHANNEL_BCS} with an inflow_profile and a cylinder; "
            f"got {faces}, {ndim}-D, obstacle {inputs.obstacle}"
        )


def solid_cells(nx: int, ny: int, dx: float, center: Any, radius: float) -> np.ndarray:
    """``(nx, ny)`` bool, True for the cells whose center lies strictly inside the cylinder.

    Cell (i, j) is centered at ((i + 1/2) dx, (j + 1/2) dx); ``center`` is the
    cylinder's (x, y) in the same units.
    """
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dx
    cx, cy = center[0], center[1]
    return (xc[:, None] - cx) ** 2 + (yc[None, :] - cy) ** 2 < radius**2
