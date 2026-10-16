# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the tesseract tests that need Docker or the DFG 2D-1 channel."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from mosaic.benchmarks.problems.navier_stokes_grid import cylinder


def image_available(image: str) -> bool:
    """True if Docker is installed and ``image`` exists locally."""
    if shutil.which("docker") is None:
        return False
    try:
        return (
            subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                timeout=30,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def channel_inputs(resolution: int, t_end: float = cylinder.T_END) -> dict:
    """Harness inputs for the benchmark inflow profile, as NumPy arrays."""
    inputs = cylinder.channel_inputs(
        cylinder.inflow_profile(resolution), resolution, t_end
    )
    return {k: np.asarray(v) if hasattr(v, "shape") else v for k, v in inputs.items()}


# Input changes that turn the channel into a configuration no solver supports.
unsupported_bcs = pytest.mark.parametrize(
    "change",
    [
        {
            "boundary_conditions": {
                **cylinder.BOUNDARY_CONDITIONS,
                "x_hi": {"type": "no_slip"},
            }
        },
        {
            "boundary_conditions": {
                **cylinder.BOUNDARY_CONDITIONS,
                "y_lo": {"type": "neumann"},
            }
        },
        {
            "v0": np.zeros((41, 41, 1, 2), dtype=np.float32),
            "domain_extent": 0.41,
            "boundary_conditions": {},
            "inflow_profile": None,
        },
    ],
    ids=["outlet_no_slip", "wall_neumann", "periodic_with_obstacle"],
)
