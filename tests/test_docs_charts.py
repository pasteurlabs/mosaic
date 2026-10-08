# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for docs/vl: the Vega-Lite chart builders behind the results pages."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("altair")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs"))

from vl import builders
from vl.common import field_cells, log_axis

# Every PNG stem the results pages show (docs/generate_results.py).
EXPECTED = {
    "agreement",
    "best_eps_vs_param",
    "conductivity_recovery_convergence",
    "cost",
    "curves",
    "drag_opt",
    "drag_opt_fields",
    "fd_check",
    "fields_raw",
    "gradient_fields",
    "horizon_sweep",
    "horizon_sweep_limits",
    "ics",
    "jacobian_svd_comparison",
    "physical_accuracy",
    "topopt",
    "topopt_3d",
    "topopt_fields",
    # Animations (GIFs)
    "drag_opt_evolution",
    "recovery_evolution",
    "topopt_evolution_FEniCS",
}


def test_every_plot_has_a_builder():
    assert set(builders()) >= EXPECTED


def test_log_axis_labels_decades():
    axis = log_axis("ε", lo=-2, hi=1).to_dict()
    assert axis["values"] == [0.01, 0.1, 1.0, 10.0]
    assert "'10⁻²'" in axis["labelExpr"]


def test_field_cells_orientation_and_subsampling():
    import numpy as np

    arr = np.arange(6.0).reshape(2, 3)  # first index → x (i), second → y (j)
    df = field_cells(arr)
    assert df.loc[(df.i == 1) & (df.j == 2), "value"].item() == 5.0
    big = field_cells(np.zeros((256, 128)))
    assert big["i"].max() + 1 <= 64
