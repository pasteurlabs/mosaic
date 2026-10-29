# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered scalar IC projections survive the shared chart wrapper."""

from types import SimpleNamespace

import numpy as np

from mosaic.benchmarks.problems.buoyancy_grid.config import problem
from mosaic.benchmarks.problems.buoyancy_grid.physics import scalar_field
from mosaic.benchmarks.problems.shared.plots import ics as wrapper
from mosaic.benchmarks.problems.shared.plots.charts import ics as charts


def test_buoyancy_scalar_panels():
    panels = charts._panels(problem, to_2d=scalar_field, field_symmetric=False)
    assert {p[0] for p in panels} == {"plume", "interacting_plumes", "diffusion"}
    for name, field, low, high in panels:
        spec = problem.make_ic[name]
        expected = scalar_field(spec(**spec.plot_params))
        np.testing.assert_array_equal(field, expected)
        assert low == float(expected.min())
        assert high == float(expected.max())


def test_wrapper_preserves_projection_and_palette(monkeypatch, tmp_path):
    captured = {}

    def render(cfg, out, builders, **kwargs):
        captured["spec"] = builders["ics"](
            SimpleNamespace(config=cfg, experiment=None)
        ).to_dict()
        captured["out"] = out

    monkeypatch.setattr(wrapper, "render", render)
    wrapper.plot_ic(
        problem,
        "plume",
        np.zeros((1, 1, 3)),
        tmp_path / "plume",
        ic_to_2d=scalar_field,
        field_cmap="inferno",
        field_symmetric=False,
    )
    spec = captured["spec"]
    assert len(spec["hconcat"]) == 3
    for panel in spec["hconcat"]:
        assert panel["encoding"]["color"]["scale"]["scheme"] == "inferno"
        assert panel["encoding"]["color"]["scale"]["reverse"] is False
    assert captured["out"] == tmp_path


def test_default_projection_still_symmetric():
    cfg = SimpleNamespace(make_ic={"velocity": lambda: np.zeros((4, 4, 1, 2))})
    panels = charts._panels(cfg)
    assert len(panels) == 1
    assert panels[0][2:] == (-1.0, 1.0)
