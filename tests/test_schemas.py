# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared Tesseract schemas: domain schemas are importable and well-formed."""

from __future__ import annotations

import pytest

# The schema modules import from ``mosaic_shared.schema_types`` and
# ``tesseract_core.runtime`` which are only available when these packages are
# installed (e.g. inside a tesseract runtime or a full dev environment).
_missing_runtime_deps = False
try:
    import mosaic_shared.schema_types  # noqa: F401
    import tesseract_core.runtime  # noqa: F401
except ModuleNotFoundError:
    _missing_runtime_deps = True

needs_runtime = pytest.mark.skipif(
    _missing_runtime_deps,
    reason="mosaic_shared / tesseract_core not installed as top-level packages",
)


@needs_runtime
@pytest.mark.parametrize(
    "module",
    [
        "mosaic.mosaic_shared.problems.navier_stokes_grid",
        "mosaic.mosaic_shared.problems.structural_mesh",
        "mosaic.mosaic_shared.problems.thermal_mesh",
    ],
)
def test_domain_schemas_define_input_and_output(module):
    """Each problem-schema module must expose populated InputSchema /
    OutputSchema. Catches a broken import or an accidentally emptied schema
    (pydantic happily accepts a zero-field model).
    """
    import importlib

    mod = importlib.import_module(module)
    for name in ("InputSchema", "OutputSchema"):
        cls = getattr(mod, name)
        assert cls.model_fields, f"{module}.{name} has no fields"


@needs_runtime
def test_ns_grid_input_declares_canonical_fields():
    """Catches a regression where the NS grid InputSchema loses its standard
    fields. The full canonical set was once silently emptied by a refactor.
    """
    from mosaic.mosaic_shared.problems.navier_stokes_grid import (
        InputSchema,
    )

    field_names = set(InputSchema.model_fields.keys())
    expected_subset = {"v0", "viscosity", "dt", "steps"}
    missing = expected_subset - field_names
    assert not missing, f"NS grid InputSchema lost canonical fields: {missing}"


_CHANNEL = {
    "x_lo": {"type": "dirichlet"},
    "x_hi": {"type": "neumann"},
    "y_lo": {"type": "no_slip"},
    "y_hi": {"type": "no_slip"},
}
_CYLINDER = {"shape": "cylinder", "center": [0.2, 0.2], "radius": 0.05}


def _channel_inputs(**overrides):
    import numpy as np

    inputs = {
        "v0": np.zeros((220, 41, 1, 2), dtype=np.float32),
        "domain_extent": [2.2, 0.41],
        "boundary_conditions": _CHANNEL,
        "obstacle": _CYLINDER,
        "inflow_profile": np.ones(41, dtype=np.float32),
    }
    return {**inputs, **overrides}


@needs_runtime
def test_ns_grid_scalar_extent_is_a_square():
    """Periodic callers keep sending a scalar extent, unchanged."""
    import numpy as np
    from mosaic_shared.problems.navier_stokes_grid import InputSchema

    s = InputSchema(v0=np.zeros((16, 16, 1, 2), dtype=np.float32), domain_extent=1.0)
    assert s.extent == (1.0, 1.0)
    assert s.dx == 1.0 / 16


@needs_runtime
def test_ns_grid_periodic_box_rejects_extent_list():
    import numpy as np
    from mosaic_shared.problems.navier_stokes_grid import InputSchema
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="scalar domain_extent"):
        InputSchema(
            v0=np.zeros((16, 16, 1, 2), dtype=np.float32), domain_extent=[1.0, 1.0]
        )


@needs_runtime
def test_ns_grid_rectangular_channel_validates():
    from mosaic_shared.problems.navier_stokes_grid import InputSchema

    s = InputSchema(**_channel_inputs())
    assert s.extent == (2.2, 0.41)
    assert s.dx == 0.01


@needs_runtime
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"domain_extent": 2.2}, "isotropic"),
        ({"domain_extent": [2.2, 0.5]}, "isotropic"),
        ({"domain_extent": [2.2, 0.41, 1.0]}, "entries"),
        ({"obstacle": None}, "require an obstacle"),
        ({"boundary_conditions": {**_CHANNEL, "x_lo": {"type": "no_slip"}}}, "x_lo"),
        ({"boundary_conditions": {**_CHANNEL, "x_hi": {"type": "periodic"}}}, "both"),
        ({"obstacle": {"shape": "cylinder", "center": [0.2, 0.2]}}, "radius"),
    ],
)
def test_ns_grid_schema_rejects_inconsistent_inputs(overrides, match):
    from mosaic_shared.problems.navier_stokes_grid import InputSchema
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=match):
        InputSchema(**_channel_inputs(**overrides))


@needs_runtime
def test_ns_grid_inflow_profile_must_match_the_grid():
    import numpy as np
    from mosaic_shared.problems.navier_stokes_grid import InputSchema
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=r"\(41,\)"):
        InputSchema(**_channel_inputs(inflow_profile=np.ones(40, dtype=np.float32)))
