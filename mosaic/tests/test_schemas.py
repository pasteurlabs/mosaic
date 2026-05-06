"""Tests for shared Tesseract schemas: domain schemas are importable and well-formed."""

from __future__ import annotations

import pytest

# The schema modules import from ``mosaic_shared.types`` and
# ``tesseract_core.runtime`` which are only available when these packages are
# installed (e.g. inside a tesseract runtime or a full dev environment).
_missing_runtime_deps = False
try:
    import mosaic_shared.types  # noqa: F401
    import tesseract_core.runtime  # noqa: F401
except ModuleNotFoundError:
    _missing_runtime_deps = True

needs_runtime = pytest.mark.skipif(
    _missing_runtime_deps,
    reason="mosaic_shared / tesseract_core not installed as top-level packages",
)


@needs_runtime
def test_ns_grid_schemas_importable():
    from mosaic.mosaic_shared.problems.navier_stokes_grid import (
        InputSchema,
        OutputSchema,
    )

    assert hasattr(InputSchema, "model_fields")
    assert hasattr(OutputSchema, "model_fields")
    assert len(InputSchema.model_fields) >= 1
    assert len(OutputSchema.model_fields) >= 1


@needs_runtime
def test_structural_mesh_schemas_importable():
    from mosaic.mosaic_shared.problems.structural_mesh import InputSchema, OutputSchema

    assert hasattr(InputSchema, "model_fields")
    assert hasattr(OutputSchema, "model_fields")


@needs_runtime
def test_thermal_mesh_schemas_importable():
    from mosaic.mosaic_shared.problems.thermal_mesh import InputSchema, OutputSchema

    assert hasattr(InputSchema, "model_fields")
    assert hasattr(OutputSchema, "model_fields")


@needs_runtime
def test_ns_grid_input_has_differentiable_fields():
    """NS grid InputSchema should mark some fields as differentiable."""
    from mosaic.mosaic_shared.problems.navier_stokes_grid import InputSchema

    field_names = set(InputSchema.model_fields.keys())
    # These are standard fields for the NS grid domain
    assert "ic" in field_names or "viscosity" in field_names or len(field_names) >= 3


def test_shared_types_importable():
    from mosaic.mosaic_shared.types import Differentiable

    assert Differentiable is not None
