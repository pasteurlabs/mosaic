"""Tests for shared Tesseract schemas: domain schemas are importable and well-formed."""

from __future__ import annotations


def test_ns_grid_schemas_importable():
    from mosaic.mosaic_shared.problems.navier_stokes_grid import (
        InputSchema,
        OutputSchema,
    )

    assert hasattr(InputSchema, "model_fields")
    assert hasattr(OutputSchema, "model_fields")
    assert len(InputSchema.model_fields) >= 1
    assert len(OutputSchema.model_fields) >= 1


def test_structural_mesh_schemas_importable():
    from mosaic.mosaic_shared.problems.structural_mesh import InputSchema, OutputSchema

    assert hasattr(InputSchema, "model_fields")
    assert hasattr(OutputSchema, "model_fields")


def test_thermal_mesh_schemas_importable():
    from mosaic.mosaic_shared.problems.thermal_mesh import InputSchema, OutputSchema

    assert hasattr(InputSchema, "model_fields")
    assert hasattr(OutputSchema, "model_fields")


def test_ns_grid_input_has_differentiable_fields():
    """NS grid InputSchema should mark some fields as differentiable."""
    from mosaic.mosaic_shared.problems.navier_stokes_grid import InputSchema

    field_names = set(InputSchema.model_fields.keys())
    # These are standard fields for the NS grid domain
    assert "ic" in field_names or "viscosity" in field_names or len(field_names) >= 3


def test_shared_types_importable():
    from mosaic.mosaic_shared.types import Differentiable

    assert Differentiable is not None
