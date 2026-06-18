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
