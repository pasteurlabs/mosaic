# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenFOAM adapter topology, physical mapping, and non-AD API contract."""

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def adapter():
    path = (
        Path(__file__).resolve().parents[1]
        / "mosaic/tesseracts/buoyancy-grid/openfoam/tesseract_api.py"
    )
    spec = importlib.util.spec_from_file_location("buoyancy_openfoam", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mesh_labels(directory):
    mesh = directory / "constant/polyMesh"
    mesh.mkdir(parents=True)
    # One x face has reversed owner/neighbour order, as flux orientation permits.
    (mesh / "owner").write_text("FoamFile {class labelList;}\n4\n(0 3 0 1)\n")
    (mesh / "neighbour").write_text("FoamFile {class labelList;}\n4\n(1 2 2 3)\n")


def test_pressure_flux_mapping_preserves_orientation(adapter, tmp_path):
    _mesh_labels(tmp_path)
    i, j, axis, sign = adapter._face_map(tmp_path, 2)
    np.testing.assert_array_equal(i, [0, 0, 0, 1])
    np.testing.assert_array_equal(j, [0, 1, 0, 0])
    np.testing.assert_array_equal(axis, [0, 0, 1, 1])
    np.testing.assert_array_equal(sign, [1, -1, 1, 1])
    state = np.zeros((2, 2, 3))
    state[:-1, :, 0] = [2, 3]
    state[:, :-1, 1] = [[4], [5]]
    state[..., 2] = [[0.1, 0.2], [0.3, 0.4]]
    (tmp_path / "0").mkdir()
    adapter._write_fields(tmp_path, state, 1.0, (i, j, axis, sign))
    phi = adapter._read_scalar(tmp_path / "0/phi", 4)
    np.testing.assert_allclose(phi, [0.5, -0.75, 1, 1.25])
    np.testing.assert_allclose(phi * sign / 0.25, state[i, j, axis])
    temperature = adapter._read_scalar(tmp_path / "0/T", 4).reshape(2, 2).T
    np.testing.assert_allclose(temperature - adapter._TREF, state[..., 2])


def test_case_matches_buoyancy_and_scalar_diffusivity(adapter, tmp_path):
    nu, kappa, b = 0.002, 0.004, 0.75
    adapter._write_case(tmp_path, 8, 1.0, nu, kappa, b, 0.005, 20)
    properties = (tmp_path / "constant/transportProperties").read_text()
    pr = float(re.search(r"\bPr\s+([^;]+);", properties)[1])
    assert nu / pr == pytest.approx(kappa)
    gravity = (tmp_path / "constant/g").read_text()
    gy = float(re.search(r"value\s*\(0\s+([^ ]+)\s+0\)", gravity)[1])
    assert -adapter._BETA * gy == pytest.approx(b)
    assert (
        "simulationType laminar"
        in (tmp_path / "constant/turbulenceProperties").read_text()
    )
    assert (
        "radiationModel none" in (tmp_path / "constant/radiationProperties").read_text()
    )
    assert "adjustTimeStep no" in (tmp_path / "system/controlDict").read_text()


@pytest.mark.parametrize("key", ["viscosity", "diffusivity"])
def test_zero_transport_coefficient_rejected(adapter, key):
    values = {
        "viscosity": np.array([0.001], np.float32),
        "diffusivity": np.array([0.001], np.float32),
    }
    values[key][0] = 0
    with pytest.raises(ValueError, match="positive viscosity and diffusivity"):
        adapter.InputSchema(**values)


def test_abstract_output_and_forward_only_contract(adapter):
    values = {
        "state0": np.zeros((8, 8, 3), np.float32),
        "diffusivity": np.array([0.001], np.float32),
    }
    assert adapter.abstract_eval(adapter.InputSchema(**values)) == {
        "result": {"shape": (8, 8, 3), "dtype": "float32"}
    }
    assert not hasattr(adapter, "vector_jacobian_product")
    assert not hasattr(adapter, "jacobian_vector_product")


def test_scalar_reader_rejects_wrong_count(adapter, tmp_path):
    path = tmp_path / "T"
    path.write_text("internalField nonuniform List<scalar> 2 (300 301);")
    with pytest.raises(ValueError, match="Invalid scalar field"):
        adapter._read_scalar(path, 3)
