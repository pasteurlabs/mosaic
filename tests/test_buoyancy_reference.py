# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from mosaic.benchmarks.problems.buoyancy_grid import forward

PHYSICS = {
    "N": 16,
    "nu": 0.0005,
    "diffusivity": 0.0005,
    "buoyancy": 1.0,
    "dt": 0.005,
    "steps": 80,
    "domain_extent": 1.0,
}
IC = {"name": "interacting_plumes", "seed": 0}


@pytest.mark.parametrize("n", [16, 32])
def test_shipped_reference_matches_contract(n):
    result = forward.require_openfoam_reference(n, PHYSICS, IC)
    assert result.shape == (n, n, 3)


def test_missing_reference_never_runs_solver(monkeypatch):
    monkeypatch.setattr(forward, "load_reference", lambda *args: None)
    monkeypatch.setattr(
        forward, "shared_agreement", lambda *args: pytest.fail("solver called")
    )
    ctx = SimpleNamespace(phys=PHYSICS, run={"ic": IC})
    with pytest.raises(FileNotFoundError, match="Required OpenFOAM"):
        forward.openfoam_agreement(None, ctx)


def test_missing_archive_is_required(tmp_path, monkeypatch):
    source = forward._REFERENCE_DIR / "forward_agreement.json"
    (tmp_path / source.name).write_bytes(source.read_bytes())
    monkeypatch.setattr(forward, "_REFERENCE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        forward.require_openfoam_reference(16, PHYSICS, IC)


@pytest.mark.parametrize("key,value", [("steps", 81), ("buoyancy", 0), ("nu", 0.001)])
def test_changed_physics_cannot_use_stale_reference(key, value):
    with pytest.raises(ValueError, match="physics mismatch"):
        forward.require_openfoam_reference(16, {**PHYSICS, key: value}, IC)


def test_changed_ic_cannot_use_reference():
    with pytest.raises(ValueError, match="initial condition"):
        forward.require_openfoam_reference(16, PHYSICS, {**IC, "seed": 1})


def test_aggregate_cannot_fall_back_to_consensus(monkeypatch):
    monkeypatch.setattr(forward, "load_reference", lambda *args: None)
    with pytest.raises(FileNotFoundError, match="Required OpenFOAM"):
        forward._openfoam_agreement_aggregate(
            {},
            cfg=SimpleNamespace(domain_extent=1.0),
            run={"physics": {**PHYSICS, "N": [16, 32]}, "ic": IC},
            sweep_values=[16, 32],
        )


@pytest.mark.parametrize("key", ["position_jitter", "width_jitter", "amplitude_jitter"])
def test_changed_plume_jitter_cannot_use_reference(key):
    with pytest.raises(ValueError, match=key):
        forward.require_openfoam_reference(16, {**PHYSICS, key: 0.2}, IC)


def test_changed_factory_default_invalidates_omitted_jitter(monkeypatch):
    from mosaic.benchmarks.problems.buoyancy_grid import physics

    def changed_factory(position_jitter=0.2, width_jitter=0.05, amplitude_jitter=0):
        raise AssertionError("Guard should inspect defaults, not generate an IC")

    monkeypatch.setattr(physics, "interacting_plumes", changed_factory)
    with pytest.raises(ValueError, match="position_jitter"):
        forward.require_openfoam_reference(16, PHYSICS, IC)
