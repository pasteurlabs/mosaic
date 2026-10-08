# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = (
    Path(__file__).parents[1] / "mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate"
)


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


storage = load("operator_storage")
model = load("operator_model")
load("operator_dataset")
loader = load("operator_loader")


def test_nested_bind_mount_is_detected():
    mounts = "1 0 0:1 / / rw - ext4 /dev/root rw\n2 1 0:2 /repo /work/source rw - nfs4 host:/data rw\n"
    assert (
        storage.filesystem_info(Path("/work/source/code.py"), mounts)["filesystem"]
        == "nfs4"
    )
    assert storage.filesystem_info(Path("/work/output"), mounts)["filesystem"] == "ext4"


def test_working_storage_rejects_network_mount(monkeypatch):
    monkeypatch.setattr(storage, "filesystem_info", lambda p: {"filesystem": "nfs4"})
    with pytest.raises(ValueError, match="node-local"):
        storage.require_local(Path("/data/example"))


def test_bundle_roundtrip_and_corruption(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"weights" * 50)
    (source / "metrics.json").write_text('{"step": 3}')
    archive = tmp_path / "published" / "run.tar"
    record = storage.publish_bundle(source, archive)
    destination = tmp_path / "restored"
    assert storage.restore_bundle(archive, destination) == record
    assert (destination / "weights.bin").read_bytes() == (
        source / "weights.bin"
    ).read_bytes()
    with archive.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        storage.restore_bundle(archive, tmp_path / "bad")


@pytest.mark.parametrize("n", [8, 16, 20, 32, 48, 64])
def test_shared_spectral_bank_preserves_low_modes_at_every_resolution(n):
    x = np.arange(n) * 2 * np.pi / n
    field = np.broadcast_to(np.sin(3 * x)[:, None, None], (n, n, n))[None, ..., None]
    bank = jnp.ones((13, 13, 7, 1, 1))
    actual = model.spectral_convolution(
        jnp.asarray(field, dtype=jnp.float32), bank, jnp.zeros_like(bank)
    )
    np.testing.assert_allclose(actual, field, atol=1e-6)


def test_small_grid_nyquist_is_not_overwritten_by_overlapping_mode_slices():
    field = np.broadcast_to((-1.0) ** np.arange(8)[:, None, None], (8, 8, 8))[
        None, ..., None
    ]
    bank = jnp.ones((13, 13, 7, 1, 1))
    actual = model.spectral_convolution(
        jnp.asarray(field, dtype=jnp.float32), bank, jnp.zeros_like(bank)
    )
    np.testing.assert_allclose(actual, 0, atol=1e-6)


def test_physical_diffusion_is_resolution_independent():
    for n in [8, 16, 32]:
        x = np.arange(n) * 2 * np.pi / n
        field = np.zeros((1, n, n, n, 3), np.float32)
        field[..., 1] = np.sin(2 * x)[None, :, None, None]
        actual = model.diffuse(
            jnp.asarray(field),
            jnp.float32(0.1),
            jnp.float32(0.5),
            jnp.float32(2 * np.pi),
        )
        np.testing.assert_allclose(actual, field * np.exp(-0.2), atol=1e-6)


def test_projection_is_idempotent_with_nyquist_content():
    field = jnp.asarray(
        np.random.default_rng(3).normal(size=(1, 8, 8, 8, 3)), dtype=jnp.float32
    )
    projected = model.project(field)
    np.testing.assert_allclose(model.project(projected), projected, atol=1e-6)


def test_zero_state_and_arbitrary_horizon_remainder():
    params = model.init_params(width=2, modes=2, layers=1)
    initial = jnp.zeros((1, 8, 8, 8, 3), jnp.float32)
    args = (jnp.float32(0.01), jnp.float32(0.02), jnp.float32(2 * np.pi))
    np.testing.assert_array_equal(
        model.predict(params, initial, *args, steps=7, stride=5, layers=1), initial
    )
    initial = initial.at[:, 1, 2, 3, 0].set(0.1)
    expected = model.one_step(params, initial, *args, span=5, layers=1)
    expected = model.one_step(params, expected, *args, span=2, layers=1)
    np.testing.assert_allclose(
        model.predict(params, initial, *args, steps=7, stride=5, layers=1),
        expected,
        atol=1e-6,
    )


def test_native_input_gradient_matches_directional_finite_difference():
    params = model.init_params(width=2, modes=2, layers=1)
    rng = np.random.default_rng(7)
    value = jnp.asarray(rng.normal(0, 0.1, (1, 8, 8, 8, 3)), dtype=jnp.float32)
    direction = jnp.asarray(rng.normal(size=value.shape), dtype=jnp.float32)
    direction /= jnp.linalg.norm(direction)

    def objective(x):
        return jnp.sum(
            model.one_step(
                params,
                x,
                jnp.float32(0.01),
                jnp.float32(0.02),
                jnp.float32(2 * np.pi),
                layers=1,
            )
            ** 2
        )

    actual = jnp.vdot(jax.grad(objective)(value), direction)
    expected = (
        objective(value + 0.01 * direction) - objective(value - 0.01 * direction)
    ) / 0.02
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=2e-4)


def test_loader_respects_sparse_times_and_parent_splits(tmp_path, monkeypatch):
    data = tmp_path / "data" / "case" / "float64"
    data.mkdir(parents=True)
    steps = np.array([0, 1, 2, 10, 11, 12])
    velocity = np.broadcast_to(
        steps[None, :, None, None, None, None], (3, 6, 8, 8, 8, 3)
    ).astype(np.float32)
    np.savez(
        data / "shard-00000.npz",
        velocity=velocity,
        snapshot_steps=steps,
        valid=[True, True, False],
        split=["train", "validation", "train"],
        parent_id=["a", "b", "c"],
    )
    (data / "shard-00000.json").write_text(
        json.dumps(
            {
                "signature": "sig",
                "sha256": "sha",
                "start": 0,
                "stop": 3,
                "case": {"id": "case", "N": 8},
            }
        )
    )
    dataset = loader.WindowDataset(tmp_path / "data", tmp_path / "cache")
    assert len(dataset.choices("case", "train", 2, 1)) == 1
    assert len(dataset.choices("case", "validation", 2, 1)) == 1
    assert not dataset.choices("case", "train", 3, 1)
    initial, target = dataset.sample("case", "train", 4, 2, 1, np.random.default_rng(9))
    np.testing.assert_array_equal(target[:, 0] - initial, np.ones_like(initial))
    np.testing.assert_array_equal(target[:, 1] - initial, 2 * np.ones_like(initial))
    assert dataset.endpoints("case", "validation")[2] == ["b"]
    # Reopening uses direct read-only mmap; no ZIP parsing or extracted copy.
    assert not list((tmp_path / "cache").glob("*.npy"))

    def unexpected_zip(*args, **kwargs):
        raise AssertionError("cached loader reparsed ZIP")

    monkeypatch.setattr(loader.zipfile, "ZipFile", unexpected_zip)
    reopened = loader.WindowDataset(tmp_path / "data", tmp_path / "cache")
    assert reopened.sample("case", "train", 1, 1, 1, np.random.default_rng(1))[
        0
    ].shape == (1, 8, 8, 8, 3)


def test_training_loss_normalization_agrees_across_grids():
    training = load("train_operator")
    losses = []
    for n in (8, 16, 32):
        axis = jnp.arange(n) * 2 * jnp.pi / n
        target = jnp.ones((1, 2, n, n, n, 3), jnp.float32) * 0.1
        error = jnp.sin(2 * axis)[None, None, :, None, None, None] * 0.01
        losses.append(float(training.trajectory_loss(target + error, target)))
    np.testing.assert_allclose(losses, losses[0], rtol=1e-5)


def test_training_update_and_optimizer_checkpoint_roundtrip(tmp_path):
    from types import SimpleNamespace

    training = load("train_operator")
    args = SimpleNamespace(layers=1, project_output=False, lr=0.001, updates=10)
    params = model.init_params(width=2, modes=2, layers=1)
    moments = jax.tree.map(jnp.zeros_like, params)
    initial = jnp.asarray(
        np.random.default_rng(1).normal(0, 0.1, (1, 8, 8, 8, 3)), dtype=jnp.float32
    )
    physics = (jnp.float32(0.01), jnp.float32(0.02), jnp.float32(2 * np.pi))
    target = model.diffuse(initial, *[physics[0], physics[1], physics[2]])[:, None]
    update = training.make_update(args, 1, 1)
    params, first, second, loss, norm, finite = update(
        params, moments, moments, initial, target, *physics, jnp.float32(1)
    )
    assert bool(finite) and np.isfinite(float(loss)) and float(norm) > 0
    path = tmp_path / "checkpoint.npz"
    training.save_checkpoint(
        path, params, first, second, {"step": 1, "rng": {"state": 4}}
    )
    restored = training.load_checkpoint(path)
    assert restored[3] == {"step": 1, "rng": {"state": 4}}
    next_a = update(params, first, second, initial, target, *physics, jnp.float32(2))
    next_b = update(*restored[:3], initial, target, *physics, jnp.float32(2))
    for a, b in zip(jax.tree.leaves(next_a), jax.tree.leaves(next_b), strict=True):
        np.testing.assert_array_equal(a, b)


def test_training_cli_smoke_and_completed_resume(tmp_path):
    import subprocess

    data = tmp_path / "data" / "case" / "float64"
    data.mkdir(parents=True)
    x = np.arange(8) * 2 * np.pi / 8
    initial = np.zeros((3, 8, 8, 8, 3), np.float32)
    initial[..., 1] = 0.1 * np.sin(x)[None, :, None, None]
    times = np.arange(41)
    velocity = (
        initial[:, None] * np.exp(-0.01 * 0.02 * times)[None, :, None, None, None, None]
    )
    np.savez(
        data / "shard-00000.npz",
        velocity=velocity.astype(np.float32),
        valid=[True] * 3,
        split=["train", "validation", "test"],
        parent_id=["a", "b", "c"],
        snapshot_steps=times,
    )
    (data / "shard-00000.json").write_text(
        json.dumps(
            {
                "signature": "sig",
                "sha256": "sha",
                "start": 0,
                "stop": 3,
                "case": {
                    "id": "case",
                    "N": 8,
                    "steps": 40,
                    "viscosity": 0.01,
                    "dt": 0.02,
                    "domain_extent": 2 * np.pi,
                },
            }
        )
    )
    output = tmp_path / "output"
    manifest = tmp_path / "manifest.json"
    case = json.loads((data / "shard-00000.json").read_text())["case"]
    manifest.write_text(json.dumps({"cases": [case]}))
    command = [
        sys.executable,
        str(ROOT / "train_operator.py"),
        "--dataset",
        str(tmp_path / "data"),
        "--manifest",
        str(manifest),
        "--data-cache",
        str(tmp_path / "cache"),
        "--output",
        str(output),
        "--width",
        "2",
        "--modes",
        "2",
        "--layers",
        "1",
        "--updates",
        "4",
        "--validation-interval",
        "2",
        "--validation-samples",
        "1",
        "--batch-size",
        "1",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads((output / "report.json").read_text())
    assert report["steps_completed"] == 4 and not report["interrupted"]
    assert report["test"][0]["finite"]
    subprocess.run(
        [*command, "--resume", str(output / "latest.npz")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads((output / "report.json").read_text()) == report
    manifest.write_text(json.dumps({"cases": []}))
    invalid = subprocess.run(command, check=False, capture_output=True, text=True)
    assert invalid.returncode != 0
    assert "dataset cases do not match" in invalid.stderr


def test_endpoint_validation_balances_families_before_repeating():
    dataset = object.__new__(loader.WindowDataset)
    dataset.shards = [
        {
            "case": {"id": "case"},
            "valid": [True] * 6,
            "split": ["validation"] * 5 + ["test"],
            "parent_id": ["z", "y", "x", "b", "a", "t"],
            "family": ["near_zero"] * 3 + ["tgv", "abc", "random"],
        }
    ]
    dataset.arrays = [np.zeros((6, 2, 8, 8, 8, 3), np.float32)]
    parents = dataset.endpoints("case", "validation", limit=3)[2]
    assert parents == ["a", "x", "b"]
    assert "t" not in dataset.endpoints("case", "validation", limit=99)[2]


def test_energy_guard_preserves_mean_and_bounds_rollout():
    params = model.init_params(width=2, modes=2, layers=1)
    params["out"] = jnp.ones_like(params["out"]) * 2
    u = (
        jnp.asarray(
            np.random.default_rng(13).normal(0.0, 0.1, (1, 8, 8, 8, 3)), jnp.float32
        )
        + 0.3
    )
    physics = (jnp.float32(0.001), jnp.float32(0.05), jnp.float32(2 * np.pi))
    result = model.predict(
        params, u, *physics, steps=100, stride=5, layers=1, conserve_energy=True
    )
    assert np.isfinite(result).all()
    np.testing.assert_allclose(
        np.mean(result, axis=(1, 2, 3)), np.mean(u, axis=(1, 2, 3)), atol=2e-6
    )
    assert float(jnp.mean(result**2)) <= float(jnp.mean(u**2)) + 1e-6
    derivative = jax.grad(
        lambda v: jnp.sum(
            model.one_step(params, v, *physics, layers=1, conserve_energy=True)
        )
    )(jnp.zeros_like(u))
    assert np.isfinite(derivative).all()
    np.testing.assert_allclose(derivative, 1.0, atol=1e-5)


def test_partial_dataset_fails_before_mapping(tmp_path):
    directory = tmp_path / "data/case/float64"
    directory.mkdir(parents=True)
    (directory / "shard-00000.json").write_text(
        json.dumps(
            {
                "case": {"id": "case", "N": 8, "samples": 4},
                "start": 0,
                "stop": 3,
                "signature": "sig",
                "sha256": "sha",
            }
        )
    )
    with pytest.raises(ValueError, match="incomplete dataset case"):
        loader.WindowDataset(tmp_path / "data", tmp_path / "cache")
