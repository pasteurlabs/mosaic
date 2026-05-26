# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmarks.core.utils functions."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from mosaic.benchmarks.core.io import (
    load_json,
    save_json,
    tesseract_content_hash,
)
from mosaic.benchmarks.core.utils import (
    is_valid,
    l2_error_rel,
)


class TestL2ErrorRel:
    def test_identical_arrays(self):
        a = jnp.array([1.0, 2.0, 3.0])
        assert l2_error_rel(a, a) == pytest.approx(0.0, abs=1e-7)

    def test_known_error(self):
        ref = jnp.array([1.0, 0.0, 0.0])
        pred = jnp.array([0.0, 0.0, 0.0])
        assert l2_error_rel(pred, ref) == pytest.approx(1.0)

    def test_scaled_difference(self):
        ref = jnp.array([2.0, 0.0])
        pred = jnp.array([3.0, 0.0])
        # error = 1/2 = 0.5
        assert l2_error_rel(pred, ref) == pytest.approx(0.5)


class TestIsValid:
    def test_valid_array(self):
        assert is_valid(jnp.array([1.0, 2.0, 3.0]))

    def test_nan_is_invalid(self):
        assert not is_valid(jnp.array([1.0, float("nan"), 3.0]))

    def test_inf_is_invalid(self):
        assert not is_valid(jnp.array([1.0, float("inf"), 3.0]))

    def test_empty_is_valid(self):
        assert is_valid(jnp.array([]))


class TestSaveLoadJson:
    def test_roundtrip(self, tmp_path):
        data = {"key": "value", "number": 42, "nested": {"a": [1, 2, 3]}}
        path = tmp_path / "test.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data


class TestTesseractContentHash:
    def test_deterministic(self, tmp_path):
        """Same content should produce the same hash."""
        d = tmp_path / "solver"
        d.mkdir()
        (d / "tesseract_api.py").write_text("print('hello')")
        (d / "config.yaml").write_text("name: test")

        h1 = tesseract_content_hash(d)
        h2 = tesseract_content_hash(d)
        assert h1 == h2

    def test_different_content(self, tmp_path):
        """Different content should produce different hashes."""
        d1 = tmp_path / "s1"
        d1.mkdir()
        (d1 / "tesseract_api.py").write_text("print('a')")

        d2 = tmp_path / "s2"
        d2.mkdir()
        (d2 / "tesseract_api.py").write_text("print('b')")

        assert tesseract_content_hash(d1) != tesseract_content_hash(d2)

    def test_ignores_pycache(self, tmp_path):
        """__pycache__ should not affect the hash."""
        d = tmp_path / "solver"
        d.mkdir()
        (d / "tesseract_api.py").write_text("print('hello')")

        h1 = tesseract_content_hash(d)

        cache = d / "__pycache__"
        cache.mkdir()
        (cache / "foo.pyc").write_bytes(b"\x00\x01\x02")

        h2 = tesseract_content_hash(d)
        assert h1 == h2
