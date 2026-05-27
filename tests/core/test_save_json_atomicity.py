# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for save_json atomic write semantics.

save_json serialises to a string first, then writes via .tmp + os.replace.
If the encoder raises, the destination must be left untouched. These tests
verify that invariant and the basic happy-path behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic.benchmarks.core.io import save_json


class TestSaveJsonAtomicity:
    def test_successful_write(self, tmp_path: Path):
        path = tmp_path / "result.json"
        save_json({"status": "ok", "mean_s": 1.5}, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == {"status": "ok", "mean_s": 1.5}

    def test_no_tmp_left_on_success(self, tmp_path: Path):
        path = tmp_path / "result.json"
        save_json({"x": 1}, path)
        tmp_file = path.with_suffix(".json.tmp")
        assert not tmp_file.exists()

    def test_encoder_failure_preserves_existing(self, tmp_path: Path):
        path = tmp_path / "result.json"
        # Write a valid file first.
        save_json({"old": "data"}, path)
        original_content = path.read_text()

        # An object that no encoder can handle (not numpy, not callable, etc.).
        class Unencodable:
            pass

        with pytest.raises(TypeError):
            save_json({"bad": Unencodable()}, path)

        # Original file must be intact.
        assert path.read_text() == original_content

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "a" / "b" / "c" / "result.json"
        save_json({"nested": True}, path)
        assert path.exists()

    def test_overwrite_replaces_content(self, tmp_path: Path):
        path = tmp_path / "result.json"
        save_json({"version": 1}, path)
        save_json({"version": 2}, path)
        data = json.loads(path.read_text())
        assert data == {"version": 2}
