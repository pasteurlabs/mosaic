# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AST-normalised ``harness_fn_hash``.

Run directly with::

    conda run -n gym python -m benchmarks.core.tests.test_harness_hash

The previous implementation hashed ``inspect.getsource(fn)`` byte-for-byte.
Any whitespace, comment, or docstring-only edit to a ``run_*`` harness flipped
the hash and marked every previously-saved result as stale. A kwarg-threading
refactor flipped ~every forward/gradient/cost/recovery cell stale with zero
behavioural change.

The AST-normalised hash strips docstrings and ignores comments / whitespace
entirely (comments are discarded by ``ast.parse``). Behavioural edits — new
statements, renamed locals, reordered expressions — still flip the hash.

The tests drive ``harness_fn_hash`` through a mocked ``inspect.getsource`` so
that the *only* variable across each test pair is the property we're
exercising (docstring, comment, whitespace, body, identifier). Using two
real nested functions would also differ on ``FunctionDef.name`` and on the
qualname path, defeating the comparison.
"""

from __future__ import annotations

import hashlib
import unittest
from unittest import mock

from mosaic.benchmarks.core.io import harness_fn_hash


def _dummy_fn():
    return 0


def _hash_for_source(src: str) -> str:
    """Run ``harness_fn_hash`` against *src* via a mocked ``inspect.getsource``."""
    with mock.patch("mosaic.benchmarks.core.io.inspect.getsource", return_value=src):
        return harness_fn_hash(_dummy_fn)


class HarnessFnHashTests(unittest.TestCase):
    # ── invariants: semantically-equivalent edits should NOT flip hash ────────

    def test_docstring_only_diff_same_hash(self) -> None:
        """Two function sources differing only in their docstring hash identically."""
        src_a = "def run():\n    '''Original docstring.'''\n    x = 1\n    return x\n"
        src_b = (
            "def run():\n"
            "    '''A completely different docstring that says other things.'''\n"
            "    x = 1\n"
            "    return x\n"
        )
        self.assertEqual(_hash_for_source(src_a), _hash_for_source(src_b))

    def test_comment_only_diff_same_hash(self) -> None:
        """Comment-only edits must not flip the hash (comments are stripped by ast.parse)."""
        src_a = "def run():\n    # original comment\n    x = 1\n    return x\n"
        src_b = (
            "def run():\n"
            "    # a wholly different comment explaining nothing\n"
            "    x = 1\n"
            "    return x  # trailing comment too\n"
        )
        self.assertEqual(_hash_for_source(src_a), _hash_for_source(src_b))

    def test_whitespace_only_diff_same_hash(self) -> None:
        """Blank-line edits (the most common refactor side-effect) must not flip the hash."""
        src_a = "def run():\n    x = 1\n    return x\n"
        src_b = "def run():\n\n    x = 1\n\n\n    return x\n"
        self.assertEqual(_hash_for_source(src_a), _hash_for_source(src_b))

    def test_no_docstring_does_not_raise(self) -> None:
        """Functions with no docstring must hash cleanly (strip is a no-op)."""
        src = "def run():\n    x = 1\n    return x\n"
        h = _hash_for_source(src)
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 16)

    # ── sensitivity: behavioural edits SHOULD flip hash ───────────────────────

    def test_body_change_different_hash(self) -> None:
        """Adding a statement to the body must flip the hash."""
        src_a = "def run():\n    x = 1\n    return x\n"
        src_b = "def run():\n    x = 1\n    x = x + 1\n    return x\n"
        self.assertNotEqual(_hash_for_source(src_a), _hash_for_source(src_b))

    def test_variable_rename_different_hash(self) -> None:
        """Renaming a local variable must flip the hash.

        ast.dump preserves identifier names, so ``x`` vs ``y`` is visible. This
        guards against semantic drift that might go unnoticed if we
        over-normalised (e.g. by alpha-renaming locals).
        """
        src_a = "def run():\n    x = 1\n    return x\n"
        src_b = "def run():\n    y = 1\n    return y\n"
        self.assertNotEqual(_hash_for_source(src_a), _hash_for_source(src_b))

    # ── safety fallback: SyntaxError path ─────────────────────────────────────

    def test_syntax_error_falls_back_to_raw_source_hash(self) -> None:
        """When ast.parse raises SyntaxError on the source, fall back to the
        raw-source SHA (first 16 hex chars) rather than returning an empty
        string or raising. Ensures decorator-generated closures whose
        ``inspect.getsource`` returns partial text still get a fingerprint.
        """
        bogus = "def broken(:\n    pass\n"
        h = _hash_for_source(bogus)
        expected = hashlib.sha256(bogus.encode("utf-8")).hexdigest()[:16]
        self.assertEqual(h, expected)

    # ── additional sanity: OSError / TypeError paths stay intact ──────────────

    def test_oserror_returns_empty_string(self) -> None:
        """``inspect.getsource`` may raise OSError for C-implemented callables
        (e.g. built-ins, lambdas in some REPLs). Preserve the legacy empty-string
        return so callers' existing empty-hash guards still fire.
        """
        with mock.patch(
            "mosaic.benchmarks.core.io.inspect.getsource",
            side_effect=OSError("no source"),
        ):
            self.assertEqual(harness_fn_hash(_dummy_fn), "")


if __name__ == "__main__":
    unittest.main()
