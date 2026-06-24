# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for failure/traceback truncation.

A forwarded Tesseract "Error 500" message is the *remote* traceback: the real
exception sits at the very bottom, below a tall stack of FastAPI/starlette
middleware frames. Clipping the head (the old ``msg[:limit]``) dropped the
actual cause. ``_truncate_error`` keeps both ends so the status report still
shows what went wrong.
"""

from __future__ import annotations

import unittest

from mosaic.benchmarks.core.experiment import MAX_ERROR_LEN, _truncate_error


class TestTruncateError(unittest.TestCase):
    def test_short_message_unchanged(self) -> None:
        self.assertEqual(_truncate_error("IndexError: boom"), "IndexError: boom")

    def test_message_at_limit_unchanged(self) -> None:
        msg = "x" * MAX_ERROR_LEN
        self.assertEqual(_truncate_error(msg), msg)

    def test_long_traceback_keeps_head_and_real_cause(self) -> None:
        head = (
            "RuntimeError: Error 500 from Tesseract: "
            "Traceback (most recent call last):\n"
        )
        cause = "IndexError: Array slice indices must have static start/stop/step"
        msg = head + ("  File frame\n" * 5000) + cause

        out = _truncate_error(msg)

        self.assertLessEqual(len(out), MAX_ERROR_LEN)
        # Head (error type + entry point) survives.
        self.assertTrue(out.startswith("RuntimeError: Error 500"))
        # The real cause at the tail survives — this is the regression guard.
        self.assertTrue(out.rstrip().endswith(cause))
        self.assertIn("traceback truncated", out)

    def test_respects_custom_limit(self) -> None:
        out = _truncate_error("a" * 100 + "TAIL", limit=40)
        self.assertLessEqual(len(out), 40)
        self.assertTrue(out.rstrip().endswith("TAIL"))


if __name__ == "__main__":
    unittest.main()
