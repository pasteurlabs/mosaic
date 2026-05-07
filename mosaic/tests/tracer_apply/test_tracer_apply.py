"""Tests for the apply_tesseract wrapper (JAX tracer detection).

Covers:
1. Happy path — underlying apply returns, wrapper is transparent.
2. Exception propagation — underlying exceptions surface unchanged.
3. JAX tracer detection — when inputs carry a tracer, apply_fn must run
   on the caller's thread so primitive binding sees the active trace.
"""

from __future__ import annotations

import threading
import unittest

from mosaic.benchmarks.core import tesseract_apply
from mosaic.benchmarks.core.exceptions import (
    ContainerDied,
    WatchdogError,
    WatchdogTimeout,
)


class TesseractApplyTests(unittest.TestCase):
    def test_happy_path(self) -> None:
        def _fake_apply(_t, inputs):
            return {"result": inputs["x"] * 2}

        out = tesseract_apply.apply_tesseract(
            None, {"x": 21}, _apply_fn=_fake_apply
        )
        self.assertEqual(out, {"result": 42})

    def test_underlying_exception_propagates(self) -> None:
        class _Custom(Exception):
            pass

        def _raising_apply(_t, _inputs):
            raise _Custom("boom")

        with self.assertRaises(_Custom):
            tesseract_apply.apply_tesseract(None, {}, _apply_fn=_raising_apply)

    def test_tracer_input_runs_on_caller_thread(self) -> None:
        """When inputs carry a JAX tracer, apply_fn must run on the
        caller's thread so primitive binding sees the active trace."""
        import jax
        import jax.numpy as jnp

        caller_ident = threading.get_ident()
        exec_idents: list[int] = []

        def _recording_apply(_t, inputs):
            exec_idents.append(threading.get_ident())
            return {"y": inputs["x"] * 2.0}

        def _loss(x):
            out = tesseract_apply.apply_tesseract(
                None, {"x": x}, _apply_fn=_recording_apply
            )
            return jnp.sum(out["y"] ** 2)

        jax.grad(_loss)(jnp.array(3.0))

        self.assertGreaterEqual(len(exec_idents), 1)
        for ident in exec_idents:
            self.assertEqual(
                ident,
                caller_ident,
                "apply_fn must run on caller thread when inputs carry a tracer",
            )

    def test_tracer_detection_false_for_plain_inputs(self) -> None:
        self.assertFalse(tesseract_apply._inputs_contain_tracer({"x": 1.0}))
        self.assertFalse(tesseract_apply._inputs_contain_tracer({}))

    def test_timeout_aliases_are_timeout_error(self) -> None:
        """Backward-compat aliases must be catchable as TimeoutError."""
        self.assertTrue(issubclass(WatchdogTimeout, TimeoutError))
        self.assertTrue(issubclass(ContainerDied, TimeoutError))
        self.assertTrue(issubclass(WatchdogError, TimeoutError))


if __name__ == "__main__":
    unittest.main()
