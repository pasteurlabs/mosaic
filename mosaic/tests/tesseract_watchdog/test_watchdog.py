"""Fake-tesseract test suite for the watchdog.

Run directly with::

    conda run -n gym python -m pytest mosaic/tests/tesseract_watchdog/test_watchdog.py -v

Covers the three scenarios that matter:

1. Happy path — underlying apply returns quickly, watchdog is transparent.
2. Container-death mid-call — watchdog raises :class:`ContainerDied` within
   ~poll+1 s even though the fake apply would block forever.
3. Wall-clock deadline — watchdog raises :class:`WatchdogTimeout` at the
   deadline when container liveness reports are stable.
4. Introspection of :class:`tesseract_core.Tesseract`-style handle with
   ``_serve_context`` — catches any accidental attribute rename before it
   hits a live daemon.

The tests are self-contained: no docker, no tesseract_jax, no network. They
inject a fake ``_apply_fn`` and monkeypatch ``introspect.container_running``.
"""

from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from benchmarks.core.watchdog import (
    ContainerDied,
    WatchdogTimeout,
)
from benchmarks.core import introspect, watchdog


class _FakeServeContextTesseract:
    """Mimics ``tesseract_core.Tesseract`` with ``_serve_context``."""

    def __init__(self, container_name: str) -> None:
        self._serve_context = {
            "container_name": container_name,
            "port": "12345",
            "network": None,
            "network_alias": None,
        }


class IntrospectTests(unittest.TestCase):
    def setUp(self) -> None:
        introspect._reset_caches()

    def test_container_id_from_serve_context(self) -> None:
        t = _FakeServeContextTesseract("tesseract-abc123")
        self.assertEqual(introspect.get_container_id(t), "tesseract-abc123")

    def test_container_id_none_when_missing(self) -> None:
        class Bare:
            pass

        self.assertIsNone(introspect.get_container_id(Bare()))

    def test_container_id_from_private_attr_fallback(self) -> None:
        class Obj:
            _container_id = "fallback-cid"

        self.assertEqual(introspect.get_container_id(Obj()), "fallback-cid")

    def test_container_id_from_container_object(self) -> None:
        class _ContainerObj:
            name = "name-attr"
            id = "id-attr"

        class Obj:
            _container = _ContainerObj()

        self.assertEqual(introspect.get_container_id(Obj()), "name-attr")


class WatchdogTests(unittest.TestCase):
    def setUp(self) -> None:
        introspect._reset_caches()

    def test_happy_path(self) -> None:
        """Underlying apply returns immediately — watchdog transparent."""
        t = _FakeServeContextTesseract("healthy-container")

        def _fake_apply(_t, inputs):
            return {"result": inputs["x"] * 2}

        with mock.patch.object(
            introspect, "container_running", return_value=True
        ):
            out = watchdog._apply_with_watchdog(
                t, {"x": 21}, timeout=10.0, poll=0.5, _apply_fn=_fake_apply
            )
        self.assertEqual(out, {"result": 42})

    def test_container_death_raises_container_died(self) -> None:
        """Fake apply blocks forever; mid-call container_running flips False.

        Watchdog must raise ``ContainerDied`` within poll + 1 s.
        """
        t = _FakeServeContextTesseract("dying-container")

        def _blocking_apply(_t, _inputs):
            # Simulate kernel-level wedge on HTTP read.
            time.sleep(3600)

        # Return True for the first ~1 s (to exercise a couple of poll
        # iterations), then False forever.
        start = time.monotonic()

        def _flaky_running(_cid):
            return (time.monotonic() - start) < 1.0

        with mock.patch.object(
            introspect, "container_running", side_effect=_flaky_running
        ):
            t0 = time.monotonic()
            with self.assertRaises(ContainerDied):
                watchdog._apply_with_watchdog(
                    t,
                    {"x": 1},
                    timeout=120.0,
                    poll=0.5,
                    _apply_fn=_blocking_apply,
                )
            elapsed = time.monotonic() - t0

        # 1 s grace + 0.5 s poll = should be under ~3 s in all cases.
        self.assertLess(
            elapsed,
            3.5,
            f"watchdog took {elapsed:.2f}s to detect container death "
            f"(expected ~1.5 s)",
        )

    def test_wall_clock_deadline_raises_watchdog_timeout(self) -> None:
        """Container stays 'running' but apply blocks past the deadline.

        Watchdog must raise ``WatchdogTimeout`` at ~deadline (not forever).
        """
        t = _FakeServeContextTesseract("healthy-but-slow")

        def _blocking_apply(_t, _inputs):
            time.sleep(3600)

        with mock.patch.object(
            introspect, "container_running", return_value=True
        ):
            t0 = time.monotonic()
            with self.assertRaises(WatchdogTimeout):
                watchdog._apply_with_watchdog(
                    t,
                    {"x": 1},
                    timeout=1.5,
                    poll=0.5,
                    _apply_fn=_blocking_apply,
                )
            elapsed = time.monotonic() - t0

        # Must fire at ~1.5 s deadline; allow generous slack.
        self.assertGreaterEqual(elapsed, 1.3)
        self.assertLess(
            elapsed,
            3.5,
            f"watchdog took {elapsed:.2f}s to fire a 1.5 s deadline",
        )

    def test_watchdog_timeout_is_timeout_error(self) -> None:
        """Code that catches ``TimeoutError`` must still work."""
        t = _FakeServeContextTesseract("healthy-but-slow")

        def _blocking_apply(_t, _inputs):
            time.sleep(3600)

        with mock.patch.object(
            introspect, "container_running", return_value=True
        ):
            with self.assertRaises(TimeoutError):
                watchdog._apply_with_watchdog(
                    t,
                    {"x": 1},
                    timeout=1.0,
                    poll=0.5,
                    _apply_fn=_blocking_apply,
                )

    def test_wall_clock_only_mode_when_no_container_id(self) -> None:
        """Handle with no discoverable container id — still enforces deadline."""

        class NoHandle:
            pass

        def _blocking_apply(_t, _inputs):
            time.sleep(3600)

        # container_running should never be called in wall-clock-only mode.
        with mock.patch.object(
            introspect, "container_running", side_effect=AssertionError
        ):
            t0 = time.monotonic()
            with self.assertRaises(WatchdogTimeout):
                watchdog._apply_with_watchdog(
                    NoHandle(),
                    {"x": 1},
                    timeout=1.0,
                    poll=0.5,
                    _apply_fn=_blocking_apply,
                )
            elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 3.0)

    def test_underlying_exception_propagates(self) -> None:
        """Exceptions from the underlying apply must surface unchanged."""
        t = _FakeServeContextTesseract("healthy")

        class _Custom(Exception):
            pass

        def _raising_apply(_t, _inputs):
            raise _Custom("boom")

        with mock.patch.object(
            introspect, "container_running", return_value=True
        ):
            with self.assertRaises(_Custom):
                watchdog._apply_with_watchdog(
                    t, {}, timeout=5.0, poll=0.5, _apply_fn=_raising_apply
                )

    def test_tracer_input_bypasses_worker_thread(self) -> None:
        """ARCH-12: when any input carries a JAX tracer, ``_apply_fn`` must
        run on the caller's thread so primitive binding sees the active
        trace. If the watchdog forked a worker thread, ``_apply_fn`` would
        observe a tracer with no active trace context and the downstream
        ``jax.grad`` call would raise ``UnexpectedTracerError``.

        We verify by asserting that ``_apply_fn`` is invoked on the same
        thread that called ``_apply_with_watchdog`` whenever a tracer is
        present in the input pytree.
        """
        import jax
        import jax.numpy as jnp

        t = _FakeServeContextTesseract("healthy-traced")
        caller_ident = threading.get_ident()
        exec_idents: list[int] = []

        def _recording_apply(_t, inputs):
            exec_idents.append(threading.get_ident())
            # Return a jax-compatible output so ``jax.grad`` of our loss works.
            return {"y": inputs["x"] * 2.0}

        def _loss(x):
            out = watchdog._apply_with_watchdog(
                t, {"x": x}, timeout=5.0, poll=0.1, _apply_fn=_recording_apply
            )
            return jnp.sum(out["y"] ** 2)

        with mock.patch.object(introspect, "container_running", return_value=True):
            # jax.grad traces _loss; inside, inputs["x"] is a Tracer. The fix
            # must short-circuit to a synchronous call on the caller thread.
            jax.grad(_loss)(jnp.array(3.0))

        self.assertGreaterEqual(len(exec_idents), 1)
        for ident in exec_idents:
            self.assertEqual(
                ident,
                caller_ident,
                "apply_fn must run on caller thread when inputs carry a tracer",
            )

    def test_no_orphan_blocking_on_shutdown(self) -> None:
        """After container death, watchdog must return control promptly
        even though the worker thread is still blocked."""
        t = _FakeServeContextTesseract("dying")
        block_event = threading.Event()

        def _blocking_apply(_t, _inputs):
            # Waits forever; test relies on pool.shutdown(wait=False).
            block_event.wait(timeout=30)

        with mock.patch.object(
            introspect, "container_running", return_value=False
        ):
            t0 = time.monotonic()
            with self.assertRaises(ContainerDied):
                watchdog._apply_with_watchdog(
                    t,
                    {},
                    timeout=60.0,
                    poll=0.2,
                    _apply_fn=_blocking_apply,
                )
            elapsed = time.monotonic() - t0

        # We must have returned well before the blocking apply would finish.
        self.assertLess(elapsed, 2.0)
        # Unblock so the orphan thread cleans up.
        block_event.set()


if __name__ == "__main__":
    unittest.main()
