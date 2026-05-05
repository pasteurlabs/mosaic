"""Core watchdog logic wrapping ``tesseract_jax.apply_tesseract``.

Design:

1. Submit the real ``tesseract_jax.apply_tesseract(t, inputs)`` call to a
   fresh single-worker :class:`concurrent.futures.ThreadPoolExecutor`.
2. Poll ``future.result(timeout=poll)`` in a loop. On ``TimeoutError``,
   ask :mod:`.introspect` whether the backing docker container is still
   running and whether the wall-clock deadline has been reached.
3. If the container is not running → raise :class:`.ContainerDied`.
   If the wall-clock deadline has been reached → raise
   :class:`.WatchdogTimeout`.
4. In ``finally``, call ``pool.shutdown(wait=False)`` so we do **not** block
   on the orphan worker thread. The worker remains stuck on its kernel
   read; it will die with the daemon process, or get a socket error when
   the container is torn down. This is an acceptable bounded leak — see
   the module docstring note on thread accumulation below.

Why a fresh daemon :class:`threading.Thread` per call
-----------------------------------------------------
Rather than a :class:`concurrent.futures.ThreadPoolExecutor` we use a
plain daemon :class:`threading.Thread`. Two reasons:

- Executor worker threads are **non-daemon** by default, so an orphan
  worker stuck on a kernel read prevents interpreter shutdown (the
  process refuses to exit until the blocked read returns, which is the
  exact wedge we're trying to escape). ``shutdown(wait=False)`` only
  affects the pool's tracking — it does not convert the worker to a
  daemon. Daemon threads, by contrast, die with the interpreter.

- A private thread per call rules out cross-contamination: an orphan
  thread from a prior wedge cannot deliver a stale result to a later
  call because it writes into a queue that is unreferenced after its
  watchdog returned.

Thread-leak bound: one orphan daemon thread per wedged call. They hold
the Python-level handle to the blocked socket read, consuming tens of KB
of kernel memory each. Negligible on agent timescales, and the daemon
process exits after each sweep anyway.

JAX compatibility
-----------------
The watchdog is invoked at concrete-value time — i.e. after JAX has
lowered the traced ``loss_fn`` and is running the callable registered by
``tesseract_jax``. ``apply_tesseract`` still runs on a worker thread and
returns the same dict shape (decoded arrays) the upstream function would,
so ``jax.grad`` / ``jax.value_and_grad`` see an identical return. The
watchdog does not itself JIT-compile anything.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any

from . import introspect as _introspect
from .exceptions import ContainerDied, WatchdogTimeout

# Default wall-clock deadline for a single ``apply_tesseract`` call, in
# seconds. Overridable at call site via the ``timeout=`` kwarg, or
# globally via ``set_default_timeout`` / the ``MOSAIC_TESSERACT_TIMEOUT``
# env var at import time. 1200 s (20 min) matches the runner's historical
# default so behaviour is unchanged on the happy path.
_DEFAULT_TIMEOUT: float = float(os.environ.get("MOSAIC_TESSERACT_TIMEOUT", "1200"))

# Default polling interval — how often we check container liveness and
# wall-clock while waiting for the call to complete. 2 s is a good trade-off:
# fast enough that a dead container is noticed within ~2 s + negative cache
# TTL, slow enough that `docker inspect` overhead is negligible (< 1% of CPU
# even for very long calls).
_DEFAULT_POLL: float = 2.0


def set_default_timeout(seconds: float) -> None:
    """Override the global default wall-clock deadline (in seconds).

    Individual calls can still pass ``timeout=`` to override further. Mostly
    useful for tests and for callers that want a suite-level override without
    touching every call site.
    """
    global _DEFAULT_TIMEOUT
    _DEFAULT_TIMEOUT = float(seconds)


def get_default_timeout() -> float:
    """Return the currently-configured default wall-clock deadline."""
    return _DEFAULT_TIMEOUT


def _inputs_contain_tracer(inputs: Any) -> bool:
    """Return True if ``inputs`` (a pytree) contains any JAX tracer.

    Tracer detection is required because JAX tracing contexts are
    thread-local: running ``tesseract_jax.apply_tesseract`` (which calls
    ``tesseract_dispatch_p.bind``) on a worker thread bypasses the main
    thread's trace, causing an ``UnexpectedTracerError`` when the primitive
    binding tries to escape the transformation scope.

    When inputs contain a tracer, the call must run on the caller's thread
    so the primitive binding participates in the active JAX trace. At that
    stage no HTTP call happens anyway (the primitive is merely recorded),
    so the watchdog's wall-clock / container-liveness guarantees are
    irrelevant — the actual HTTP execution happens later during primitive
    lowering, on whatever thread JAX chooses.

    Running ``_apply_fn`` on a worker thread unconditionally would silently
    kill every JAX-transformed call across cost/gradient/optimization suites.
    """
    try:
        import jax  # deferred: CLI-only hosts may lack jax
        import jax.core as _jc
    except ImportError:  # pragma: no cover — CI-only path
        return False
    try:
        flat, _ = jax.tree_util.tree_flatten(inputs)
    except Exception:
        # tree_flatten may fail on exotic inputs (e.g. HexMesh pydantic
        # models). Fall back to a shallow scan of dict values.
        if isinstance(inputs, dict):
            for v in inputs.values():
                if isinstance(v, _jc.Tracer):
                    return True
        return False
    for leaf in flat:
        if isinstance(leaf, _jc.Tracer):
            return True
    return False


def _apply_with_watchdog(
    t: Any,
    inputs: dict,
    *,
    timeout: float | None = None,
    poll: float = _DEFAULT_POLL,
    _apply_fn: Any = None,
) -> Any:
    """Core implementation. See :func:`apply_tesseract` for the public entry
    point — this function is kept separate so the test suite can inject
    ``_apply_fn`` (a fake ``apply_tesseract``) without monkey-patching the
    real ``tesseract_jax`` module.
    """
    if _apply_fn is None:
        # Deferred import so merely importing this module does not require
        # tesseract_jax to be present (the CLI imports benchmarks on hosts
        # that never execute solvers).
        from tesseract_jax import apply_tesseract as _apply_fn  # type: ignore

    # JAX tracing is thread-local. When any input carries a tracer
    # (``jax.grad``/``jit``/``vmap`` in flight), running ``_apply_fn`` on a
    # worker thread breaks primitive binding. Short-circuit to a direct call
    # on the caller's thread; no watchdog needed since primitive binding does
    # not make an HTTP request (that happens later during primitive lowering).
    if _inputs_contain_tracer(inputs):
        return _apply_fn(t, inputs)

    deadline_seconds = _DEFAULT_TIMEOUT if timeout is None else float(timeout)
    container_id = _introspect.get_container_id(t)

    result_q: queue.Queue = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            result_q.put(("ok", _apply_fn(t, inputs)))
        except BaseException as exc:  # noqa: BLE001
            # Propagate underlying exceptions back to the caller instead of
            # swallowing them. We use BaseException so even SystemExit /
            # KeyboardInterrupt from inside the worker become normal
            # exceptions on the caller side.
            result_q.put(("err", exc))

    thread = threading.Thread(
        target=_worker,
        name="tess-watchdog-apply",
        daemon=True,  # critical: orphan threads must die with interpreter
    )
    thread.start()

    deadline = time.monotonic() + deadline_seconds
    while True:
        remaining = deadline - time.monotonic()
        wait = min(poll, max(0.0, remaining))
        try:
            tag, payload = result_q.get(timeout=wait)
        except queue.Empty:
            pass
        else:
            if tag == "ok":
                return payload
            # tag == "err": re-raise the underlying exception.
            raise payload  # type: ignore[misc]

        # Check container liveness first — a dead container with a stuck
        # socket is the primary failure mode this watchdog guards against.
        if container_id is not None and not _introspect.container_running(container_id):
            raise ContainerDied(
                f"tesseract-runtime container {container_id!r} is no "
                f"longer running; apply_tesseract was wedged on its "
                f"HTTP socket. Aborted after "
                f"{deadline_seconds - remaining:.1f}s."
            )

        if time.monotonic() >= deadline:
            mode = (
                "container-liveness + wall-clock"
                if container_id is not None
                else "wall-clock-only (container id unknown)"
            )
            raise WatchdogTimeout(
                f"apply_tesseract did not return within "
                f"{deadline_seconds:.0f}s ({mode} watchdog). "
                f"Container {container_id!r} reported as running."
            )


def apply_tesseract(
    t: Any,
    inputs: dict,
    *,
    timeout: float | None = None,
    poll: float = _DEFAULT_POLL,
) -> Any:
    """Drop-in replacement for :func:`tesseract_jax.apply_tesseract` with a
    container-liveness + wall-clock watchdog.

    Happy path is fully transparent: same call, same return value, same
    exceptions. On a hang the watchdog raises :class:`.ContainerDied` or
    :class:`.WatchdogTimeout` (both subclasses of :class:`.WatchdogError`;
    ``WatchdogTimeout`` also inherits from :class:`TimeoutError` so
    ``except TimeoutError`` callers keep working).

    Parameters
    ----------
    t : tesseract_core.Tesseract
        The tesseract handle, as returned by ``Tesseract.from_image``.
    inputs : dict
        Call payload — passed through verbatim.
    timeout : float | None
        Wall-clock deadline in seconds. Defaults to
        :func:`get_default_timeout` (which in turn defaults to
        ``MOSAIC_TESSERACT_TIMEOUT`` or 1200 s).
    poll : float
        How often to re-check container liveness. 2 s by default — does not
        need tuning in normal use.
    """
    return _apply_with_watchdog(t, inputs, timeout=timeout, poll=poll)
