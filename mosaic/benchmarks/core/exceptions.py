"""Exception types for the tesseract watchdog.

Callers catch ``WatchdogError`` to treat both modes (container death and
wall-clock deadline) uniformly, or catch the specific subclasses when they
care to distinguish. ``WatchdogTimeout`` additionally inherits from
``TimeoutError`` so existing ``except TimeoutError`` clauses in the runner
continue to work.
"""

from __future__ import annotations


class WatchdogError(Exception):
    """Base class for watchdog-triggered aborts of a tesseract call."""


class ContainerDied(WatchdogError):
    """Raised when the watchdog observes the tesseract-runtime container in a
    non-running state (exited / dead / removed) while the Python-level
    ``apply_tesseract`` worker is still blocked.

    This is the primary failure mode the watchdog is designed to catch: the
    container dies (OOM kill, segfault, SIGKILL from health probe) and the
    HTTP socket enters CLOSE_WAIT, but ``requests`` never returns because
    ``urllib3`` is blocked in a kernel read that no amount of application-level
    timeout= kwargs can interrupt.
    """


class WatchdogTimeout(WatchdogError, TimeoutError):
    """Raised when the wall-clock deadline expires while the container is
    still (reportedly) running — i.e. the tesseract is alive but stuck in a
    pathologically long computation, or we could not determine container
    liveness (introspection fell back to wall-clock-only mode).

    Subclasses ``TimeoutError`` so callers that already handle generic
    ``TimeoutError`` from ``concurrent.futures`` keep working unchanged.
    """
