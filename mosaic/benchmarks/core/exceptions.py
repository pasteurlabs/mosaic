"""Exception types for tesseract call failures."""

from __future__ import annotations


class TesseractTimeout(TimeoutError):
    """Raised when a tesseract call exceeds its deadline.

    Subclasses ``TimeoutError`` so existing ``except TimeoutError`` clauses
    continue to work.
    """


# Backward-compat aliases.
WatchdogError = TesseractTimeout
ContainerDied = TesseractTimeout
WatchdogTimeout = TesseractTimeout
