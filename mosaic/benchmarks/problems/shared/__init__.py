"""Shared toolbox of run_* harnesses and plot helpers.

This module is a *library*, not a framework. Each problem in
:mod:`mosaic.benchmarks.problems.<name>` imports the harnesses it needs
(``run_agreement``, ``run_fd_check``, ``run_topopt``, …) directly from the
corresponding submodule (``forward``, ``gradient``, ``optimization``, …)
and wires each ``Experiment`` with explicit closure-captured deps in its
own ``experiments.py``. Plot functions live under :mod:`.plots` and follow
the same pattern.

There is no central experiment registry here — only :data:`SUITES`, the
canonical suite ordering used by the CLI for display, completion, and
``mosaic status`` table layout.
"""

from __future__ import annotations

SUITES: tuple[str, ...] = ("forward", "cost", "gradient", "optimization", "ics")
