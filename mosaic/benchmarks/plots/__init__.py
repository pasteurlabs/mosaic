"""Top-level plotting namespace.

Shared plot harnesses live under
:mod:`mosaic.benchmarks.problems.shared.plots` (per-suite per-experiment
plots) and each problem's ``extras.py`` (cross-domain aggregator plots
registered as ``_extra/<name>`` via :meth:`Problem.add_extra_plot`).
Both kinds run through ``mosaic run --plots-only``.
"""
