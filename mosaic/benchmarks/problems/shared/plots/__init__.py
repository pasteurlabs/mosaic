# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark plots, drawn with Altair (Vega-Lite).

The ``plot_*`` functions in the submodules are what problem configs register
via ``plot=`` on :meth:`Problem.add_experiment`. Each builds its charts with
:mod:`.charts` and writes ``<stem>.png`` and ``<stem>.vl.json`` next to the
experiment's results (see :func:`.charts.common.save_chart`).
"""

from mosaic.benchmarks.problems.shared.plots.style import (
    grad_magnitude_2d,
    vorticity_2d,
)

__all__ = ["grad_magnitude_2d", "vorticity_2d"]
