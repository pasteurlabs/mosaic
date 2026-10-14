# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial-condition overview plot (``ics/ics.png``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.problems.shared.plots.charts import ics as charts
from mosaic.benchmarks.problems.shared.plots.charts.common import render


def plot_ic(
    cfg: Problem,
    ic_name: str,
    ic: np.ndarray,
    out_dir: Path,
    **_kw: Any,
) -> None:
    """Draw ONE combined figure of every registered IC as ``ics/ics.png``.

    Registered per IC experiment and so invoked once per IC, but each call
    regenerates every IC of the problem and overwrites the same combined
    chart in the shared ICs directory (the parent of ``out_dir``).
    """
    render(cfg, Path(out_dir).parent, {"ics": charts.ics}, suite="ics")
