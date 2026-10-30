# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial-condition overview plot (``ics/ics.png``)."""

from __future__ import annotations

from functools import partial
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
    ic_to_2d: Any = None,
    field_cmap: str = "RdBu_r",
    field_symmetric: bool = True,
    **_kw: Any,
) -> None:
    """Draw ONE combined figure of every registered IC as ``ics/ics.png``.

    Registered per IC experiment and so invoked once per IC, but each call
    regenerates every IC of the problem and overwrites the same combined
    chart in the shared ICs directory (the parent of ``out_dir``).
    """
    builder = partial(
        charts.ics,
        to_2d=ic_to_2d,
        field_cmap=field_cmap,
        field_symmetric=field_symmetric,
    )
    render(cfg, Path(out_dir).parent, {"ics": builder}, suite="ics")
