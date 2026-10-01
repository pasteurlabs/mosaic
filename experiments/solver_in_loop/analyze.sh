#!/usr/bin/env bash
# CPU reporting and plot checks on an isolated cluster allocation.
set -euo pipefail
campaign=$1
work="$WORKDIR/mosaic"
mkdir -p "$work"
tar -xf "$campaign/source.tar" -C "$work"
cd "$work"
cp production.uv.lock uv.lock
cp "$campaign/report.py" experiments/solver_in_loop/report.py
cp "$campaign/plots.py" mosaic/benchmarks/problems/navier_stokes_grid/plots.py
export UV_PROJECT_ENVIRONMENT="$SCRATCH_DIR/mosaic-venv"
export UV_CACHE_DIR="$SCRATCH_DIR/uv-cache"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/usr/local/bin/uv sync --frozen --extra dev
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH" PYTHONPATH="$work"
ruff check experiments/solver_in_loop/report.py mosaic/benchmarks/problems/navier_stokes_grid/plots.py
ruff format --check experiments/solver_in_loop/report.py mosaic/benchmarks/problems/navier_stokes_grid/plots.py
python experiments/solver_in_loop/report.py "$campaign" --out "$campaign/report.json" --plots "$campaign/plots"
python - "$campaign" <<'PY'
import io
import sys
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.problems.navier_stokes_grid.plots import _plot_supervised_fields

campaign = Path(sys.argv[1])
for path in sorted((campaign / "results").glob("*/results.tar")):
    with tarfile.open(path) as archive:
        try:
            member = archive.extractfile("./ns-grid/optimization/solver_in_loop_supervised/corrector_fields.npz")
        except KeyError:
            continue
        with np.load(io.BytesIO(member.read())) as data:
            arrays = dict(data)
        if "rollout_supervised_0" not in arrays:
            continue
        fig = _plot_supervised_fields(arrays, arrays["solver_names"].tolist(), campaign, save=False)
        assert fig is not None
        fig.canvas.draw()
        plt.close(fig)
        print(f"Repository field renderer checked against {path.parent.name}", flush=True)
        break
else:
    raise RuntimeError("No completed supervised rollout available to check the field renderer")
PY
