# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tables of the solver-in-the-loop study from a mosaic results directory.

Usage: python summarize.py RESULTS_DIR...

Each RESULTS_DIR is the ``-o`` directory of a ``mosaic run``. Several can be
given when the solvers ran separately; ``mosaic run`` overwrites a
``result.json`` rather than merging solvers into it.
"""

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

EXPERIMENTS = ("kolmogorov_sil", "kolmogorov_sil_stability")


def entries(dirs: list[str], experiment: str) -> Iterator[dict]:
    """Every solver's result of ``experiment`` across ``dirs``."""
    for d in dirs:
        path = Path(d) / "ns-grid" / "optimization" / experiment / "result.json"
        if path.exists():
            yield from json.loads(path.read_text())["results"]


def gates(m: dict) -> str:
    """Table cells of the gates of one solver's metrics."""
    ratio = max(a["ratio"] for split in m["audits"].values() for a in split)
    closure = max(max(v) for v in m["closure"].values())
    fd = "/".join(
        f"{m['fd'][k]['relative_error']:.1e}"
        for k in ("zero_readout", "nonzero_readout")
    )
    cost = "/".join(f"{c['update_seconds']:.2f}" for c in m["cost"].values())
    return f"{ratio:.3f} | {closure:.1e} | {fd} | {cost} | {m['eligible']}"


def mean_test_error(errors: list) -> np.ndarray:
    """Mean over test states and frames after the first, per model seed."""
    err = np.asarray(errors, dtype=float)
    return np.nanmean(err[..., 1:], axis=tuple(range(1, err.ndim)))


def stability(m: dict) -> dict:
    """Non-finite rollouts and peak energy relative to uncorrected, per arm."""
    st = m["stability"]
    base = [np.nanmean(np.asarray(r["energy"], dtype=float)) for r in st["uncorrected"]]
    rows = {}
    for key, v in st.items():
        if key == "uncorrected":
            continue
        arm = key.split("/")[1]
        for which in ("selected", "final"):
            for r, b in zip(v[which], base, strict=True):
                agg = rows.setdefault(arm, {"nonfinite": 0, "peak": []})
                agg["nonfinite"] += r["first_nonfinite"] is not None
                agg["peak"].append(np.nanmax(np.asarray(r["energy"], dtype=float)) / b)
    return rows


def main(dirs: list[str]) -> None:
    """Print the gates, training and rollout tables."""
    print("## optimization/kolmogorov_sil and _stability: gates")
    print(
        "experiment | solver | audit ratio | closure | FD error zero/nonzero | "
        "s per update (arms) | eligible"
    )
    for exp in EXPERIMENTS:
        for r in entries(dirs, exp):
            m = r["metrics"]
            row = (
                gates(m)
                if m.get("eligible") is not None and "audits" in m
                else str(m.get("reason"))
            )
            print(f"{exp} | {r['solver']} | {row}")

    print("\n## optimization/kolmogorov_sil_stability: training and rollouts")
    print(
        "solver | arm | failures | test error per seed (uncorrected) | "
        "rollouts non-finite | peak energy / uncorrected mean"
    )
    for r in entries(dirs, "kolmogorov_sil_stability"):
        m = r["metrics"]
        if not m.get("eligible"):
            print(f"{r['solver']} | not eligible: {m.get('reason')}")
            continue
        uncorrected = np.asarray(m["evaluation"]["uncorrected"]["errors"], dtype=float)
        unc = float(np.nanmean(uncorrected[..., 1:]))
        failures = {}
        for key, t in m["training"].items():
            arm = key.split("/")[1]
            failures[arm] = failures.get(arm, 0) + (t["failure"] is not None)
        rolls = stability(m)
        for key, e in m["evaluation"].items():
            if key == "uncorrected":
                continue
            arm = key.split("/")[1]
            per_seed = np.round(mean_test_error(e["errors"]), 3)
            peak = rolls[arm]["peak"]
            print(
                f"{r['solver']} | {arm} | {failures[arm]} | {per_seed} ({unc:.3f}) | "
                f"{rolls[arm]['nonfinite']}/{len(peak)} | {min(peak):.2f}–{max(peak):.2f}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
