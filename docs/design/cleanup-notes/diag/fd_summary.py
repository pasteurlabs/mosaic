"""Best-eps median FD rel. error and cosine per solver from an fd_check result.json."""

import json
import sys

import numpy as np

for path in sys.argv[1:]:
    for r in json.load(open(path))["results"]:
        sw = r["metrics"].get("eps_sweep") or {}
        best = min(
            sw.items(), key=lambda kv: np.median(kv[1]["rel_error"]), default=None
        )
        if best:
            print(
                f"{path}: {r['solver']} best eps={best[0]} median rel_err={np.median(best[1]['rel_error']):.3e} 1-cos={1 - best[1]['cosine']:.2e}"
            )
