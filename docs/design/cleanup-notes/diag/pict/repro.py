"""PICT on Mosaic's DFG O-grid: non_orthogonal=True vs False, stock solver (no warm start).

Usage: python repro.py true|false
"""

import pickle
import sys

sys.path.insert(0, "/tesseract")
import tesseract_api as m

flag = sys.argv[1] == "true"
orig = m._simulate


def patched(*a, **kw):
    kw["non_orthogonal"] = flag
    kw["warm_start_tol"] = None
    return orig(*a, **kw)


m._simulate = patched
inp = m.InputSchema.model_validate(pickle.load(open("/data/inputs.pkl", "rb")))
out = m.apply(inp)
print("ok, body_force[-1] =", out["body_force"][-1])
