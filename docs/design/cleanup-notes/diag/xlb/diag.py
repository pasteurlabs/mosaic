"""XLB DFG channel: smoothness of Cd(profile) and VJP vs FD for one collision kind.

Usage (inside the XLB image): python diag.py <bgk|kbc|kbc_smooth>
"""

import json
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "/tesseract")
import jax.numpy as jnp
import tesseract_api as m
from xlb.operator.collision import KBC

kind = sys.argv[1]
f64 = m._USE_F64_DIFF
if kind == "kbc":
    pp = m.PrecisionPolicy.FP64FP64 if f64 else m.PrecisionPolicy.FP32FP32
    vs = m._vsets[(2, f64)]
    ops = m._make_ops(
        vs, pp, m.ComputeBackend.JAX, jnp.float64 if f64 else jnp.float32, "bgk"
    )
    ops["bgk"] = KBC(
        velocity_set=vs, precision_policy=pp, compute_backend=m.ComputeBackend.JAX
    )
    m._OPS[(2, f64, "kbc")] = ops
m._select_collision = lambda omega, obstacle: kind

S = pickle.load(open("/data/setup.pkl", "rb"))
w, scale = S["window"], S["scale"]


def run(profile):
    inp = dict(S["base"])
    inp["inflow_profile"] = np.asarray(profile, dtype=np.float32)
    return m.InputSchema.model_validate(inp)


def cd(profile):
    bf = np.asarray(m.apply(run(profile))["body_force"], dtype=np.float64)
    return scale * bf[-w:, 0].mean()


p0 = S["p0"].astype(np.float64)
t = time.time()
cd0 = cd(p0)
t_fwd = time.time() - t
out = {"kind": kind, "f64": f64, "cd0": cd0, "t_fwd_s": t_fwd, "dirs": {}}
inp0 = run(p0)
nsteps = S["base"]["steps"]
cot = np.zeros((nsteps, 2), dtype=np.float32)
cot[-w:, 0] = scale / w
g = np.asarray(
    m.vector_jacobian_product(
        inp0, {"inflow_profile"}, {"body_force"}, {"body_force": cot}
    )["inflow_profile"],
    dtype=np.float64,
)
for name, d in S["dirs"].items():
    vjp_d = float(g @ d)
    fd = {}
    for eps in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4):
        fd[eps] = (cd(p0 + eps * d) - cd(p0 - eps * d)) / (2 * eps)
    alphas = np.linspace(-0.05, 0.05, 21)
    curve = np.array([cd(p0 + a * d) for a in alphas])
    resid = curve - np.polyval(np.polyfit(alphas, curve, 4), alphas)
    out["dirs"][name] = {
        "vjp_dot_d": vjp_d,
        "fd": {str(k): v for k, v in fd.items()},
        "rel_err": {str(k): abs(vjp_d - v) / abs(v) for k, v in fd.items()},
        "curve_alpha": alphas.tolist(),
        "curve_cd": curve.tolist(),
        "quartic_fit_resid_max": float(np.abs(resid).max()),
        "curve_span": float(curve.max() - curve.min()),
    }
print(json.dumps(out))
