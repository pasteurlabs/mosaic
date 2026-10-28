# Quick probe: TGV 2D error vs analytic for each scheme, plus dt-convergence.
import importlib.util
import json
import sys

import jax.numpy as jnp
import numpy as np

spec = importlib.util.spec_from_file_location("api", "/work/api.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
L = 2 * np.pi


def tgv(N, t, nu):
    x = (np.arange(N) + 0.5) * L / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    f = np.exp(-2 * nu * t)
    return np.stack([np.sin(X) * np.cos(Y) * f, -np.cos(X) * np.sin(Y) * f], -1)[
        :, :, None, :
    ].astype(np.float32)


out = {}
nu = 0.01
T = 1.0
for scheme in sys.argv[1:]:
    api._SCHEME = scheme
    for N in (16, 32, 64):
        for steps in (10, 20, 40, 80):
            dt = T / steps
            try:
                r = api.periodic_fwd(jnp.asarray(tgv(N, 0, nu)), nu, dt, steps, L)
                err = float(
                    np.sqrt(np.mean((np.asarray(r) - tgv(N, T, nu)) ** 2))
                    / np.sqrt(np.mean(tgv(N, T, nu) ** 2))
                )
            except Exception as e:
                err = repr(e)[:80]
            out[f"{scheme} N={N} steps={steps}"] = err
            print(scheme, N, steps, err, flush=True)
with open(f"/out/probe_{'_'.join(sys.argv[1:])}.json", "w") as f:
    json.dump(out, f, indent=1)
