"""GPU-only numerical checks of the experimental Warp stage composition."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    """Check taped/forward parity, directional derivatives and periodic composition."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("warp_rk3_candidate", args.candidate)
    api = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = api
    spec.loader.exec_module(api)
    assert api.wp.is_cuda_available()
    device = "cuda:0"
    os.environ.pop("MOSAIC_WARP_INTEGRATOR", None)
    assert not api._rk3_enabled()
    os.environ["MOSAIC_WARP_INTEGRATOR"] = "ssprk3"
    assert api._rk3_enabled()
    records = []
    for n in (64, 192):
        x = np.arange(n) * (2 * np.pi / n)
        u = np.zeros((n, n, 1, 2), np.float32)
        u[:, :, 0, 0] = -0.1 * np.sin(x[None, :]) + 0.03 * np.cos(2 * x[:, None])
        u[:, :, 0, 1] = 0.1 * np.sin(x[:, None]) + 0.03 * np.cos(2 * x[None, :])
        nu, dt, steps = 0.01, 0.01, 3
        output = api.ns2d_ssprk3_forward(u, nu, dt, steps, 2 * np.pi, device=device)
        tape, ux, uy, icx, icy, nu_wp, dt_wp = api.ns2d_ssprk3_tape(
            u, nu, dt, steps, 2 * np.pi, device=device, track_scalar_grads=True
        )
        taped = np.stack([ux.numpy(), uy.numpy()], axis=-1)[:, :, None, :]
        cotangent = u.copy()
        cotangent /= np.linalg.norm(cotangent)
        gradients = api.ns2d_vjp(
            tape, ux, uy, icx, icy, cotangent, device, nu_wp=nu_wp, dt_wp=dt_wp
        )
        checks = []
        direction = cotangent
        for parameter, epsilon in [("v0", 0.001), ("viscosity", 0.002), ("dt", 0.001)]:
            up, um, nup, num, dtp, dtm = u, u, nu, nu, dt, dt
            if parameter == "v0":
                up, um = u + epsilon * direction, u - epsilon * direction
                ad = float(np.sum(gradients["v0"].astype(np.float64) * direction))
            elif parameter == "viscosity":
                nup, num = nu + epsilon, nu - epsilon
                ad = float(gradients[parameter][0])
            else:
                dtp, dtm = dt + epsilon, dt - epsilon
                ad = float(gradients[parameter][0])
            plus = api.ns2d_ssprk3_forward(
                up, nup, dtp, steps, 2 * np.pi, device=device
            )
            minus = api.ns2d_ssprk3_forward(
                um, num, dtm, steps, 2 * np.pi, device=device
            )
            fd = float(
                np.sum((plus.astype(np.float64) - minus) * cotangent) / (2 * epsilon)
            )
            relative = abs(ad - fd) / max(abs(ad), abs(fd), 1e-12)
            checks.append(
                {
                    "parameter": parameter,
                    "epsilon": epsilon,
                    "autodiff": ad,
                    "finite_difference": fd,
                    "relative_error": relative,
                    "passed": relative < 0.05,
                }
            )
        one = api.ns2d_ssprk3_forward(u, nu, dt, 1, 2 * np.pi, device=device)
        split = api.ns2d_ssprk3_forward(one, nu, dt, 2, 2 * np.pi, device=device)
        row = {
            "n": n,
            "finite": bool(np.isfinite(output).all()),
            "forward_tape_max_absolute": float(np.max(np.abs(output - taped))),
            "semigroup_max_absolute": float(np.max(np.abs(output - split))),
            "checks": checks,
        }
        row["passed"] = (
            row["finite"]
            and row["forward_tape_max_absolute"] < 1e-6
            and row["semigroup_max_absolute"] < 1e-6
            and all(c["passed"] for c in checks)
        )
        records.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "passed": all(r["passed"] for r in records),
        "records": records,
        "scope": "Small direct kernel checks, not forced-burn admission",
    }
    args.out.write_text(json.dumps(result, indent=2))
    if not result["passed"]:
        raise AssertionError("experimental kernel gate failed; retain failed checks")


if __name__ == "__main__":
    main()
