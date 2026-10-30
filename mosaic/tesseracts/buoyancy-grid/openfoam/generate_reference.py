# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and audit native references; execute on a Slurm worker."""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from mosaic_shared.problems.buoyancy_grid import (
    make_interacting_plumes_ic,
    restrict_state,
)


def sha(path: str | Path) -> str:
    """Hash a provenance artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """Compute relative L2 difference."""
    return float(
        np.linalg.norm(a.astype(np.float64) - b.astype(np.float64))
        / max(np.linalg.norm(b.astype(np.float64)), 1e-20)
    )


def diagnostics(state: np.ndarray, initial: np.ndarray) -> dict:
    """Record finiteness, closed-box divergence and mass drift."""
    n = state.shape[0]
    ux = np.pad(state[:-1, :, 0].astype(np.float64), ((1, 1), (0, 0)))
    uy = np.pad(state[:, :-1, 1].astype(np.float64), ((0, 0), (1, 1)))
    div = n * (np.diff(ux, axis=0) + np.diff(uy, axis=1))
    return {
        "divergence_rms": float(np.sqrt(np.mean(div**2))),
        "mass_relative_drift": float(
            abs(
                state[..., 2].mean(dtype=np.float64)
                - initial[..., 2].mean(dtype=np.float64)
            )
            / initial[..., 2].mean(dtype=np.float64)
        ),
        "scalar_min": float(state[..., 2].min()),
        "scalar_max": float(state[..., 2].max()),
        "max_abs_velocity": float(abs(state[..., :2]).max()),
        "finite": bool(np.isfinite(state).all()),
    }


def run(args: argparse.Namespace) -> None:
    """Run one native case and persist its receipt."""
    api = Path(args.repo) / "mosaic/tesseracts/buoyancy-grid/openfoam/tesseract_api.py"
    spec = importlib.util.spec_from_file_location("openfoam_buoyancy_reference", api)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    initial = make_interacting_plumes_ic(N=args.n, seed=0)
    inputs = module.InputSchema(
        state0=initial,
        viscosity=np.array([0.0005], np.float32),
        diffusivity=np.array([0.0005], np.float32),
        buoyancy=np.array([1.0], np.float32),
        dt=np.array([args.dt], np.float32),
        steps=args.steps,
        domain_extent=1.0,
    )
    start = time.perf_counter()
    state = module.apply(inputs)["result"]
    elapsed = time.perf_counter() - start
    stem = Path(args.output)
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stem.with_suffix(".npz"), state=state, initial=initial)
    receipt = {
        "solver": "OpenCFD v2412 buoyantBoussinesqPimpleFoam",
        "container": "opencfd/openfoam-default:2412",
        "ic": "interacting_plumes",
        "seed": 0,
        "N": args.n,
        "dt_requested": args.dt,
        "dt_runtime": float(inputs.dt[0]),
        "steps": args.steps,
        "nu": float(inputs.viscosity[0]),
        "kappa": float(inputs.diffusivity[0]),
        "buoyancy": 1.0,
        "domain_extent": 1.0,
        "boundary": "closed no-slip velocity; homogeneous Neumann scalar",
        "state_transfer": "native pressure-corrected phi/face area; cell temperature minus Tref",
        "wall_seconds": elapsed,
        "adapter_sha256": sha(api),
        "ic_source_sha256": sha(
            Path(args.repo) / "mosaic/mosaic_shared/problems/buoyancy_grid.py"
        ),
        "artifact_sha256": sha(stem.with_suffix(".npz")),
        "diagnostics": diagnostics(state, initial),
    }
    stem.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def assemble(args: argparse.Namespace) -> None:
    """Assemble a candidate only after independently reviewing all audits."""
    paths = [Path(args.coarse), Path(args.fine), Path(args.audit)]
    states = [np.load(p.with_suffix(".npz"))["state"] for p in paths]
    receipts = [json.loads(p.with_suffix(".json").read_text()) for p in paths]
    if [x.shape[0] for x in states] != [64, 128, 128]:
        raise ValueError("Expected N64,N128,N128.")
    for key in (
        "adapter_sha256",
        "ic_source_sha256",
        "ic",
        "seed",
        "nu",
        "kappa",
        "buoyancy",
        "domain_extent",
    ):
        if len({r[key] for r in receipts}) != 1:
            raise ValueError(f"Mismatched provenance: {key}")
    times = [r["dt_runtime"] * r["steps"] for r in receipts]
    if not np.allclose(times, times[0], rtol=1e-9, atol=1e-12):
        raise ValueError("Unequal physical times.")
    audit = {}
    for n in (16, 32):
        a, b, c = [restrict_state(s, s.shape[0] // n) for s in states]
        audit[str(n)] = {}
        for label, sl in [("velocity", slice(0, 2)), ("scalar", 2)]:
            spatial = relative(a[..., sl], b[..., sl])
            temporal = relative(b[..., sl], c[..., sl])
            audit[str(n)][label] = {
                "N64_vs_N128": spatial,
                "N128_dt_vs_half_dt": temporal,
            }
    # External report only: no implicit admission or relaxed acceptance threshold.
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "forward_agreement.npz",
        reference_0=restrict_state(states[1], 8),
        reference_1=restrict_state(states[1], 4),
        sweep_values=np.array([16, 32]),
    )
    manifest = {
        "reference_type": "independent numerical reference, not an exact solution",
        "benchmark": {
            "ic": "interacting_plumes",
            "seed": 0,
            "domain_extent": 1.0,
            "nu": 0.0005,
            "diffusivity": 0.0005,
            "buoyancy": 1.0,
            "dt": 0.005,
            "steps": 80,
        },
        "physics_time": times[0],
        "sweep_values": [16, 32],
        "source_receipts": receipts,
        "convergence_audit": audit,
        "reference_sha256": sha(out / "forward_agreement.npz"),
        "selection": "N128 at dt=.00125; N64 spatial comparison and dt-halved N128 audit; admission requires review",
    }
    (out / "forward_agreement.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--n", type=int, required=True)
    r.add_argument("--dt", type=float, required=True)
    r.add_argument("--steps", type=int, required=True)
    a = sub.add_parser("assemble")
    [
        a.add_argument("--" + key, required=True)
        for key in ("coarse", "fine", "audit", "output")
    ]
    args = p.parse_args()
    run(args) if args.cmd == "run" else assemble(args)
