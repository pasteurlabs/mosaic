"""Independent centered projection/null-mode and actual-kernel energy checks."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp


def main() -> None:
    """Evaluate analytical discrete identities on an allocated GPU."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("invariant_api", args.api)
    api = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = api
    spec.loader.exec_module(api)
    wp.init()
    device = "cuda:0"
    rng = np.random.default_rng(116)
    result = {}
    for n in [12, 32]:
        shape = (n,) * 3
        u = rng.normal(scale=0.1, size=(*shape, 3)).astype(np.float32)
        h = 2 * np.pi / n
        frequency = np.sin(2 * np.pi * np.fft.fftfreq(n)) / h
        frequency[0] = frequency[n // 2] = 0
        waves = np.stack(
            np.meshgrid(frequency, frequency, frequency, indexing="ij"), axis=-1
        )
        denom = np.sum(waves**2, axis=-1)
        spectrum = np.fft.fftn(u, axes=(0, 1, 2))
        correction = np.zeros(shape, dtype=np.complex128)
        np.divide(
            np.sum(waves * spectrum, axis=-1), denom, out=correction, where=denom > 0
        )
        expected = np.fft.ifftn(
            spectrum - waves * correction[..., None], axes=(0, 1, 2)
        ).real
        actual = api.ns3d_solve_forward(u, 0.0, 1e-9, 1, 2 * np.pi, device)
        projection_error = float(np.max(np.abs(actual - expected)))
        # With dt=1, subtracting input exposes the actual kernel's inviscid RHS.
        inputs = [
            wp.array(u[..., i], dtype=wp.float32, device=device) for i in range(3)
        ]
        outputs = [wp.zeros(shape, dtype=wp.float32, device=device) for _ in range(3)]
        scalar = lambda v: wp.array(
            np.asarray([v], np.float32), dtype=wp.float32, device=device
        )
        wp.launch(
            api.tentative_vel_3d_kernel,
            dim=shape,
            inputs=[*inputs, *outputs, scalar(1), 0.5 / h, 1 / h**2, scalar(0)],
            device=device,
        )
        rhs = np.stack([v.numpy() for v in outputs], axis=-1).astype(np.float64) - u
        energy = abs(float(np.sum(u * rhs))) / max(
            float(np.linalg.norm(u) * np.linalg.norm(rhs)), 1e-30
        )
        indices = np.indices(shape)
        null_errors = []
        for bits in itertools.product([0, 1], repeat=3):
            signs = (-1.0) ** sum(b * indices[j] for j, b in enumerate(bits))
            field = (signs[..., None] * np.asarray([0.1, -0.2, 0.3])).astype(np.float32)
            solved = api.ns3d_solve_forward(field, 0.0, 0.01, 2, 2 * np.pi, device)
            null_errors.append(
                {"mode": bits, "max_error": float(np.max(np.abs(solved - field)))}
            )
        result[str(n)] = {
            "projection_max_error": projection_error,
            "inviscid_energy_relative_production": energy,
            "null_modes": null_errors,
        }
    args.out.write_text(json.dumps(result, indent=2))
    assert all(
        v["projection_max_error"] < 1e-5
        and v["inviscid_energy_relative_production"] < 1e-7
        and max(m["max_error"] for m in v["null_modes"]) < 1e-6
        for v in result.values()
    ), result


if __name__ == "__main__":
    main()
