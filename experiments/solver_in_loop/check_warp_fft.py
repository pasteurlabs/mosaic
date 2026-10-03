"""GPU regression check for Warp's non-power-of-two pressure projection.

Run inside the Warp solver image; --api can point at a proposed adapter patch.
No benchmark training or local-machine GPU is required.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

import numpy as np
import warp as wp


def main() -> None:
    """Check transform, pressure, and adjoint scales against NumPy on a GPU."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="/tesseract/tesseract_api.py")
    parser.add_argument("--expect-broken", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("fft_check_api", args.api)
    api = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = api
    spec.loader.exec_module(api)
    wp.init()
    device = "cuda:0"
    rng = np.random.default_rng(116)

    def packed(value: np.ndarray) -> wp.array:
        return wp.array(
            np.stack([value.real, value.imag], axis=-1).astype(np.float32),
            dtype=wp.vec2f,
            device=device,
            requires_grad=True,
        )

    def unpacked(value: wp.array) -> np.ndarray:
        raw = value.numpy()
        return raw[..., 0] + 1j * raw[..., 1]

    def check(label: str, actual: np.ndarray, expected: np.ndarray) -> float:
        error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        print(json.dumps({"check": label, "relative_error": error}), flush=True)
        if not args.expect_broken:
            assert np.isfinite(error) and error < 1e-4, (label, error)
        return error

    for n in (12, 192):
        values = rng.normal(size=(3, n)) + 1j * rng.normal(size=(3, n))
        forward = unpacked(api._bluestein_fft_rows(packed(values), n, "fwd", device))
        error = check(f"DFT N={n}", forward, np.fft.fft(values, axis=-1))
        if args.expect_broken:
            assert error > 1, "Baseline did not reproduce the normalization defect"
            return
        backward = unpacked(api._bluestein_fft_rows(packed(values), n, "bwd", device))
        check(f"unnormalized inverse N={n}", backward, np.fft.ifft(values, axis=-1) * n)

    for dimensions, n in ((2, 64), (2, 192), (3, 12)):
        shape = (n,) * dimensions
        rhs = rng.normal(size=shape).astype(np.float32)
        waves = np.meshgrid(*[np.fft.fftfreq(n, d=1 / n)] * dimensions, indexing="ij")
        eigenvalues = (
            -sum(k**2 for k in waves)
            if dimensions == 2
            else -4
            / (2 * np.pi / n) ** 2
            * sum(np.sin(np.pi * k / n) ** 2 for k in waves)
        )
        inverse = np.zeros(shape)
        np.divide(1, eigenvalues, out=inverse, where=eigenvalues != 0)
        solve = (
            api._spectral_poisson_2d_core
            if dimensions == 2
            else api._spectral_poisson_3d_core
        )
        inputs = packed(rhs)
        with wp.Tape() as tape:
            outputs = solve(inputs, 2 * np.pi, device)
        expected = np.fft.ifftn(np.fft.fftn(rhs) * inverse)
        check(
            f"Poisson {dimensions}D N={n}", unpacked(outputs) / n**dimensions, expected
        )
        cotangent = rng.normal(size=shape).astype(np.float32)
        tape.backward(grads={outputs: packed(cotangent / n**dimensions)})
        expected_adjoint = np.fft.ifftn(np.fft.fftn(cotangent) * inverse)
        check(
            f"Poisson VJP {dimensions}D N={n}",
            unpacked(inputs.grad),
            expected_adjoint,
        )


if __name__ == "__main__":
    main()
