"""GPU-only analytical and actual centered-projection/null-mode checks."""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np


def main() -> None:
    """Compare the actual FFT pipeline against the exact discrete projector."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "warp_projection_candidate", args.candidate
    )
    api = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = api
    spec.loader.exec_module(api)
    os.environ["MOSAIC_WARP_INTEGRATOR"] = "ssprk3"
    wp = api.wp
    assert wp.is_cuda_available()
    rows = []
    for n in (64, 192):
        dx = 2 * np.pi / n

        def project(u: np.ndarray, n: int = n, dx: float = dx):
            ux, uy = [
                wp.array(u[..., i].copy(), dtype=wp.float32, device="cuda:0")
                for i in (0, 1)
            ]
            dt = wp.array(np.ones(1, np.float32), device="cuda:0")
            rhs = wp.zeros((n, n), dtype=wp.vec2f, device="cuda:0")
            wp.launch(
                api.divergence_2d_to_complex_kernel,
                dim=(n, n),
                inputs=[ux, uy, rhs, dt, 0.5 / dx],
                device="cuda:0",
            )
            pressure = api._spectral_poisson_2d_core(rhs, 2 * np.pi, "cuda:0")
            ox, oy = [
                wp.zeros((n, n), dtype=wp.float32, device="cuda:0") for _ in range(2)
            ]
            wp.launch(
                api.pressure_correct_2d_from_complex_kernel,
                dim=(n, n),
                inputs=[ux, uy, pressure, float(n * n), ox, oy, dt, 0.5 / dx],
                device="cuda:0",
            )
            return np.stack([ox.numpy(), oy.numpy()], axis=-1)

        u = np.random.default_rng(116).normal(size=(n, n, 2)).astype(np.float32)
        symbols = np.sin(2 * np.pi * np.fft.fftfreq(n)) / dx
        symbols[[0, n // 2]] = 0
        sx, sy = np.meshgrid(symbols, symbols, indexing="ij")
        denom = sx * sx + sy * sy
        hats = np.fft.fftn(u, axes=(0, 1))
        dot = sx * hats[..., 0] + sy * hats[..., 1]
        correction = np.divide(dot, denom, out=np.zeros_like(dot), where=denom != 0)
        expected = np.fft.ifftn(
            hats - np.stack([sx * correction, sy * correction], axis=-1), axes=(0, 1)
        ).real
        actual = project(u)
        div = sum(
            (np.roll(actual[..., i], -1, axis=i) - np.roll(actual[..., i], 1, axis=i))
            / (2 * dx)
            for i in (0, 1)
        )
        nulls = []
        for kx, ky in ((0, 0), (n // 2, 0), (0, n // 2), (n // 2, n // 2)):
            phase = (kx * np.arange(n)[:, None] + ky * np.arange(n)[None, :]) % n
            field = np.stack(
                [np.cos(2 * np.pi * phase / n), np.cos(2 * np.pi * phase / n)], axis=-1
            ).astype(np.float32)
            nulls.append(float(np.max(np.abs(project(field) - field))))
        row = {
            "n": n,
            "analytic_max_absolute": float(np.max(np.abs(actual - expected))),
            "divergence_rms": float(np.sqrt(np.mean(div**2))),
            "idempotence_max_absolute": float(np.max(np.abs(project(actual) - actual))),
            "null_mode_errors": nulls,
        }
        row["passed"] = (
            row["analytic_max_absolute"] < 2e-5
            and row["divergence_rms"] < 5e-5
            and row["idempotence_max_absolute"] < 2e-5
            and max(nulls) < 1e-6
        )
        if hasattr(api, "skew_tentative_vel_2d_kernel"):
            velocity = u.astype(np.float64)
            energy_checks = []
            for viscosity in (0.0, 0.001):
                ux, uy = [
                    wp.array(u[..., i].copy(), dtype=wp.float32, device="cuda:0")
                    for i in (0, 1)
                ]
                outputs = [
                    wp.zeros((n, n), dtype=wp.float32, device="cuda:0") for _ in (0, 1)
                ]
                dt = wp.array(np.ones(1, np.float32), device="cuda:0")
                nu = wp.array(np.array([viscosity], np.float32), device="cuda:0")
                wp.launch(
                    api.skew_tentative_vel_2d_kernel,
                    dim=(n, n),
                    inputs=[ux, uy, *outputs, dt, 0.5 / dx, 1 / dx**2, nu],
                    device="cuda:0",
                )
                rhs = (
                    np.stack([o.numpy() for o in outputs], axis=-1).astype(np.float64)
                    - velocity
                )
                reference = np.zeros_like(velocity)
                for component in (0, 1):
                    for axis in (0, 1):
                        value = velocity[..., component]
                        product = velocity[..., axis] * value
                        reference[..., component] -= (
                            0.5
                            * (
                                velocity[..., axis]
                                * (
                                    np.roll(value, -1, axis=axis)
                                    - np.roll(value, 1, axis=axis)
                                )
                                + np.roll(product, -1, axis=axis)
                                - np.roll(product, 1, axis=axis)
                            )
                            / (2 * dx)
                        )
                        reference[..., component] += (
                            viscosity
                            * (
                                np.roll(value, -1, axis=axis)
                                + np.roll(value, 1, axis=axis)
                                - 2 * value
                            )
                            / dx**2
                        )
                power = float(np.sum(velocity * rhs))
                relative_power = abs(power) / max(
                    float(np.sum(np.abs(velocity * rhs))), 1e-30
                )
                relative_rhs = float(
                    np.linalg.norm(rhs - reference) / np.linalg.norm(reference)
                )
                energy_checks.append(
                    {
                        "viscosity": viscosity,
                        "power": power,
                        "relative_energy_production": relative_power,
                        "relative_rhs_error": relative_rhs,
                        "passed": relative_rhs < 1e-6
                        and (relative_power < 1e-6 if viscosity == 0 else power < 0),
                    }
                )
            row["energy_checks"] = energy_checks
            row["passed"] = row["passed"] and all(c["passed"] for c in energy_checks)
        rows.append(row)
    result = {"passed": all(r["passed"] for r in rows), "records": rows}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)
    assert result["passed"]


if __name__ == "__main__":
    main()
