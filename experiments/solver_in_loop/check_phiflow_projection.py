"""Validate a proposed periodic PhiFlow projection inside its GPU image."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
from phi import field


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _relative(a: jax.Array | np.ndarray, b: jax.Array | np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(a) - np.asarray(b))
        / max(np.linalg.norm(np.asarray(b)), 1e-12)
    )


def main() -> None:
    """Check discrete operator parity, differentiation, and warm runtime."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    api = _load(args.candidate, "candidate")
    original_path = Path("/tesseract/tesseract_api.py")
    relaxed_path = Path("/tmp/phiflow_relaxed.py")
    relaxed_path.write_text(
        original_path.read_text().replace(
            'math.Solve("CG", 1e-10, 1e-10)', 'math.Solve("CG", 1e-5, 1e-5)'
        )
    )
    relaxed = _load(relaxed_path, "relaxed")
    original = _load(original_path, "original")
    rng = np.random.default_rng(116)
    rows = []
    for shape in [(64, 64), (192, 192), (12, 12, 12), (64, 48), (12, 8, 10)]:
        dimensions = len(shape)
        n = shape[0]
        names = tuple("xyz"[:dimensions])
        bounds = api.Box(**{name: (0, 2 * np.pi) for name in names})
        faces = jnp.asarray(rng.normal(size=(dimensions, *shape)).astype(np.float32))

        def grid(
            a: jax.Array,
            names: tuple[str, ...] = names,
            bounds: object = bounds,
            shape: tuple[int, ...] = shape,
        ):
            return api.StaggeredGrid(
                api.math.stack(
                    {
                        name: api.math.tensor(a[i], api.math.spatial(",".join(names)))
                        for i, name in enumerate(names)
                    },
                    dim=api.math.dual("vector"),
                ),
                api.extrapolation.PERIODIC,
                bounds=bounds,
                **dict(zip(names, shape, strict=True)),
            )

        projected = api._project_periodic_faces_fft(faces)
        div = field.divergence(grid(projected)).values.native(",".join(names))
        div0 = field.divergence(grid(faces)).values.native(",".join(names))

        def native_projection(
            a: jax.Array,
            grid: Callable = grid,
            dimensions: int = dimensions,
            names: tuple[str, ...] = names,
        ):
            native, _ = api.fluid.make_incompressible(
                grid(a), (), solve=api.math.Solve("CG", 1e-5, 1e-5)
            )
            return jnp.stack(
                [
                    native.vector[i].values.native(",".join(names))
                    for i in range(dimensions)
                ]
            )

        native_faces = jax.jit(native_projection)(faces)
        cot = jnp.asarray(rng.normal(size=faces.shape).astype(np.float32))
        _, pull = jax.vjp(api._project_periodic_faces_fft, faces)
        row = {
            "N": n,
            "dimensions": dimensions,
            "shape": shape,
            "native_divergence_ratio": float(
                np.linalg.norm(div) / np.linalg.norm(div0)
            ),
            "native_projection_relative_error": _relative(projected, native_faces),
            "idempotence_relative_error": _relative(
                api._project_periodic_faces_fft(projected), projected
            ),
            "self_adjoint_relative_error": _relative(
                pull(cot)[0], api._project_periodic_faces_fft(cot)
            ),
        }
        assert row["native_divergence_ratio"] < 1e-5, row
        # Independently assemble pressure using NumPy and PhiFlow's native
        # divergence/gradient, because CG itself is the suspected defect.
        div_np = np.asarray(div0)
        waves = np.meshgrid(*[np.fft.fftfreq(size) for size in shape], indexing="ij")
        laplace = -sum(
            4 * np.sin(np.pi * wave) ** 2 / (2 * np.pi / size) ** 2
            for wave, size in zip(waves, shape, strict=True)
        )
        reciprocal = np.zeros_like(laplace)
        np.divide(1, laplace, out=reciprocal, where=laplace != 0)
        pressure = np.fft.ifftn(np.fft.fftn(div_np) * reciprocal).real.astype(
            np.float32
        )
        pressure_grid = api.CenteredGrid(
            api.math.tensor(pressure, api.math.spatial(",".join(names))),
            api.extrapolation.PERIODIC,
            bounds=bounds,
        )
        native_gradient = field.spatial_gradient(pressure_grid, at="face")
        gradient_faces = jnp.stack(
            [
                native_gradient.vector[i].values.native(",".join(names))
                for i in range(dimensions)
            ]
        )
        row["native_operator_relative_error"] = _relative(
            projected, faces - gradient_faces
        )
        row["native_cg_divergence_ratio"] = float(
            np.linalg.norm(
                field.divergence(grid(native_faces)).values.native(",".join(names))
            )
            / np.linalg.norm(div0)
        )
        assert row["native_operator_relative_error"] < 1e-5, row
        assert row["idempotence_relative_error"] < 1e-5, row
        assert row["self_adjoint_relative_error"] < 1e-5, row
        rows.append(row)
        print(json.dumps(row), flush=True)
    for n in (64, 192):
        x = np.arange(n) * 2 * np.pi / n
        xx, yy = np.meshgrid(x, x, indexing="ij")
        initial = (
            np.stack(
                [np.sin(xx) * np.cos(yy), -np.cos(xx) * np.sin(yy)], axis=-1
            ).astype(np.float32)[:, :, None, :]
            * 0.5
        )
        parameters = {
            "boundary_conditions": {
                face: {"type": "periodic"}
                for face in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
            },
            "viscosity": 0.001,
            "dt": 0.02 if n == 64 else 0.02 / 3,
            "steps": 12,
            "domain_extent": 2 * np.pi,
            "return_state": True,
        }
        fwd = jax.jit(
            lambda v, parameters=parameters: api.phiflow_fwd(v, **parameters)[0]
        )
        ref = jax.jit(
            lambda v, parameters=parameters: relaxed.phiflow_fwd(v, **parameters)[0]
        )
        old = jax.jit(
            lambda v, parameters=parameters: original.phiflow_fwd(v, **parameters)[0]
        )
        result = fwd(jnp.asarray(initial)).block_until_ready()
        expected = ref(jnp.asarray(initial)).block_until_ready()
        cot = jnp.asarray(rng.normal(size=initial.shape).astype(np.float32))
        direction = jnp.asarray(rng.normal(size=initial.shape).astype(np.float32))
        direction = direction / jnp.sqrt(jnp.mean(direction**2))

        def loss(v: jax.Array, fwd: Callable = fwd, cot: jax.Array = cot):
            return jnp.mean(fwd(v) * cot)

        gradient = jax.grad(loss)(jnp.asarray(initial))
        derivative = float(jnp.sum(gradient * direction))
        fd = float(
            (loss(initial + 0.001 * direction) - loss(initial - 0.001 * direction))
            / 0.002
        )
        timings = {}
        for name, fn in [("candidate", fwd), ("relaxed_cg", ref), ("original_cg", old)]:
            fn(jnp.asarray(initial)).block_until_ready()
            elapsed = []
            for _ in range(5):
                start = time.perf_counter()
                fn(jnp.asarray(initial)).block_until_ready()
                elapsed.append(time.perf_counter() - start)
            timings[name] = elapsed
        row = {
            "N": n,
            "forward_vs_relaxed_cg": _relative(result, expected),
            "directional_autodiff": derivative,
            "directional_fd": fd,
            "fd_relative_error": abs(derivative - fd)
            / max(abs(fd), abs(derivative), 1e-12),
            "seconds": timings,
        }
        assert row["forward_vs_relaxed_cg"] < 1e-4, row
        assert row["fd_relative_error"] < 0.02, row
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps({"checks_passed": True, "checks": rows}, indent=2))


if __name__ == "__main__":
    main()
