# ruff: noqa: ANN001, ANN201, B023
# Loop-local JIT functions are compiled and consumed before the next iteration.
"""Isolate native-population serialization precision; diagnostic only, on a GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def load_api(path: Path, name: str):
    """Import a preserved adapter copy under a distinct module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Compare identical forced steps differing only in checkpoint quantization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    original = Path("/tesseract/tesseract_api.py").read_text()
    anchor = "            f_final = f_final.astype(jnp.float32)"
    if original.count(anchor) != 1:
        raise ValueError("expected exactly one native-state float32 conversion")
    variants = {
        "original": original,
        "float64_native": original.replace(
            anchor,
            "            f_final = f_final  # diagnostic: retain native precision",
        ),
    }
    records = []
    arrays = {}
    for variant, source in variants.items():
        copy = args.out / f"{variant}_api.py"
        copy.write_text(source)
        api = load_api(copy, f"xlb_{variant}")
        if jax.default_backend() != "gpu":
            raise RuntimeError("GPU required")
        for index, filename in enumerate(p["cache_paths"]):
            with np.load(filename, allow_pickle=False) as data:
                initial = data["state"].copy()
                if hashlib.sha256(initial.tobytes()).hexdigest() != str(data["sha256"]):
                    raise ValueError("cache checksum mismatch")
            curves = {}
            for factor in p["factors"]:
                dt = 0.01 / factor
                x = jnp.asarray(initial)
                y = jnp.arange(x.shape[1], dtype=x.dtype) / x.shape[1]
                force = (
                    jnp.zeros_like(x)
                    .at[..., 0]
                    .set(jnp.sin(2 * jnp.pi * 6 * y)[None, :, None])
                )

                def step(x, native):
                    value, _, state = api.xlb_fwd(
                        x + 0.5 * dt * force,
                        viscosity=0.001,
                        dt=dt,
                        steps=1,
                        domain_extent=2 * np.pi,
                        state=native,
                        return_state=True,
                        _use_f64=True,
                    )
                    return value + 0.5 * dt * force, state

                first = jax.jit(step)(x, None)

                def body(carry, _):
                    return step(*carry), None

                frame = jax.jit(
                    lambda carry: jax.lax.scan(body, carry, None, length=4 * factor)[0]
                )
                rest = jax.jit(
                    lambda carry: jax.lax.scan(
                        body, carry, None, length=4 * factor - 1
                    )[0]
                )
                carry = rest(first)
                frames = [initial, np.asarray(carry[0])]
                for _ in range(47):
                    carry = frame(carry)
                    frames.append(np.asarray(carry[0]))
                curve = np.stack(frames)
                curves[factor] = curve
                arrays[f"{variant}_ic{index}_f{factor}"] = curve
                print(
                    json.dumps(
                        {
                            "variant": variant,
                            "ic": index,
                            "factor": factor,
                            "finite": bool(np.isfinite(curve).all()),
                        }
                    ),
                    flush=True,
                )
            pairs = []
            for low, high in zip(p["factors"][:-1], p["factors"][1:], strict=True):
                # Compare on the full192 grid as an additional diagnostic.
                a, b = curves[low].astype(np.float64), curves[high].astype(np.float64)
                errors = np.linalg.norm(
                    (a - b).reshape(49, -1), axis=1
                ) / np.linalg.norm(b.reshape(49, -1), axis=1)
                pairs.append(
                    {
                        "factors": [low, high],
                        "max_error": float(errors.max()),
                        "errors": errors.tolist(),
                    }
                )
            records.append(
                {
                    "variant": variant,
                    "ic": index,
                    "pairs": pairs,
                    "adapter_sha256": hashlib.sha256(source.encode()).hexdigest(),
                }
            )
    np.savez_compressed(args.out / "fields.npz", **arrays)
    (args.out / "outcome.json").write_text(
        json.dumps(
            {
                "completed": True,
                "admitted": False,
                "scope": "diagnostic internal call; full192 errors; native precision only",
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
