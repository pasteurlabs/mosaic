"""Diagnose XLB temporal convergence on saved post-burn fields on Slurm."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
from tesseract_core import Tesseract

from mosaic.benchmarks.problems import get_config
from mosaic.benchmarks.problems.navier_stokes_grid.corrector import (
    relative_l2,
    spectral_restrict,
)


def main() -> None:
    """Preserve all refinement results without admitting a training campaign."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU allocation required")
    cfg = get_config("ns-grid")
    spec = next(s for s in cfg.solvers if s.key == "xlb")
    core = importlib.import_module(
        "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop"
    )
    ctx = SimpleNamespace(
        name=spec.name,
        make_inputs=cfg.make_inputs,
        phys=payload["run"]["physics"],
        domain_extent=2 * np.pi,
        output_key="result",
    )
    records = []
    arrays = {}
    with Tesseract.from_url(args.url, timeout=(30, 1200)) as solver:
        for index, filename in enumerate(payload["cache_paths"]):
            with np.load(filename, allow_pickle=False) as cache:
                initial = cache["state"].copy()
                if hashlib.sha256(initial.tobytes()).hexdigest() != str(
                    cache["sha256"]
                ):
                    raise ValueError("cache checksum mismatch")
            curves = {}
            for factor in payload["factors"]:
                value, native = initial, None
                frames = [np.asarray(spectral_restrict(value, 64))]
                for _ in range(48):
                    value, native = core._solver_advance_with_physics(
                        solver,
                        ctx,
                        value,
                        dt=0.01 / factor,
                        steps=4 * factor,
                        native_state=native,
                    )
                    frames.append(np.asarray(spectral_restrict(value, 64)))
                curve = np.stack(frames)
                if not np.isfinite(curve).all():
                    raise ValueError("nonfinite rollout")
                curves[factor] = curve
                arrays[f"ic{index}_factor{factor}"] = curve
                print(
                    json.dumps({"cache": filename, "factor": factor, "finite": True}),
                    flush=True,
                )
            pairs = []
            for low, high in zip(
                payload["factors"][:-1], payload["factors"][1:], strict=True
            ):
                errors = [
                    relative_l2(a, b)
                    for a, b in zip(curves[low], curves[high], strict=True)
                ]
                pairs.append(
                    {"factors": [low, high], "max_error": max(errors), "errors": errors}
                )
            records.append(
                {
                    "cache": filename,
                    "state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                    "pairs": pairs,
                }
            )
    np.savez_compressed(args.out / "fields.npz", **arrays)
    (args.out / "outcome.json").write_text(
        json.dumps(
            {
                "completed": True,
                "admitted": False,
                "purpose": "diagnostic only; reused original post-burn fields",
                "records": records,
                "image_sha256": payload["image_sha256"],
                "source_sha256": payload["source_sha256"],
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
