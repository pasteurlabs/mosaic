# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, resolution-independent identities and ICs for operator data.

This module needs only NumPy. Dataset plans can be prepared without the teacher
image; all related physics/resolution/amplitude variants share a parent split.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

VERSION = 1
FAMILIES = ("random", "tgv", "abc", "mean_flow", "near_zero", "perturbed")


def fingerprint(value: Any) -> str:
    """Hash a canonical JSON value."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    """Hash a file without reading the entire dataset into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parent_identity(seed: int, index: int) -> tuple[str, str]:
    """Assign a split independent of grid, physics, batching, and worker count."""
    parent = fingerprint({"version": VERSION, "seed": seed, "parent": index})
    bucket = int(parent[:16], 16) % 100
    return parent, "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def make_field(n: int, seed: int, index: int) -> tuple[np.ndarray, dict]:
    """Sample the same smooth physical parent field on any supported cubic grid.

    Random Fourier waves use integer wavevectors independent of N; low modes
    avoid introducing different random fields merely by changing resolution.
    Benchmark TGV/ABC instances are not generated verbatim.
    """
    parent, split = parent_identity(seed, index)
    rng = np.random.default_rng(int(parent[:16], 16))
    family = FAMILIES[index % len(FAMILIES)]
    axis = np.arange(n, dtype=np.float64) * (2 * np.pi / n)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    phase = rng.uniform(0, 2 * np.pi, size=3)
    amplitude = float(rng.uniform(0.1, 1.25))
    if family == "tgv":
        x, y, z = x + phase[0], y + phase[1], z + phase[2]
        field = np.stack(
            [
                np.sin(x) * np.cos(y) * np.cos(z),
                -np.cos(x) * np.sin(y) * np.cos(z),
                np.zeros_like(x),
            ],
            axis=-1,
        )
    elif family == "abc":
        a, b, c = rng.uniform(0.3, 1.0, size=3)
        field = np.stack(
            [
                a * np.sin(z + phase[2]) + c * np.cos(y + phase[1]),
                b * np.sin(x + phase[0]) + a * np.cos(z + phase[2]),
                c * np.sin(y + phase[1]) + b * np.cos(x + phase[0]),
            ],
            axis=-1,
        ) / math.sqrt(3 * (a + b + c) ** 2)
    else:
        field = np.zeros((n, n, n, 3), dtype=np.float64)
        for _ in range(12):
            wave = rng.integers(-3, 4, size=3)
            if not np.any(wave):
                wave[0] = 1
            direction = rng.normal(size=3)
            direction -= wave * (np.dot(wave, direction) / np.dot(wave, wave))
            direction /= max(np.linalg.norm(direction), 1e-12)
            angle = wave[0] * x + wave[1] * y + wave[2] * z + rng.uniform(0, 2 * np.pi)
            field += np.sin(angle)[..., None] * direction / 12
        if family == "mean_flow":
            field += rng.uniform(-0.3, 0.3, size=3)
        elif family == "near_zero":
            amplitude = 0.0 if index % 12 == 4 else float(rng.uniform(0.001, 0.05))
        elif family == "perturbed":
            # Small longitudinal perturbations exercise the public input space.
            field[..., 0] += 0.02 * np.sin(x + phase[0])
    return (amplitude * field).astype(np.float32), {
        "parent_id": parent,
        "split": split,
        "family": family,
        "amplitude": amplitude,
        "index": index,
    }


def snapshot_steps(steps: int, dense: int = 10, stride: int = 10) -> list[int]:
    """Retain early/late dense windows and sparse intervening checkpoints."""
    if steps < 1 or dense < 0 or stride < 1:
        raise ValueError("steps/stride must be positive and dense nonnegative")
    return sorted(
        {
            0,
            steps,
            *range(1, min(steps, dense) + 1),
            *range(max(0, steps - dense), steps + 1),
            *range(0, steps + 1, stride),
        }
    )


def validate_case(case: dict) -> None:
    """Reject invalid or unsupported physics before a GPU job is launched."""
    for key in ("N", "steps"):
        if type(case.get(key)) is not int or case[key] < (8 if key == "N" else 1):
            raise ValueError(f"invalid {key}: {case.get(key)}")
    for key in ("viscosity", "dt", "domain_extent"):
        if not math.isfinite(case[key]) or case[key] <= 0:
            raise ValueError(f"invalid {key}: {case[key]}")
    if case.get("boundary", "periodic") != "periodic":
        raise ValueError("this generator currently supports periodic 3D only")


def benchmark_manifest() -> dict:
    """Expand registered 3D benchmark cases using the real XLB input factory."""
    from mosaic.benchmarks.problems import get_config

    cfg = get_config("ns-3d-grid")
    cases = []
    spec = next(s for s in cfg.solvers if s.key == "xlb")
    for key, experiment in cfg.experiments.items():
        if key.startswith("ics/"):
            continue
        payloads = [
            d
            for d in experiment.fn.__defaults__ or ()
            if isinstance(d, dict) and "runs" in d
        ]
        if len(payloads) != 1:
            raise ValueError(f"cannot recover registration: {key}")
        for run in payloads[0]["runs"]:
            physics = dict(run["physics"])
            sweep = run.get("sweep", {})
            values = sweep.get("values", [None])
            # Horizon-limit and custom recovery runners consume steps directly.
            if isinstance(physics.get("steps"), list):
                if sweep:
                    raise ValueError(f"ambiguous sweep: {key}")
                sweep = {"key": "steps"}
                values = physics["steps"]
            for value in values:
                phys = dict(physics)
                if sweep:
                    phys[sweep["key"]] = value
                # Only shape is needed by make_inputs; no benchmark IC is sampled.
                dummy = np.broadcast_to(np.float32(0), (phys["N"],) * 3 + (3,))
                inputs = cfg.make_inputs(spec.name, dummy, **phys)
                case = {
                    "N": phys["N"],
                    "viscosity": float(inputs["viscosity"][0]),
                    "dt": float(inputs["dt"][0]),
                    "steps": inputs["steps"],
                    "domain_extent": float(inputs["domain_extent"]),
                    "boundary": "periodic",
                    "experiment": key,
                    "nominal_physics": phys,
                    "ic": run.get("ic", {"name": next(iter(cfg.make_ic)), "seed": 0}),
                }
                validate_case(case)
                case["id"] = fingerprint(case)[:16]
                cases.append(case)
    return {
        "version": VERSION,
        "problem": "ns-3d-grid",
        "teacher": "xlb",
        "cases": cases,
    }


def training_manifest(
    manifest: dict, samples: int = 384, holdout_samples: int = 128
) -> dict:
    """Deduplicate physics and retain bounded windows for training and grid holdouts."""
    groups = {}
    for case in manifest["cases"]:
        key = tuple(case[k] for k in ("N", "viscosity", "dt", "domain_extent"))
        groups.setdefault(key, []).append(case)
    cases = []
    for (n, nu, dt, extent), rows in groups.items():
        horizon = max(r["steps"] for r in rows)
        horizon = min(horizon, 320) if n == 20 else max(40, horizon)
        item = {
            "N": n,
            "viscosity": nu,
            "dt": dt,
            "domain_extent": extent,
            "boundary": "periodic",
            "steps": horizon,
            "samples": samples if n in (8, 16, 32) else holdout_samples,
            "role": "train" if n in (8, 16, 32) else "resolution_holdout",
            "snapshot_steps": snapshot_steps(
                horizon, dense=20, stride=max(5, horizon // 20)
            ),
        }
        item["id"] = fingerprint(item)[:16]
        cases.append(item)
    return {
        "version": VERSION,
        "seed": 20261009,
        "precision": "float64",
        "purpose": "expanded 3D operator training with family-balanced validation and longer horizons",
        "coverage_sha256": fingerprint(manifest),
        "cases": cases,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    args = parser.parse_args()
    coverage = benchmark_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(coverage, indent=2) + "\n")
    if args.training:
        args.training.parent.mkdir(parents=True, exist_ok=True)
        args.training.write_text(
            json.dumps(training_manifest(coverage), indent=2) + "\n"
        )
