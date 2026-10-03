"""Compare immutable pilot datasets and quantify cross-run numerical drift on Slurm."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

_TASK_FIELDS = (
    "task_seeds",
    "initial",
    "goal",
    "fine_initial",
    "fine_goal",
    "fine_goal_rollout",
    "generating_controls",
    "initial_seed",
    "goal_seed",
    "goal_audit_terminal",
)
_DATASET_KEYS = {
    "times",
    *(f"{split}_{name}" for split in ("train", "validation") for name in _TASK_FIELDS),
}


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _dataset_sha(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _read(
    path: Path,
) -> tuple[dict, dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    with tarfile.open(path) as archive:
        members = {m.name.lstrip("./"): m for m in archive.getmembers() if m.isfile()}
        outcome = json.load(archive.extractfile(members["outcome.json"]))
        protocol = json.load(archive.extractfile(members["protocol.json"]))
        with np.load(
            io.BytesIO(archive.extractfile(members["fields.npz"]).read()),
            allow_pickle=False,
        ) as data:
            arrays = {key: np.array(data[key]) for key in sorted(_DATASET_KEYS)}
            final_key = f"validation_update{outcome['updates']}_fine_rollouts"
            evaluation = {
                "final": np.array(data[final_key][:, -1]),
                "zero": np.array(data["validation_fine_zero_rollouts"][:, -1]),
            }
    return outcome, protocol, arrays, evaluation


def _difference(value: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    if value.shape != baseline.shape:
        return {"shape_mismatch": [list(baseline.shape), list(value.shape)]}
    delta = value.astype(np.float64) - baseline.astype(np.float64)
    rms = float(np.sqrt(np.mean(delta**2)))
    baseline_rms = float(np.sqrt(np.mean(baseline.astype(np.float64) ** 2)))
    return {
        "identical_bytes": value.dtype == baseline.dtype
        and _sha(value) == _sha(baseline),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "changed_entries": int(np.count_nonzero(delta)),
        "total_entries": int(delta.size),
        "max_absolute": float(np.max(np.abs(delta))),
        "rms_absolute": rms,
        "relative_l2": rms / max(baseline_rms, 1e-30),
        "baseline_rms": baseline_rms,
        "finite": bool(np.isfinite(value).all() and np.isfinite(baseline).all()),
    }


def _physics(
    arrays: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
    velocity_scale: float,
) -> dict[str, Any]:
    final = evaluation["final"].astype(np.float64)
    own_goal = arrays["validation_fine_goal"].astype(np.float64)
    common_goal = baseline["validation_fine_goal"].astype(np.float64)
    scale_squared = velocity_scale**2
    own_mse = np.mean((final - own_goal) ** 2, axis=(1, 2, 3, 4)) / scale_squared
    common_mse = np.mean((final - common_goal) ** 2, axis=(1, 2, 3, 4)) / scale_squared
    drift = np.mean((own_goal - common_goal) ** 2, axis=(1, 2, 3, 4)) / scale_squared
    signal = (
        np.mean((own_goal - evaluation["zero"]) ** 2, axis=(1, 2, 3, 4)) / scale_squared
    )
    return {
        "note": "Rescoring holds saved final states fixed; this is not rerunning policies on common initial states.",
        "own_goal_terminal_mse_by_task": own_mse.tolist(),
        "common_goal_terminal_mse_by_task": common_mse.tolist(),
        "mean_terminal_mse_shift": float(np.mean(common_mse - own_mse)),
        "max_absolute_task_mse_shift": float(np.max(np.abs(common_mse - own_mse))),
        "goal_drift_normalized_mse_by_task": drift.tolist(),
        "zero_control_goal_signal_normalized_mse_by_task": signal.tolist(),
        "goal_drift_to_control_signal_rms_by_task": np.sqrt(
            drift / np.maximum(signal, 1e-30)
        ).tolist(),
        "train_goal_drift_normalized_mse": float(
            np.mean(
                (
                    arrays["train_goal"].astype(np.float64)
                    - baseline["train_goal"].astype(np.float64)
                )
                ** 2
            )
            / scale_squared
        ),
    }


def main() -> None:
    """Retain every cell and per-array/task difference; never rewrite source results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        path
        for path in sorted((args.campaign / "results").glob("*/results.tar"))
        if not path.parent.name.startswith("warm-")
    ]
    if len(paths) != 12:
        raise ValueError(f"expected all twelve pilot cells, found {len(paths)}")
    rows, baseline, baseline_cell = [], None, paths[0].parent.name
    for path in paths:
        outcome, protocol, arrays, evaluation = _read(path)
        reproduced = _dataset_sha(arrays)
        if reproduced != outcome["dataset_sha256"]:
            raise ValueError(
                f"dataset identity could not be reproduced: {path.parent.name}"
            )
        if baseline is None:
            baseline = arrays
        differences = {
            name: _difference(value, baseline[name]) for name, value in arrays.items()
        }
        changed = [
            name
            for name, difference in differences.items()
            if not difference["identical_bytes"]
        ]
        per_task = {}
        for name in changed:
            if name == "times":
                continue
            split = name.split("_", 1)[0]
            seeds = arrays[f"{split}_task_seeds"]
            per_task[name] = [
                {"task_seed": int(seed), **_difference(value, reference)}
                for seed, value, reference in zip(
                    seeds, arrays[name], baseline[name], strict=True
                )
            ]
            if name.endswith("fine_goal_rollout"):
                for row, value, reference in zip(
                    per_task[name], arrays[name], baseline[name], strict=True
                ):
                    row["per_frame"] = [
                        _difference(v, b) for v, b in zip(value, reference, strict=True)
                    ]
        row = {
            "cell": path.parent.name,
            "method": outcome["method"],
            "model_seed": outcome["model_seed"],
            "dataset_sha256": reproduced,
            "source_sha256": protocol["source_sha256"],
            "image": protocol["image"],
            "image_sha256": protocol["image_sha256"],
            "initial_model_sha256": outcome["initial_model_sha256"],
            "array_sha256": {key: _sha(value) for key, value in arrays.items()},
            "changed_arrays": changed,
            "differences": differences,
            "per_task": per_task,
            "physical_scale": _physics(
                arrays,
                baseline,
                evaluation,
                float(outcome["control"]["velocity_scale"]),
            ),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "cell": row["cell"],
                    "changed_arrays": changed,
                    "mse_shift": row["physical_scale"]["mean_terminal_mse_shift"],
                }
            ),
            flush=True,
        )
    groups = {}
    for row in rows:
        groups.setdefault(row["dataset_sha256"], []).append(row["cell"])
    result = {"baseline_cell": baseline_cell, "dataset_groups": groups, "cells": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dataset-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"groups": groups, "report": str(args.out / "dataset-audit.json")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
