"""Single-IC transfer preparation and exact ordered assembly on Slurm."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
from tesseract_core import Tesseract

from experiments.solver_in_loop.final_run import (
    file_hash,
    identity,
    load_dataset,
    prepare,
)
from mosaic.benchmarks.problems import get_config


def assemble(payload: dict[str, Any], out: Path) -> dict[str, Any]:
    """Assemble disjoint observations; normalize exclusively from all training ICs."""
    started = time.perf_counter()
    train_ids = list(payload["run"]["dataset"]["train_seeds"])
    eval_ids = list(payload["run"]["dataset"]["test_seeds"])
    if train_ids != list(range(8)) or eval_ids != list(range(20000, 20032)):
        raise ValueError("transfer freezes eight training ICs and32 confirmation ICs")
    shards = {}
    for directory in map(Path, payload["shard_dirs"]):
        metadata = json.loads((directory / "dataset.json").read_text())
        configured = {
            **payload,
            "dataset_path": str(directory / "dataset.npz"),
            "dataset_metadata_path": str(directory / "dataset.json"),
            "dataset_sha256": metadata["dataset_sha256"],
        }
        arrays, metadata = load_dataset(configured)
        ids = metadata["train_seeds"]
        if len(ids) != 1 or metadata["eval_seeds"] or not metadata.get("single_ic"):
            raise ValueError("expected a single-IC shard")
        seed = int(ids[0])
        if seed in shards or seed not in train_ids + eval_ids:
            raise ValueError("duplicate or unexpected IC")
        if seed in eval_ids and arrays["supervised_inputs"].size:
            raise ValueError("held-out shards must not generate supervised pairs")
        if (
            seed in train_ids
            and arrays["supervised_inputs"].shape != arrays["train"][:, 1:].shape
        ):
            raise ValueError("training supervised pairs missing")
        shards[seed] = (arrays, metadata, directory)
    if set(shards) != set(train_ids + eval_ids):
        raise ValueError("missing ICs; never assemble a survivor-only subset")

    def stack(key: str, seeds: list[int]) -> np.ndarray:
        return np.stack([shards[seed][0][key][0] for seed in seeds])

    combined = {
        "train": stack("train", train_ids),
        "train_rollouts": stack("train_rollouts", train_ids),
        "reference": stack("train_rollouts", eval_ids),
        "supervised_inputs": stack("supervised_inputs", train_ids),
        "train_native_errors": stack("train_native_errors", train_ids),
        "native_errors": stack("train_native_errors", eval_ids),
        "native_rollouts": stack("train_native_rollouts", eval_ids),
        "train_seeds": np.asarray(train_ids),
        "eval_seeds": np.asarray(eval_ids),
    }
    velocity_scale = float(np.sqrt(np.mean(combined["train"] ** 2)) + 1e-8)
    seen = min(
        int(payload["run"]["evaluation"].get("seen_ic_trajectories", 4)), len(train_ids)
    )
    horizon = min(
        combined["train"].shape[1] - 1, combined["train_native_errors"].shape[1] - 1
    )
    training = payload["run"]["training"]
    loss_scale = max(
        float(
            np.mean(
                np.mean(combined["train_native_errors"][:seen], axis=0)[1 : horizon + 1]
                ** 2
            )
        ),
        float(training.get("loss_scale_floor", 1e-6)),
    )
    if training.get("loss_normalization") == "target_energy":
        loss_scale = 1.0
    elif training.get("loss_normalization") != "solver_baseline":
        raise ValueError("unsupported loss normalization")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "dataset.npz", **combined)
    ordered = [shards[seed] for seed in train_ids + eval_ids]
    metadata = {
        "identity": identity(payload),
        "dataset_sha256": file_hash(out / "dataset.npz"),
        "dataset_role": "validation",
        "comparison_role": "frozen_transfer_confirmation_no_selection",
        "train_seeds": train_ids,
        "eval_seeds": eval_ids,
        "velocity_scale": velocity_scale,
        "training_loss_scale": loss_scale,
        "dataset_preparation_wall_time_s": sum(
            m["dataset_preparation_wall_time_s"] for _, m, _ in ordered
        ),
        "supervised_dataset_wall_time_s": sum(
            shards[seed][1]["supervised_dataset_wall_time_s"] for seed in train_ids
        ),
        "native_state_threading": all(
            m["native_state_threading"] for _, m, _ in ordered
        ),
        "reference_audits_by_shard": [m["reference_audit"] for _, m, _ in ordered],
        "shards": [
            {
                "seed": seed,
                "path": str(shards[seed][2]),
                "dataset_sha256": shards[seed][1]["dataset_sha256"],
                "metadata_sha256": file_hash(shards[seed][2] / "dataset.json"),
            }
            for seed in train_ids + eval_ids
        ],
        "admission_scope": (
            "Every IC must pass original reference gates and stricter individual native-closure gates; "
            "no averaging away a failed IC."
        ),
        "completed": True,
        "admitted": True,
        "assembly_wall_time_s": time.perf_counter() - started,
    }
    (out / "dataset.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    """Delegate simulation to a GPU allocation and assembly to a CPU allocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "assemble"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocol.json").write_text(json.dumps(payload, indent=2))
    try:
        if args.phase == "assemble":
            result = assemble(payload, args.out)
        else:
            if jax.default_backend() != "gpu":
                raise RuntimeError(
                    "transfer reference preparation requires allocated GPU"
                )
            if not payload.get("single_ic"):
                raise ValueError("single-IC contract required")
            cfg = get_config("ns-grid")
            spec = next(
                s for s in cfg.solvers if s.key == payload["solver"].replace("-", "_")
            )
            ctx = SimpleNamespace(
                name=spec.name,
                make_inputs=cfg.make_inputs,
                phys=payload["run"]["physics"],
                domain_extent=float(payload.get("domain_extent", 2 * np.pi)),
                run=payload["run"],
                output_key="result",
            )
            os.environ["MOSAIC_REFERENCE_CACHE_ID"] = (
                payload["source_sha256"] + ":" + payload["image"]
            )
            with Tesseract.from_url(args.url, timeout=(30, 1200)) as solver:
                result = prepare(solver, ctx, payload, args.out)
    except Exception as error:
        result = {
            "completed": False,
            "admitted": False,
            "failure": f"{type(error).__name__}: {error}",
        }
    (args.out / "outcome.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
