"""Read-only pilot diagnostics. Run inside a Slurm CPU allocation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from experiments.flow_control.control import (
    ControlConfig,
    controls_from_latent,
    init_policy,
)


def main() -> None:
    """Summarize training fit and held-out development behavior without new fitting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for directory in sorted((args.campaign / "results").iterdir()):
        if directory.name.startswith("warm-"):
            continue
        archive = directory / "results.tar"
        if not archive.exists():
            continue
        with tarfile.open(archive) as handle:

            def read(name: str) -> bytes:
                return handle.extractfile("./" + name).read()

            outcome = json.loads(read("outcome.json"))
            fields = np.load(io.BytesIO(read("fields.npz")))
            config = ControlConfig(**outcome["control"])
            trace = outcome["training_trace"]
            key = (
                "paired_loss_mean"
                if outcome["method"] == "spsa"
                else "action_mse"
                if "imitation" in outcome["method"]
                else "loss"
            )
            values = np.asarray([r[key] for r in trace])
            checkpoints = []
            for item in outcome["validation_checkpoints"]:
                update = item["update"]
                model = eqx.tree_deserialise_leaves(
                    io.BytesIO(read(f"model-update{update}.eqx")),
                    init_policy(outcome["model_seed"], config),
                )
                checkpoint = {"update": update, "fine_validation": item}
                for split in ("train", "validation"):
                    initial, goal = fields[f"{split}_initial"], fields[f"{split}_goal"]
                    controls = np.stack(
                        [
                            np.asarray(
                                controls_from_latent(
                                    model(jnp.asarray(a), jnp.asarray(b)), config
                                )
                            )
                            for a, b in zip(initial, goal, strict=True)
                        ]
                    )
                    labels = (
                        fields["training_labels"]
                        if split == "train" and "training_labels" in fields
                        else fields[f"{split}_generating_controls"]
                    )
                    checkpoint[split] = {
                        "action_mse_to_labels_normalized": float(
                            np.mean(((controls - labels) / config.control_bound) ** 2)
                        ),
                        "action_mse_to_zero_normalized": float(
                            np.mean((controls / config.control_bound) ** 2)
                        ),
                        "label_mse_to_zero_normalized": float(
                            np.mean((labels / config.control_bound) ** 2)
                        ),
                        "fraction_near_bound": float(
                            np.mean(np.abs(controls) > 0.95 * config.control_bound)
                        ),
                        "per_task_action_mse": np.mean(
                            ((controls - labels) / config.control_bound) ** 2,
                            axis=(1, 2),
                        ).tolist(),
                        "action_mse_to_demonstrations_normalized": float(
                            np.mean(
                                (
                                    (controls - fields[f"{split}_generating_controls"])
                                    / config.control_bound
                                )
                                ** 2
                            )
                        ),
                    }
                checkpoints.append(checkpoint)
            experts = outcome.get("expert_labels", [])
            rows.append(
                {
                    "cell": directory.name,
                    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "dataset_sha256": outcome["dataset_sha256"],
                    "completed": outcome["completed"],
                    "trace_loss_type": key,
                    "trace_mean_quarters": [
                        float(np.mean(v)) for v in np.array_split(values, 4)
                    ],
                    "trace_last_50_mean": float(np.mean(values[-50:])),
                    "training_task_counts": {
                        str(i): sum(
                            r.get("task_index", r.get("task_seed", -1) - 1000) == i
                            for r in trace
                        )
                        for i in range(len(fields["train_initial"]))
                    },
                    "expert_label_sources": [r["label_source"] for r in experts],
                    "expert_labels": experts,
                    "parameter_count": sum(
                        x.size
                        for x in eqx.filter(model, eqx.is_array).layers
                        for x in (x.weight, x.bias)
                    ),
                    "checkpoints": checkpoints,
                    "linear": outcome.get("linear_validation"),
                    "costs": outcome["costs"],
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "cells": rows,
                "interpretation": (
                    "Action error to demonstrations is not task-optimality; full/SPSA traces are "
                    "sampled training tasks, SPSA losses are perturbed."
                ),
            },
            indent=2,
        )
    )
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
