# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a selected local training checkpoint without optimizer state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from operator_dataset import file_hash
from operator_storage import require_local


def export(checkpoint: Path, destination: Path) -> dict:
    """Preserve architecture and training identity in a compact inference artifact."""
    require_local(checkpoint)
    require_local(destination)
    with np.load(checkpoint, allow_pickle=False) as data:
        training = json.loads(str(data["metadata"]))
        metadata = {
            **training["identity"],
            "selected_step": training["step"],
            "validation_score": training["best_score"],
            "checkpoint_sha256": file_hash(checkpoint),
        }
        params = {
            k.removeprefix("param_"): data[k]
            for k in data.files
            if k.startswith("param_")
        }
        if not params or not all(np.isfinite(v).all() for v in params.values()):
            raise ValueError("empty or nonfinite operator checkpoint")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                **params,
                metadata=np.asarray(json.dumps(metadata, allow_nan=False)),
            )
        temporary.replace(destination)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.destination)))
