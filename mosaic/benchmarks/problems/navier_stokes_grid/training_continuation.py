# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic, protocol-bound continuation at completed optimizer-update boundaries."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np


class TrainingYield(Exception):
    """A saved allocation boundary, not a numerical failure or completed training."""

    def __init__(self, updates: int, active_time_s: float) -> None:
        super().__init__(f"training continuation required after {updates} updates")
        self.updates = updates
        self.active_time_s = active_time_s


def tree_bytes(tree: Any) -> bytes:
    """Serialize model and optimizer leaves without Python pickle."""
    stream = io.BytesIO()
    eqx.tree_serialise_leaves(stream, tree)
    return stream.getvalue()


def array_digest(array: np.ndarray | None) -> str | None:
    """Bind dtype, shape and contents of the actual training inputs."""
    if array is None:
        return None
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass
class TrainingContinuation:
    """Opt-in persistence; allocation limits never change the scientific schedule."""

    path: Path
    identity: dict[str, str]
    checkpoint_every: int = 100
    max_updates_per_allocation: int | None = None
    wall_limit_s: float | None = None
    active_time_s: float = 0.0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if any(
            not self.identity.get(key)
            for key in ("source_sha256", "image_sha256", "solver", "dataset_sha256")
        ):
            raise ValueError(
                "continuation requires source/image/solver/dataset identity"
            )
        if (
            self.checkpoint_every < 1
            or (
                self.max_updates_per_allocation is not None
                and self.max_updates_per_allocation < 1
            )
            or (
                self.wall_limit_s is not None
                and (not np.isfinite(self.wall_limit_s) or self.wall_limit_s <= 0)
            )
        ):
            raise ValueError(
                "continuation intervals and allocation limits must be positive"
            )

    def binding(self, context: dict) -> dict:
        """Normalize the complete protocol to its saved JSON representation."""
        return json.loads(
            json.dumps(
                {"version": 1, "identity": self.identity, "context": context},
                sort_keys=True,
            )
        )

    def load(self, template: Any, binding: dict) -> tuple[Any, dict] | None:
        """Refuse mismatched, corrupted or numerically failed continuation state."""
        if not self.path.exists():
            return None
        with zipfile.ZipFile(self.path) as archive:
            metadata = json.loads(archive.read("metadata.json"))
            payload = archive.read("state.eqx")
        if metadata["binding"] != binding:
            raise ValueError("training continuation protocol binding mismatch")
        if hashlib.sha256(payload).hexdigest() != metadata["state_sha256"]:
            raise ValueError("training continuation state checksum mismatch")
        if metadata["status"] == "failed":
            raise ValueError("numerically failed training cannot resume")
        if metadata["status"] not in {"ready", "complete"}:
            raise ValueError("unknown training continuation status")
        restored = eqx.tree_deserialise_leaves(io.BytesIO(payload), template)
        self.active_time_s = metadata["active_time_s"]
        return restored, metadata

    def save(self, state: Any, binding: dict, metadata: dict) -> None:
        """Publish one self-contained archive only after all bytes are durable."""
        payload = tree_bytes(state)
        metadata = {
            **metadata,
            "binding": binding,
            "state_sha256": hashlib.sha256(payload).hexdigest(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=self.path.name + ".", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                with zipfile.ZipFile(
                    stream, "w", compression=zipfile.ZIP_STORED
                ) as archive:
                    archive.writestr("metadata.json", json.dumps(metadata))
                    archive.writestr("state.eqx", payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.active_time_s = metadata["active_time_s"]
