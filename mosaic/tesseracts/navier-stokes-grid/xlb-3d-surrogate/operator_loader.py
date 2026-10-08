# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map uncompressed local NPZ members directly, without a second dataset copy."""

from __future__ import annotations

import fcntl
import json
import struct
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from operator_dataset import fingerprint
from operator_storage import atomic_json, require_local


def prepare_cache(dataset: Path, cache: Path) -> dict:
    """Serialize local cache creation when several GPU workers share a node."""
    require_local(dataset)
    require_local(cache)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _prepare_cache(dataset, cache)


def _prepare_cache(dataset: Path, cache: Path) -> dict:
    """Index local shard headers once; never unpack or open ZIP files per update."""
    require_local(dataset)
    require_local(cache)
    markers = sorted(dataset.glob("*/float64/shard-*.json"))
    if not markers:
        raise ValueError(f"no completed float64 shards in {dataset}")
    records = [json.loads(p.read_text()) for p in markers]
    by_case = defaultdict(list)
    for record in records:
        by_case[record["case"]["id"]].append(record)
    for rows in by_case.values():
        rows.sort(key=lambda row: row["start"])
        stop = 0
        for row in rows:
            if (
                row["case"] != rows[0]["case"]
                or row["start"] != stop
                or row["stop"] <= stop
            ):
                raise ValueError("inconsistent or incomplete dataset shard ranges")
            stop = row["stop"]
        if "samples" in rows[0]["case"] and stop != rows[0]["case"]["samples"]:
            raise ValueError("incomplete dataset case")
    identity = fingerprint(
        [
            {
                "cache_format": 2,
                "signature": r["signature"],
                "sha256": r["sha256"],
                "start": r["start"],
                "stop": r["stop"],
            }
            for r in records
        ]
    )
    index_path = cache / "index.json"
    if index_path.exists():
        result = json.loads(index_path.read_text())
        if result["identity"] != identity:
            raise ValueError("local data cache identity mismatch")
        if not all((dataset / s["archive"]).exists() for s in result["shards"]):
            raise ValueError("incomplete local mmap cache")
        return result
    cache.mkdir(parents=True, exist_ok=True)
    shards = []
    for marker, record in zip(markers, records, strict=True):
        source = marker.with_suffix(".npz")
        # np.savez writes ZIP_STORED members. Read only the small headers here;
        # training accesses the float32 payload directly through a read-only mmap.
        # Generation hashes each complete shard; restore verifies the bundle hash.
        with zipfile.ZipFile(source) as archive:
            info = archive.getinfo("velocity.npy")
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError("direct mapping requires uncompressed NPZ shards")
            with source.open("rb") as handle:
                handle.seek(info.header_offset + 26)
                name_length, extra_length = struct.unpack("<HH", handle.read(4))
            payload = info.header_offset + 30 + name_length + extra_length
            with archive.open(info) as member:
                version = np.lib.format.read_magic(member)
                if version != (1, 0):
                    raise ValueError(f"unsupported NPY header version: {version}")
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    member
                )
                header_bytes = member.tell()
            if fortran_order or dtype != np.dtype("float32"):
                raise ValueError("velocity must be C-contiguous float32")
            if header_bytes + int(np.prod(shape)) * dtype.itemsize != info.file_size:
                raise ValueError("velocity payload size mismatch")
            offset = payload + header_bytes
        with np.load(source, allow_pickle=False) as arrays:
            valid = arrays["valid"].astype(bool)
            splits = arrays["split"].astype(str)
            parents = arrays["parent_id"].astype(str)
            snapshots = arrays["snapshot_steps"].astype(int)
            families = (
                arrays["family"].astype(str).tolist()
                if "family" in arrays
                else ["unknown"] * len(valid)
            )
        mapped = np.memmap(source, mode="r", dtype=dtype, offset=offset, shape=shape)
        if (
            mapped.shape[:2] != (len(valid), len(snapshots))
            or mapped.dtype != np.float32
            or mapped.shape[2:] != (record["case"]["N"],) * 3 + (3,)
        ):
            raise ValueError(f"invalid velocity array in {source}")
        shards.append(
            {
                "archive": str(source.relative_to(dataset)),
                "offset": offset,
                "shape": list(shape),
                "dtype": str(dtype),
                "case": record["case"],
                "source_sha256": record["sha256"],
                "snapshot_steps": snapshots.tolist(),
                "valid": valid.tolist(),
                "split": splits.tolist(),
                "parent_id": parents.tolist(),
                "family": families,
            }
        )
    result = {"identity": identity, "shards": shards, "archives_read": len(shards)}
    atomic_json(index_path, result)
    return result


class WindowDataset:
    """Shape-bucketed, split-safe windows with no dataset reads on shared storage."""

    def __init__(self, dataset: Path, cache: Path) -> None:
        self.index = prepare_cache(dataset, cache)
        self.shards = self.index["shards"]
        self.arrays = [
            np.memmap(
                dataset / s["archive"],
                mode="r",
                dtype=s["dtype"],
                offset=s["offset"],
                shape=tuple(s["shape"]),
            )
            for s in self.shards
        ]
        self.cases = {s["case"]["id"]: s["case"] for s in self.shards}
        self._choices: dict = {}

    def choices(self, case_id: str, split: str, updates: int, span: int) -> list:
        """Locate windows with exactly the requested elapsed teacher steps."""
        key = (case_id, split, updates, span)
        if key in self._choices:
            return self._choices[key]
        if updates < 1 or span < 1:
            raise ValueError("positive updates and span required")
        choices = []
        for shard_index, shard in enumerate(self.shards):
            if shard["case"]["id"] != case_id:
                continue
            positions = {step: i for i, step in enumerate(shard["snapshot_steps"])}
            windows = []
            for start in shard["snapshot_steps"]:
                times = [start + j * span for j in range(updates + 1)]
                if all(t in positions for t in times):
                    windows.append([positions[t] for t in times])
            for row, (valid, row_split) in enumerate(
                zip(shard["valid"], shard["split"], strict=True)
            ):
                if valid and row_split == split and windows:
                    choices.append((shard_index, row, windows))
        self._choices[key] = choices
        return choices

    def sample(
        self,
        case_id: str,
        split: str,
        batch_size: int,
        updates: int,
        span: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw parents uniformly, then windows; return contiguous float32 arrays."""
        choices = self.choices(case_id, split, updates, span)
        if not choices:
            raise ValueError(
                f"no {split} windows for {case_id}, updates={updates}, span={span}"
            )
        fields = []
        for selected in rng.integers(0, len(choices), size=batch_size):
            shard, row, windows = choices[selected]
            window = windows[int(rng.integers(len(windows)))]
            fields.append(np.asarray(self.arrays[shard][row, window]))
        array = np.stack(fields)
        return np.ascontiguousarray(array[:, 0]), np.ascontiguousarray(array[:, 1:])

    def endpoints(self, case_id: str, split: str, limit: int = 4) -> tuple:
        """Balance fixed held-out parents across families without loading unused rows."""
        groups = defaultdict(list)
        pairs, parents = [], []
        for shard_index, shard in enumerate(self.shards):
            if shard["case"]["id"] != case_id:
                continue
            for row, (valid, row_split) in enumerate(
                zip(shard["valid"], shard["split"], strict=True)
            ):
                if valid and row_split == split:
                    family = shard.get("family", ["unknown"] * len(shard["valid"]))[row]
                    groups[family].append((shard["parent_id"][row], shard_index, row))
        for rows in groups.values():
            rows.sort()
        for rank in range(max(map(len, groups.values()), default=0)):
            for family in sorted(groups):
                if rank < len(groups[family]) and len(pairs) < limit:
                    parent, shard_index, row = groups[family][rank]
                    pairs.append(np.asarray(self.arrays[shard_index][row, [0, -1]]))
                    parents.append(parent)
        if not pairs:
            raise ValueError(f"no {split} endpoint pairs for {case_id}")
        array = np.stack(pairs)
        return (
            np.ascontiguousarray(array[:, 0]),
            np.ascontiguousarray(array[:, 1]),
            parents,
        )
