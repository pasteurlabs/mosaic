# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node-local working directories and explicit, coarse-grained artifact export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

_LOCAL_FILESYSTEMS = {"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "overlay", "tmpfs"}


def _unescape_mount(value: str) -> str:
    for encoded, decoded in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    return value


def filesystem_info(path: Path, mountinfo: str | None = None) -> dict:
    """Resolve bind mounts and return the longest matching Linux mount entry."""
    resolved = path.expanduser().resolve()
    if mountinfo is None:
        mountinfo = Path("/proc/self/mountinfo").read_text()
    matches = []
    for line in mountinfo.splitlines():
        before, after = line.split(" - ", 1)
        fields, details = before.split(), after.split()
        mount = Path(_unescape_mount(fields[4]))
        if resolved == mount or mount in resolved.parents:
            matches.append((len(mount.parts), mount, details[0], details[1]))
    if not matches:
        raise ValueError(f"cannot establish filesystem for {resolved}")
    _, mount, kind, device = max(matches, key=lambda x: x[0])
    return {
        "path": str(resolved),
        "mount": str(mount),
        "filesystem": kind,
        "device": device,
    }


def require_local(path: Path) -> dict:
    """Fail closed for network or unknown working storage, before any writes."""
    info = filesystem_info(path)
    if info["filesystem"] not in _LOCAL_FILESYSTEMS:
        raise ValueError(
            f"node-local storage required: {info}; stage this path on local scratch"
        )
    return info


def atomic_json(path: Path, value: dict) -> None:
    """Atomically write a small local metadata file without per-step fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def publish_bundle(source: Path, destination: Path) -> dict:
    """Export a quiescent local directory using one sequential archive write.

    The caller owns consistency: stop training or pass an immutable checkpoint
    snapshot. Hash while streaming local files once to shared storage, atomically
    rename, then publish the small completion record. No destination tree walk
    or remote checksum read is performed. Hash verification happens on restore.
    """
    require_local(source)
    if destination.suffix != ".tar":
        raise ValueError("destination must be an uncompressed .tar archive")
    if (
        source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
    ):
        raise ValueError("archive destination must be outside the source tree")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()

    def keep_artifact(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        excluded = {"compilation-cache", "__pycache__", "mmap-cache"}
        if excluded.intersection(Path(member.name).parts) or member.name.endswith(
            (".tmp", ".lock")
        ):
            return None
        return member

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    count = 0
    try:
        with temporary.open("wb", buffering=8 * 1024 * 1024) as dst:

            class HashingWriter:
                def write(self, block: bytes) -> int:
                    nonlocal count
                    written = dst.write(block)
                    if written != len(block):
                        raise OSError("short archive write")
                    digest.update(block)
                    count += written
                    return written

            # Stream from immutable local files directly into one buffered shared
            # file. No additional dataset-sized local tar is written or reread.
            with tarfile.open(
                fileobj=HashingWriter(), mode="w|", bufsize=8 * 1024 * 1024
            ) as handle:
                handle.add(
                    source, arcname="artifacts", recursive=True, filter=keep_artifact
                )
            dst.flush()
            os.fsync(dst.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    record = {
        "sha256": digest.hexdigest(),
        "bytes": count,
        "archive": destination.name,
        "format": "tar",
        "root": "artifacts",
    }
    atomic_json(destination.with_suffix(".json"), record)
    return record


def restore_bundle(source: Path, destination: Path) -> dict:
    """Read an archive once from shared storage, verify locally, then extract."""
    require_local(destination)
    if destination.exists():
        raise ValueError("restore destination must not already exist")
    record = json.loads(source.with_suffix(".json").read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(
        prefix="operator-restore-", dir=destination.parent
    ) as directory:
        archive = Path(directory) / "bundle.tar"
        with source.open("rb") as src, archive.open("wb") as dst:
            for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                digest.update(block)
                dst.write(block)
        if (
            archive.stat().st_size != record["bytes"]
            or digest.hexdigest() != record["sha256"]
        ):
            raise ValueError("archive checksum mismatch")
        unpack = Path(directory) / "unpacked"
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                target = unpack / member.name
                if (
                    Path(member.name).is_absolute()
                    or ".." in Path(member.name).parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError(f"unsupported archive member: {member.name}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with handle.extractfile(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        (unpack / "artifacts").replace(destination)
    return record


def main() -> None:
    """Check a working directory or explicitly transfer a completed bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("paths", nargs="+", type=Path)
    for name in ("publish", "restore"):
        command = sub.add_parser(name)
        command.add_argument("source", type=Path)
        command.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        print(json.dumps([require_local(p) for p in args.paths]))
    else:
        fn = publish_bundle if args.command == "publish" else restore_bundle
        print(json.dumps(fn(args.source, args.destination)))


if __name__ == "__main__":
    main()
