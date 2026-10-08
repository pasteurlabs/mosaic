# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize generated shards and project compute/storage for a larger campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root: Path, manifest: dict, target_samples: int = 1024) -> dict:
    """Aggregate unique shards, retaining missing cases and precision failures."""
    rows = []
    for case in manifest["cases"]:
        for precision in ("float32", "float64"):
            folder = root / case["id"] / precision
            shards = [
                json.loads(p.read_text()) for p in sorted(folder.glob("shard-*.json"))
            ]
            failures = [
                json.loads(p.read_text()) for p in sorted(folder.glob("failure-*.json"))
            ]
            if not shards:
                if precision == manifest.get("precision", "float64") or failures:
                    rows.append(
                        {
                            "case_id": case["id"],
                            "N": case["N"],
                            "precision": precision,
                            "status": "failed" if failures else "missing",
                            "failures": failures,
                        }
                    )
                continue
            signatures = {s["signature"] for s in shards}
            if len(signatures) != 1:
                raise ValueError(f"mixed shard identities in {folder}")
            expected = {
                (start, min(start + shards[0]["shard_size"], case["samples"]))
                for start in range(0, case["samples"], shards[0]["shard_size"])
            }
            actual = {(s["start"], s["stop"]) for s in shards}
            count = sum(s["count"] for s in shards)
            valid = sum(s["valid_count"] for s in shards)
            compute = sum(s["compute_seconds"] for s in shards)
            write = sum(s["write_seconds"] for s in shards)
            host = sum(s["ic_seconds"] + s["transfer_seconds"] for s in shards)
            size = sum(s["bytes"] for s in shards)
            # Each shard repeats case-level checks; deduplicate by their content.
            check_sets = {
                json.dumps(s["trajectory_checks"], sort_keys=True) for s in shards
            }
            precision_checks = [
                p for check in check_sets for p in json.loads(check)["final_vs_float64"]
            ]
            errors = [p["relative_l2"] for p in precision_checks if p["finite"]]
            parity = [
                p["max_abs"]
                for check in check_sets
                for p in json.loads(check)["native_vs_api"]
            ]
            workers = [json.loads(p.read_text()) for p in folder.glob("worker-*.json")]
            peaks = [
                s.get("device_memory", {}).get("peak_bytes_in_use", 0)
                for s in shards
                if s.get("device_memory")
            ]
            rows.append(
                {
                    "case_id": case["id"],
                    "N": case["N"],
                    "precision": precision,
                    "status": "complete" if actual == expected else "partial",
                    "physics": {
                        k: case[k]
                        for k in ("viscosity", "dt", "steps", "domain_extent")
                    },
                    "samples": count,
                    "valid": valid,
                    "invalid": count - valid,
                    "batch_sizes": sorted({s["batch_size"] for s in shards}),
                    "devices": sorted({w["device_kind"] for w in workers}),
                    "compile_seconds": sum(w["compile_seconds"] for w in workers),
                    "invocation_seconds": sum(w["invocation_seconds"] for w in workers),
                    "compute_seconds": compute,
                    "host_seconds": host,
                    "write_seconds": write,
                    "trajectories_per_compute_second": count / compute,
                    "peak_device_gib": max(peaks, default=0) / 2**30,
                    "native_parity_max_abs": max(parity),
                    "float64_comparison_samples": len(precision_checks),
                    "float64_comparison_nonfinite": sum(
                        not p["finite"] for p in precision_checks
                    ),
                    "float64_relative_l2_median": float(np.median(errors))
                    if errors
                    else None,
                    "float64_relative_l2_max": max(errors, default=None),
                    "bytes_per_trajectory": size / count,
                    "projection": {
                        "samples": target_samples,
                        "compute_gpu_hours": target_samples / count * compute / 3600,
                        # Conservative serial sum; writes can overlap computation.
                        "compute_plus_host_write_hours": target_samples
                        / count
                        * (compute + host + write)
                        / 3600,
                        "storage_gib": target_samples / count * size / 2**30,
                        "excludes": "startup, compilation, parity checks, queueing; same horizon and snapshot policy",
                    },
                }
            )
    return {
        "cases": rows,
        "scope": "data generation only; no training throughput claim",
        "precision_note": "Measured float32/float64 differences do not certify every IC or longer horizon.",
    }


def main() -> None:
    """Write a JSON report from case and shard metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-samples", type=int, default=1024)
    args = parser.parse_args()
    if args.target_samples < 1:
        parser.error("target-samples must be positive")
    report = summarize(
        args.results, json.loads(args.manifest.read_text()), args.target_samples
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in report["cases"]:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
