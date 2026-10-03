"""CPU-only selection for the final INS search; no survivor-only averages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _write_once(path: Path, value: Any) -> None:
    content = json.dumps(value, indent=2, sort_keys=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"refusing to change frozen selection: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content)
    temporary.replace(path)


def _tie(candidate: dict) -> tuple:
    total = candidate["updates"] + (candidate.get("pretrain") or {}).get("updates", 0)
    return total, candidate["lr"], candidate["unroll"], candidate["candidate_id"]


def _recipe(candidate: dict) -> dict:
    return {k: candidate[k] for k in ("lr", "unroll", "updates", "pretrain")}


def select(campaign: Path, stage: str) -> dict:
    """Check the frozen registry, then rank complete eligible candidates on validation."""
    if stage not in {"select_extension", "select_final"}:
        raise ValueError("unknown selection stage")
    plan = json.loads((campaign / "plan.json").read_text())
    registry = json.loads((campaign / "report-input.json").read_text())
    seeds = plan.get("tuning_model_seeds", [0, 1, 2])
    ic_seeds = plan.get("validation_ic_seeds", list(range(10000, 10016)))
    if seeds != [0, 1, 2] or ic_seeds != list(range(10000, 10016)):
        raise ValueError("selection split differs from final protocol")
    candidates = list(plan["candidates"])
    if stage == "select_extension":
        candidates = [
            c
            for c in candidates
            if c["arm"] in {"full", "stopped"}
            and c["pretrain"] is None
            and c["updates"] == 1000
        ]
        if len(candidates) != 18:
            raise ValueError("extension requires all eighteen cold candidates")
    else:
        extensions = json.loads((campaign / "extensions.json").read_text())
        if set(extensions) != {"full", "stopped"} or any(
            len(v) != 2 for v in extensions.values()
        ):
            raise ValueError(
                "extension manifest must contain two candidates per recurrent arm"
            )
        candidates += [c for group in extensions.values() for c in group]
        if len(candidates) != 32:
            raise ValueError(
                "final selection requires the frozen thirty-two candidates"
            )
    ids = [c["candidate_id"] for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate identity")
    base = plan["base_payload"]
    scores = []
    dataset_hashes = set()
    for candidate in candidates:
        rows = [
            r
            for r in registry["validation"]
            if r["candidate"]["candidate_id"] == candidate["candidate_id"]
        ]
        if sorted(r["model_seed"] for r in rows) != seeds or any(
            r["candidate"] != candidate for r in rows
        ):
            raise ValueError(
                f"missing/duplicate/mismatched registry entries: {candidate['candidate_id']}"
            )
        values, failures, provenance = [], [], []
        for row in rows:
            path = Path(row["path"]) / "outcome.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing result; resolve infrastructure before selection: {path}"
                )
            result = json.loads(path.read_text())
            provenance.append(
                {
                    "model_seed": row["model_seed"],
                    "path": str(path),
                    "outcome_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            if not result.get("completed") or not result.get("admitted"):
                failures.append(
                    {
                        "model_seed": row["model_seed"],
                        "reason": result.get("failure", "incomplete or inadmissible"),
                    }
                )
                continue
            training = result["training"]
            if (
                result["arm"] != candidate["arm"]
                or result["model_seed"] != row["model_seed"]
                or result["evaluation_seeds"] != ic_seeds
                or any(
                    training[k] != candidate[v]
                    for k, v in [
                        ("lr", "lr"),
                        ("unroll", "unroll"),
                        ("max_updates", "updates"),
                    ]
                )
            ):
                raise ValueError(f"candidate/split identity mismatch: {path}")
            identity = result["identity"]
            if (
                any(
                    identity[k] != base[k]
                    for k in ["solver", "image", "image_sha256", "source_sha256"]
                )
                or identity["physics"] != base["run"]["physics"]
                or identity["domain_extent"] != base.get("domain_extent", 2 * math.pi)
            ):
                raise ValueError(f"source/image/physics mismatch: {path}")
            if bool(candidate["pretrain"]) != bool(result.get("initial_model_sha256")):
                raise ValueError(f"warm-start identity mismatch: {path}")
            dataset_hashes.add(result["training_dataset_sha256"])
            value = float(result["mean_rollout_error"])
            if not math.isfinite(value) or value < 0:
                failures.append(
                    {
                        "model_seed": row["model_seed"],
                        "reason": "invalid rollout metric",
                    }
                )
            else:
                values.append(value)
        scores.append(
            {
                "candidate": candidate,
                "eligible": not failures and len(values) == 3,
                "mean_rollout_error": statistics.mean(values)
                if not failures and len(values) == 3
                else None,
                "seed_values": values,
                "failures": failures,
                "provenance": provenance,
            }
        )
    if len(dataset_hashes) != 1:
        raise ValueError(
            "successful candidates do not share one frozen validation dataset"
        )
    decision = {}
    for arm in (
        ["full", "stopped"]
        if stage == "select_extension"
        else ["full", "stopped", "supervised"]
    ):
        ranked = sorted(
            [r for r in scores if r["eligible"] and r["candidate"]["arm"] == arm],
            key=lambda r: (r["mean_rollout_error"], *_tie(r["candidate"])),
        )
        count = 2 if stage == "select_extension" else 1
        if len(ranked) < count:
            _write_once(
                campaign / f"{stage}-scores.json",
                {"scores": scores, "blocked_arm": arm},
            )
            raise ValueError(
                f"not enough eligible candidates for {arm}; no replacement search"
            )
        if stage == "select_extension":
            decision[arm] = [
                {
                    **r["candidate"],
                    "updates": 3000,
                    "candidate_id": r["candidate"]["candidate_id"].replace(
                        "u1000", "u3000"
                    ),
                }
                for r in ranked[:2]
            ]
        else:
            decision[arm] = ranked[0]["candidate"]
    if stage == "select_final":
        full, stopped = decision["full"], decision["stopped"]
        decision["matched_stopped"] = (
            stopped
            if _recipe(full) == _recipe(stopped)
            else {
                **full,
                "arm": "stopped",
                "candidate_id": "matched-stopped-" + full["candidate_id"],
            }
        )
    _write_once(
        campaign / f"{stage}-scores.json",
        {
            "scores": scores,
            "decision": decision,
            "validation_ic_seeds": ic_seeds,
            "model_seeds": seeds,
            "dataset_sha256": next(iter(dataset_hashes)),
        },
    )
    _write_once(
        campaign
        / ("extensions.json" if stage == "select_extension" else "selection.json"),
        decision,
    )
    return decision


def main() -> None:
    """Run registered selection on the Slurm CPU allocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["select_extension", "select_final"], required=True
    )
    args = parser.parse_args()
    print(json.dumps(select(args.campaign, args.stage)))


if __name__ == "__main__":
    main()
