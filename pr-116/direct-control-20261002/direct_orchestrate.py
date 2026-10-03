"""Select only on registered development tasks, then run frozen confirmation jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path


def outcome(path: Path) -> dict:
    """Read metrics only on the login node; preserve missing results as failures."""
    try:
        with tarfile.open(path) as archive:
            return json.load(archive.extractfile("./outcome.json"))
    except (OSError, KeyError, ValueError) as error:
        return {"completed": False, "admitted": False, "failure": repr(error)}


def select_settings(
    results: list[dict], settings: list[dict], expected_seeds: list[int]
) -> dict:
    """Require every registered task at120s; resolve exact ties by manifest order."""
    if sorted(row.get("task_seed", -1) for row in results) != sorted(expected_seeds):
        raise ValueError("development task identities do not match the frozen split")
    scores, selected = [], {}
    for setting in settings:
        values, failures = [], []
        for result in results:
            found = [
                row
                for row in result.get("settings", [])
                if row["setting_id"] == setting["setting_id"]
            ]
            if (
                not result.get("common_admitted", result.get("admitted"))
                or len(found) != 1
            ):
                failures.append(result["task_seed"])
                continue
            run = found[0]
            if run.get("method") != setting["method"] or any(
                run.get("optimizer_settings", {}).get(key) != value
                for key, value in setting["optimizer_settings"].items()
            ):
                failures.append(result["task_seed"])
                continue
            snapshots = [
                row
                for row in run.get("snapshots", [])
                if row["budget_kind"] == "wall_time_s" and row["budget"] == 120
            ]
            if (
                run.get("failure")
                or len(snapshots) != 1
                or not snapshots[0].get("available")
                or not snapshots[0].get("admitted")
            ):
                failures.append(result["task_seed"])
                continue
            value = snapshots[0].get("fine_objective")
            if value is None or not float("-inf") < value < float("inf"):
                failures.append(result["task_seed"])
                continue
            values.append(value)
        mean = (
            statistics.mean(values)
            if not failures and len(values) == len(expected_seeds)
            else None
        )
        scores.append(
            {
                **setting,
                "eligible": mean is not None,
                "mean_fine_objective_120s": mean,
                "failed_task_seeds": failures,
                "task_values": values,
            }
        )
    for method in ("adam", "spsa", "powell"):
        candidates = [
            row for row in scores if row["method"] == method and row["eligible"]
        ]
        if candidates:
            best = min(candidates, key=lambda row: row["mean_fine_objective_120s"])
            selected[method] = next(
                setting
                for setting in settings
                if setting["setting_id"] == best["setting_id"]
            )
    return {
        "selected": selected,
        "development_scores": scores,
        "development_task_seeds": expected_seeds,
        "primary_budget_seconds": 120,
        "rule": "minimum mean fine objective on every registered development task; exact ties retain manifest order",
    }


def main() -> None:
    """Continue the registered experiment without retuning on confirmation results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign
    manifest = json.loads((campaign / "direct-plan.json").read_text())
    active = {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "REQUEUED",
        "SUSPENDED",
    }

    def status(state: str, **details: object) -> None:
        value = {"state": state, "updated_at": datetime.now(UTC).isoformat(), **details}
        (campaign / "direct-status.json").write_text(json.dumps(value, indent=2))
        print(json.dumps(value), flush=True)

    def wait(jobs: list[int], stage: str) -> dict:
        while True:
            response = subprocess.run(
                [
                    "sacct",
                    "-X",
                    "-n",
                    "-P",
                    "-j",
                    ",".join(map(str, jobs)),
                    "--format=JobIDRaw,State%30",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if response.returncode:
                status("accounting_unavailable", stage=stage, error=response.stderr)
                time.sleep(55)
                continue
            states = {
                line.split("|")[0]: line.split("|")[1].split()[0].rstrip("+")
                for line in response.stdout.splitlines()
                if line
            }
            if all(
                str(job) in states and states[str(job)] not in active for job in jobs
            ):
                return states
            status(stage, jobs=states)
            time.sleep(55)

    def submit(arguments: list[str], record: str) -> list[dict]:
        response = subprocess.run(
            [
                sys.executable,
                str(campaign / "submit.py"),
                "--host",
                "local",
                "--campaign",
                str(campaign),
                *arguments,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        (campaign / record).write_text(response.stdout + response.stderr)
        if response.returncode:
            status("submission_failed", log=record)
            raise RuntimeError(response.stderr)
        return [
            json.loads(line)
            for line in response.stdout.splitlines()
            if line.startswith("{")
        ]

    wait(manifest["development_jobs"], "development_running")
    results = [
        outcome(campaign / "results" / cell / "results.tar")
        for cell in manifest["development_cells"]
    ]
    # Preserve identity for an entirely missing task without fabricating a result.
    for value, seed in zip(results, manifest["development_task_seeds"], strict=True):
        value.setdefault("task_seed", seed)
    selection = select_settings(
        results, manifest["settings"], manifest["development_task_seeds"]
    )
    selection.update(
        source_sha256=manifest["base_config"]["source_sha256"],
        image_sha256=manifest["base_config"]["image_sha256"],
        test_task_seeds=manifest["test_task_seeds"],
    )
    selection_path = campaign / "selections.json"
    if selection_path.exists():
        raise RuntimeError(
            "refusing to overwrite a frozen selection; inspect existing continuation state"
        )
    selection_path.write_text(json.dumps(selection, indent=2))
    digest = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    submit(["--report"], "development-report-submission.log")
    if set(selection["selected"]) != {"adam", "spsa", "powell"}:
        status("blocked_no_eligible_configuration", selections=selection)
        return
    cells = []
    for seed in manifest["test_task_seeds"]:
        cell = f"test-{seed}"
        cells.append(cell)
        config = {
            **manifest["base_config"],
            "phase": "direct_compare",
            "comparison_stage": "test",
            "task_seed": seed,
            "settings": list(selection["selected"].values()),
            "selection_sha256": digest,
        }
        (campaign / "configs" / f"{cell}.json").write_text(json.dumps(config, indent=2))
    launched = submit(["--cells", ",".join(cells)], "test-submissions.jsonl")
    states = wait([row["slurm_id"] for row in launched], "confirmation_running")
    reports = submit(["--report"], "final-report-submission.log")
    report_states = wait([row["slurm_id"] for row in reports], "report_running")
    status(
        "finished",
        test_jobs=states,
        report_jobs=report_states,
        selection_sha256=digest,
        note="Inspect numerical admission and paired report before claiming an advantage.",
    )


if __name__ == "__main__":
    main()
