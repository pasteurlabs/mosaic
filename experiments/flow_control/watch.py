"""Watch only this campaign's jobs on the login node, then submit a CPU report."""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    """Keep persisted status and cancel dependent jobs if shared data cannot pass."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign
    plan = json.loads((campaign / "submissions.json").read_text())
    jobs = {str(value): name for name, value in plan.items()}
    active = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"}
    prerequisite = {
        str(value)
        for name, value in plan.items()
        if name.startswith("prepare-") or name == "assemble"
    }
    report_id = None
    while True:
        result = subprocess.run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                ",".join(jobs),
                "--format=JobIDRaw,State%30",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            print(result.stderr, flush=True)
            time.sleep(55)
            continue
        states = {
            line.split("|")[0]: line.split("|")[1].split()[0].rstrip("+")
            for line in result.stdout.splitlines()
            if line
        }
        failed_prerequisite = {
            job: states[job]
            for job in prerequisite
            if job in states and states[job] not in active | {"COMPLETED"}
        }
        if failed_prerequisite:
            pending = [
                job
                for job, state in states.items()
                if state == "PENDING" and not jobs[job].startswith("prepare-")
            ]
            if pending:
                subprocess.run(["scancel", *pending], check=True)
            state = "blocked_by_dataset_failure"
        elif all(job in states and states[job] not in active for job in jobs):
            if report_id is None:
                launch = subprocess.run(
                    [
                        sys.executable,
                        str(campaign / "submit.py"),
                        "--host",
                        "local",
                        "--report",
                        "--campaign",
                        str(campaign),
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                (campaign / "report-submission.log").write_text(
                    launch.stdout + launch.stderr
                )
                if launch.returncode:
                    state = "report_submission_failed"
                else:
                    report_id = str(
                        json.loads(launch.stdout.strip().splitlines()[-1])["slurm_id"]
                    )
                    jobs[report_id] = "report"
                    state = "report_submitted"
            else:
                state = (
                    "jobs_finished_report_complete"
                    if states[report_id] == "COMPLETED"
                    else "report_failed"
                )
        else:
            state = "running_or_waiting"
        value = {
            "state": state,
            "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "jobs": {
                jobs[job]: {"slurm_id": int(job), "state": states.get(job, "UNKNOWN")}
                for job in jobs
            },
            "note": "Slurm completion alone does not establish numerical admission or a learning advantage.",
        }
        (campaign / "control-status.json").write_text(json.dumps(value, indent=2))
        print(json.dumps(value), flush=True)
        if state in {
            "blocked_by_dataset_failure",
            "report_submission_failed",
            "jobs_finished_report_complete",
            "report_failed",
        }:
            return
        time.sleep(55)


if __name__ == "__main__":
    main()
