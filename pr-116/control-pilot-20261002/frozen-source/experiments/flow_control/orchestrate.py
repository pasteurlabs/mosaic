"""Continue from frozen control gates to a frozen development pilot on the login node."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def outcome(path: Path) -> dict:
    """Read metrics only; numerical work is never performed by this controller."""
    with tarfile.open(path) as archive:
        member = next(
            m for m in archive.getmembers() if m.name.lstrip("./") == "outcome.json"
        )
        return json.load(archive.extractfile(member))


def main() -> None:
    """Launch the pilot only after every explicit validation and task gate passes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    status_path = args.manifest.parent / "control-status.json"

    def status(state: str, **details: object) -> None:
        value = {"state": state, **details}
        status_path.write_text(json.dumps(value, indent=2))
        print(json.dumps(value), flush=True)

    ids = manifest["gate_jobs"] + [manifest["pilot_validation_job"]]
    deadline = time.monotonic() + 12 * 3600
    active = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"}
    while True:
        raw = subprocess.check_output(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                ",".join(map(str, ids)),
                "--format=JobIDRaw,State%30",
            ],
            text=True,
        )
        states = {
            line.split("|")[0]: line.split("|")[1].split()[0].rstrip("+")
            for line in raw.splitlines()
            if line
        }
        failures = {k: v for k, v in states.items() if v not in active | {"COMPLETED"}}
        if failures:
            status("blocked_by_job_failure", jobs=failures)
            return
        if all(states.get(str(job)) == "COMPLETED" for job in ids):
            break
        if time.monotonic() > deadline:
            status("timed_out_waiting", jobs=states)
            return
        status("waiting_for_gates", jobs=states)
        time.sleep(55)
    gate = Path(manifest["gate_campaign"])
    results = {
        cell: outcome(gate / "results" / cell / "results.tar")
        for cell in manifest["gate_cells"]
    }
    if not all(value.get("passed") is True for value in results.values()):
        status("blocked_by_numerical_gate", outcomes=results)
        return
    campaign = Path(manifest["pilot_campaign"])
    launch = subprocess.run(
        [
            sys.executable,
            str(campaign / "submit.py"),
            "--host",
            "local",
            "--campaign",
            str(campaign),
            "--cells",
            ",".join(manifest["pilot_cells"]),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    (campaign / "submissions.jsonl").write_text(launch.stdout)
    (campaign / "submission-errors.log").write_text(launch.stderr)
    status(
        "pilot_submitted" if launch.returncode == 0 else "pilot_submission_failed",
        submissions=launch.stdout,
        errors=launch.stderr,
    )
    if launch.returncode:
        raise RuntimeError("Pilot submission failed; inspect preserved logs")


if __name__ == "__main__":
    main()
