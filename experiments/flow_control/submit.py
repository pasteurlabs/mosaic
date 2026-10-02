"""Submit immutable control gate cells through Slurm; no local numerical work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from runner import JobRegistry, JobSpec, Runner
from runner.transport import LocalTransport, SshTransport


def main() -> None:
    """Submit gate or validation jobs without importing numerical packages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--cells", default="gate-0,gate-1,gate-2,gate-3")
    parser.add_argument("--host", default="kander-login")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--assemble", action="store_true")
    parser.add_argument("--after", type=int, nargs="+")
    args = parser.parse_args()
    runner = Runner(
        JobRegistry(Path("mosaic-results/slurm-registry") / args.campaign.name),
        Path("/home/andrinr/slurm-runner/runner"),
        transport=LocalTransport() if args.host == "local" else SshTransport(args.host),
    )
    gpu = not (args.validate or args.report or args.assemble)
    cells = (
        ["validation"]
        if args.validate
        else ["report"]
        if args.report
        else ["assemble"]
        if args.assemble
        else args.cells.split(",")
    )
    for cell in cells:
        spec = JobSpec(
            name=f"m116-control-{cell}",
            cmd=["bash", str(args.campaign / "analyze.sh"), str(args.campaign)]
            if args.report
            else [
                "bash",
                str(args.campaign / "node.sh"),
                str(args.campaign),
                "validate"
                if args.validate
                else "assemble"
                if args.assemble
                else "train",
                cell,
            ],
            template="gpu" if gpu else "cpu",
            account="research",
            partition="dev",
            qos="rtx5090-pool" if gpu else "dev",
            time_limit="02:00:00",
            gpus="gpu:5090:1" if gpu else None,
            cpus=8,
            mem="64G",
            out_path=args.campaign / "results" / cell,
            depends_on=args.after or [],
            env={
                "PROJECT_ISOLATED_COMMAND": "1",
                "PROJECT_REPO_ROOT": str(args.campaign),
                "PYTHONUNBUFFERED": "1",
            },
            extra_sbatch=[
                f"--output={args.campaign}/logs/%x-%j.out",
                "--exclude=rtx03",
            ],
        )
        result = runner.submit(spec, cluster="kander")
        if result.job.state == "failed" or result.slurm_id is None:
            raise RuntimeError(result.job.reason)
        print(json.dumps({"cell": cell, "slurm_id": result.slurm_id}), flush=True)


if __name__ == "__main__":
    main()
