"""Submit frozen solver-loop cells through the shared slurm-runner toolkit.

Requires slurm-runner on PYTHONPATH. The campaign contains source.tar and
node.sh, uploaded before submission; images.json maps solver slugs to built
SquashFS images. Submission imports no numerical libraries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

from runner import JobRegistry, JobSpec, Runner
from runner.transport import SshTransport


def _remote(host: str, argv: list[str], payload: str | None = None) -> str:
    result = subprocess.run(
        ["ssh", host, shlex.join(argv)],
        input=payload,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def protocol(
    n: int, k0: float, unroll: int, updates: int, seed: int, confirm: bool
) -> dict:
    """Freeze the physical task and all three arms before observing outcomes."""
    return {
        "ic": {"name": "multimode", "seed": 0},
        "physics": {"N": n, "nu": 0.001, "dt": 0.02, "steps": 4},
        "dataset": {
            "reference_kind": "solver_self_refined",
            "reference_factor": 3,
            "reference_temporal_factor": 3,
            "reference_audit_factor": 3,
            "reference_audit_temporal_factor": 6,
            "prefix_audit_seeds": [0, 1000 if confirm else 100],
            "prefix_audit_frames": [1, 8, 24, 48],
            "reference_convergence_tolerance": 0.005,
            "train_seeds": list(range(16 if confirm else 8)),
            "test_seeds": list(range(1000, 1008)) if confirm else list(range(100, 104)),
            "train_frames": 24,
            "k0": k0,
            "sigma_k": 1.0 if k0 > 2 else 0.5,
            "amplitude": 0.5,
        },
        "training": {
            "max_updates": updates,
            "unroll": unroll,
            "loss_mode": "mean",
            "solver_loss_weight": 0.0,
            "loss_normalization": "solver_baseline",
            "loss_scale_floor": 1e-6,
            "lr": 1e-4,
            "clip_norm": 5.0,
            "architecture": "periodic_residual_cnn",
            "hidden_channels": 32,
            "kernel_size": 5,
            "seed": 2026,
            "model_seeds": [seed],
            "check_grad": True,
            "fd_epsilon": 1e-2,
            "include_supervised_baseline": True,
        },
        "evaluation": {
            "rollout_frames": 48,
            "seen_ic_trajectories": 4,
            "stable_error_threshold": 1.0,
            "first_interval_error_tolerance": 0.2,
            "native_long_error_tolerance": 1.0,
        },
    }


def main() -> None:
    """Submit validation, exact-source image builds, or independent GPU cells."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["validate", "build", "train", "report"])
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--host", default="kander-login")
    parser.add_argument(
        "--toolkit", type=Path, default=Path("/home/andrinr/slurm-runner/runner")
    )
    parser.add_argument("--solvers", default="jax-cfd,phiflow,pict,warp-ns,xlb,ins-jl")
    parser.add_argument("--regimes", default="64:4:8")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--pretrain-updates", type=int, default=0)
    parser.add_argument("--pretrain-unroll", type=int, default=8)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--reference-factor", type=int)
    parser.add_argument("--reference-temporal-factor", type=int)
    parser.add_argument("--audit-factor", type=int)
    parser.add_argument("--audit-temporal-factor", type=int)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument(
        "--after", default="", help="comma-separated prerequisite job IDs"
    )
    parser.add_argument(
        "--image-jobs", type=Path, help="remote JSON of solver build job IDs"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    campaign = Path(args.campaign)
    registry = JobRegistry(Path("mosaic-results/slurm-registry") / campaign.name)
    runner = Runner(registry, args.toolkit, transport=SshTransport(args.host))
    source_hash = _remote(
        args.host, ["sha256sum", str(campaign / "source.tar")]
    ).split()[0]
    cells = []
    image_jobs = (
        json.loads(_remote(args.host, ["cat", str(args.image_jobs)]))
        if args.image_jobs
        else {}
    )
    dependencies = {}
    if args.phase == "train":
        images = json.loads(_remote(args.host, ["cat", str(campaign / "images.json")]))
        for solver in args.solvers.split(","):
            for regime in args.regimes.split(","):
                n, k0, unroll = regime.split(":")
                for seed in map(int, args.seeds.split(",")):
                    phase = "confirm" if args.confirm else "explore"
                    cell = (
                        f"{phase}-{solver}-n{n}-k{k0}-h{unroll}-u{args.updates}-s{seed}"
                    )
                    if args.amplitude != 0.5:
                        cell += f"-a{args.amplitude:g}"
                    run = protocol(
                        int(n), float(k0), int(unroll), args.updates, seed, args.confirm
                    )
                    if args.pretrain_updates:
                        cell += f"-pre{args.pretrain_updates}"
                        run["training"].update(
                            {
                                "pretrain_updates": args.pretrain_updates,
                                "pretrain_unroll": args.pretrain_unroll,
                                "pretrain_seed": 2025,
                            }
                        )
                    run["dataset"]["amplitude"] = args.amplitude
                    for option, field in (
                        (args.reference_factor, "reference_factor"),
                        (args.reference_temporal_factor, "reference_temporal_factor"),
                        (args.audit_factor, "reference_audit_factor"),
                        (args.audit_temporal_factor, "reference_audit_temporal_factor"),
                    ):
                        if option is not None:
                            run["dataset"][field] = option
                    payload = json.dumps(
                        {
                            "solver": solver,
                            "image": images[solver],
                            "source_sha256": source_hash,
                            "run": run,
                        },
                        indent=2,
                    )
                    # Identity includes source + protocol, so revisions cannot overwrite results.
                    cell += "-" + hashlib.sha256(payload.encode()).hexdigest()[:8]
                    if not args.dry_run:
                        _remote(
                            args.host,
                            ["tee", str(campaign / "configs" / f"{cell}.json")],
                            payload,
                        )
                    cells.append(cell)
                    dependencies[cell] = (
                        [image_jobs[solver]] if solver in image_jobs else []
                    )
    else:
        cells = args.solvers.split(",") if args.phase == "build" else ["gate"]
    for cell in cells:
        gpu = args.phase == "train"
        prerequisites = [
            int(job) for job in args.after.split(",") if job
        ] + dependencies.get(cell, [])
        report_dependencies = (
            ["--dependency=afterany:" + ":".join(map(str, prerequisites))]
            if args.phase == "report" and prerequisites
            else []
        )
        spec = JobSpec(
            name=f"m116-{args.phase}-{cell}",
            cmd=(
                ["bash", str(campaign / "analyze.sh"), str(campaign)]
                if args.phase == "report"
                else [
                    "bash",
                    str(campaign / "node.sh"),
                    str(campaign),
                    args.phase,
                    cell,
                ]
            ),
            template="gpu" if gpu else "cpu",
            account="research",
            partition="dev",
            qos="rtx5090-pool" if gpu else "dev",
            time_limit="04:00:00" if gpu else "02:00:00",
            gpus="gpu:5090:1" if gpu else None,
            cpus=8,
            mem="64G",
            out_path=campaign / "results" / cell,
            depends_on=[] if args.phase == "report" else prerequisites,
            env={
                "PROJECT_ISOLATED_COMMAND": "1",
                "PROJECT_REPO_ROOT": str(campaign),
                "PYTHONUNBUFFERED": "1",
                "MLFLOW_DISABLE_AGENT_HINT": "1",
            },
            extra_sbatch=[f"--output={campaign}/logs/%x-%j.out", *report_dependencies],
        )
        result = runner.submit(spec, cluster="kander", dry_run=args.dry_run)
        print(
            json.dumps(
                {"cell": cell, "slurm_id": result.slurm_id, "state": result.job.state}
            ),
            flush=True,
        )
        if result.job.state == "failed":
            raise RuntimeError(result.job.reason)


if __name__ == "__main__":
    main()
