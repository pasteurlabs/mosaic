"""Advance admitted transfer solvers after exact-source continuation validation."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def save(path: Path, value: Any) -> None:
    """Atomically retain control-plane evidence."""
    pending = path.with_suffix(".part")
    pending.write_text(json.dumps(value, indent=2))
    pending.replace(path)


def launch(master: Path, solver: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Launch one immutable same-hardware confirmation campaign, without retuning."""
    child = master / solver
    child.mkdir(exist_ok=True)
    for name in ("configs", "logs", "results", "checkpoints"):
        (child / name).mkdir(exist_ok=True)
    for name in ("source.tar", "source.sha256"):
        if not (child / name).exists():
            shutil.copy2(master / name, child / name)
    node = (
        master
        / "controller-source"
        / "experiments"
        / "solver_in_loop"
        / "transfer_node.sh"
    )
    if not (child / "node.sh").exists():
        shutil.copy2(node, child / "node.sh")
    specification = plan["solvers"][solver]
    base = copy.deepcopy(plan["base_payload"])
    base.update(
        solver=solver,
        image=specification["image"],
        image_sha256=specification["image_sha256"],
        source_sha256=plan["source_sha256"],
    )
    base["run"]["dataset"].update(
        train_seeds=list(range(8)),
        test_seeds=list(range(20000, 20032)),
        reference_temporal_factor=specification["reference_temporal_factor"],
        reference_audit_temporal_factor=specification["audit_temporal_factor"],
    )
    childplan = {
        "base_payload": base,
        "source_sha256": plan["source_sha256"],
        "source_git_commit": plan["source_git_commit"],
        "runner_toolkit": plan["runner_toolkit"],
        "controller_python": plan["controller_python"],
        "confirmation_recipes": plan["confirmation_recipes"],
        "resources": plan["resources"],
        "continuation_wall_limit_s": 36000,
        "preflight_campaign": plan["preflight_campaign"],
        "comparison_scope": "Frozen INS recipe transfer; no selection or tuning on these32ICs; all three arms onB200.",
    }
    target = child / "plan.json"
    if target.exists() and json.loads(target.read_text()) != childplan:
        raise RuntimeError("child immutable plan mismatch")
    save(target, childplan)
    script = (
        master
        / "controller-source"
        / "experiments"
        / "solver_in_loop"
        / "transfer_confirm.py"
    )
    pidfile = child / "controller.pid"
    if pidfile.exists():
        prior = int(pidfile.read_text())
        try:
            command = Path(f"/proc/{prior}/cmdline").read_bytes()
            if str(child).encode() in command:
                return {
                    "campaign": str(child),
                    "controller_pid": prior,
                    "status": "all_IC_admission_started",
                }
        except FileNotFoundError:
            pass
    with (child / "controller.log").open("a") as out:
        proc = subprocess.Popen(
            [plan["controller_python"], str(script), "--campaign", str(child)],
            cwd=child,
            env={
                **os.environ,
                "PYTHONPATH": f"{master}/controller-source:/home/andrinr/slurm-runner",
            },
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (child / "controller.pid").write_text(str(proc.pid))
    return {
        "campaign": str(child),
        "controller_pid": proc.pid,
        "status": "all_IC_admission_started",
    }


def main() -> None:
    """Wait for both validation and per-solver preflight before consuming GPU budget."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    master = args.campaign
    lock = (master / "advance.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((master / "plan.json").read_text())
    statepath = master / "advance-state.json"
    state = json.loads(statepath.read_text()) if statepath.exists() else {"solvers": {}}
    state["status"] = "waiting_for_exact_source_validation"
    save(statepath, state)
    receiptpath = master / "validation.json"
    while not receiptpath.exists():
        time.sleep(30)
    receipt = json.loads(receiptpath.read_text())
    if receipt["source_sha256"] != plan["source_sha256"] or not receipt.get("passed"):
        state["status"] = "blocked_by_continuation_validation"
        save(statepath, state)
        raise RuntimeError("continuation validation failed or used a different source")
    output = subprocess.check_output(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(map(str, receipt["job_ids"])),
            "--format=JobIDRaw,State%30",
        ],
        text=True,
    )
    states = {
        r.split("|")[0]: r.split("|")[1].split()[0].rstrip("+")
        for r in output.splitlines()
        if r
    }
    if any(states.get(str(job)) != "COMPLETED" for job in receipt["job_ids"]):
        raise RuntimeError("CPU/GPU continuation validation jobs must have completed")
    probe = json.loads(Path(receipt["probe_path"]).read_text())
    if (
        not probe.get("passed")
        or probe["identity"]["source_sha256"] != plan["source_sha256"]
    ):
        raise RuntimeError(
            "real solver continuation parity not verified for this source"
        )
    state["validation_receipt_sha256"] = hashlib.sha256(
        receiptpath.read_bytes()
    ).hexdigest()
    save(statepath, state)
    pending = set(plan["confirmation_solver_candidates"]) - set(state["solvers"])
    while pending:
        preflight = json.loads(
            (Path(plan["preflight_campaign"]) / "controller-state.json").read_text()
        )
        for solver in sorted(pending.copy()):
            status = preflight["solvers"].get(solver, {})
            if status.get("status") == "preflight_passed_requires_all_IC_admission":
                state["solvers"][solver] = launch(master, solver, plan)
                pending.remove(solver)
                save(statepath, state)
            elif status.get("status") in {
                "blocked_by_reference_admission",
                "reference_preflight_failed",
                "blocked_by_gradient_or_runtime_gate",
                "gradient_runtime_preflight_failed",
            }:
                state["solvers"][solver] = {
                    "status": "blocked_by_preflight",
                    "evidence": status,
                }
                pending.remove(solver)
                save(statepath, state)
        if pending:
            time.sleep(55)
    state["admission_dispatch_complete"] = True
    save(statepath, state)


if __name__ == "__main__":
    main()
