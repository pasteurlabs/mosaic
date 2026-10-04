"""Durable metadata-only controller for the last frozen INS comparison."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from runner import JobRegistry, JobSpec, Runner
from runner.transport import LocalTransport

ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED"}


def candidates() -> list[dict[str, Any]]:
    """Freeze the initial84fits; twelve selected independent refits follow."""
    rows = []
    for arm in ("full", "stopped", "supervised"):
        for lr in (1e-5, 3e-5, 1e-4):
            for horizon in (8,) if arm == "supervised" else (4, 8, 16):
                for updates in (1000, 3000) if arm == "supervised" else (1000,):
                    rows.append(
                        {
                            "candidate_id": f"{arm}-lr{lr:g}-h{horizon}-u{updates}",
                            "arm": arm,
                            "lr": lr,
                            "unroll": horizon,
                            "updates": updates,
                            "pretrain": None,
                        }
                    )
    for arm in ("full", "stopped"):
        for lr in (1e-5, 3e-5):
            rows.append(
                {
                    "candidate_id": f"{arm}-warm-lr{lr:g}-h8-u1000",
                    "arm": arm,
                    "lr": lr,
                    "unroll": 8,
                    "updates": 1000,
                    "pretrain": {"lr": 1e-5, "updates": 1000, "unroll": 8},
                }
            )
    return rows


def atomic_json(path: Path, value: Any) -> None:
    """Publish metadata atomically so a restarted controller can reuse it."""
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


class Controller:
    """Resume known job IDs; never silently resubmit a failed numerical cell."""

    def __init__(self, campaign: Path) -> None:
        self.campaign = campaign
        self.plan = json.loads((campaign / "plan.json").read_text())
        self.state_path = campaign / "controller-state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else {"jobs": {}}
        )
        self.report_path = campaign / "report-input.json"
        self.report = (
            json.loads(self.report_path.read_text())
            if self.report_path.exists()
            else {
                "validation": [],
                "training": [],
                "test": [],
                "selected": {},
                "expected_model_seeds": list(range(8, 16)),
                "expected_ic_seeds": list(range(20000, 20032)),
                "failures": [],
            }
        )
        self.registry = JobRegistry(campaign / "registry")
        plan_hash = hashlib.sha256((campaign / "plan.json").read_bytes()).hexdigest()
        if self.state.get("plan_sha256", plan_hash) != plan_hash:
            raise RuntimeError("frozen plan changed after controller started")
        self.state["plan_sha256"] = plan_hash
        self.runner = Runner(
            self.registry,
            Path(self.plan["runner_toolkit"]),
            transport=LocalTransport(),
        )

    def save(self) -> None:
        """Persist jobs and report references independently of controller lifetime."""
        atomic_json(self.state_path, self.state)
        atomic_json(self.report_path, self.report)

    def submit(
        self, cell: str, phase: str, payload: dict[str, Any] | None = None
    ) -> int:
        """Submit once, binding each cell to immutable JSON configuration."""
        config = self.campaign / "configs" / f"{cell}.json"
        if payload is not None:
            content = json.dumps(payload, indent=2)
            if config.exists() and config.read_text() != content:
                raise RuntimeError(f"immutable config changed: {cell}")
            if not config.exists():
                config.write_text(content)
        if cell in self.state["jobs"]:
            return int(self.state["jobs"][cell]["job_id"])
        recovered = [
            job
            for job in self.registry.query()
            if job.name == f"m116-final-{cell}" and job.slurm_id is not None
        ]
        if len(recovered) > 1:
            raise RuntimeError(f"ambiguous registry submissions for {cell}")
        if recovered:
            self.state["jobs"][cell] = {"job_id": recovered[0].slurm_id, "phase": phase}
            self.save()
            return int(recovered[0].slurm_id)
        gpu = phase in {"prepare", "train", "evaluate"}
        resources = self.plan.get("resources", {})
        spec = JobSpec(
            name=f"m116-final-{cell}",
            cmd=[
                "bash",
                str(self.campaign / "node.sh"),
                str(self.campaign),
                phase,
                cell,
            ],
            template="gpu" if gpu else "cpu",
            account="research",
            partition=resources.get("partition", "dev") if gpu else "dev",
            qos=resources.get("qos", "rtx5090-pool") if gpu else "dev",
            time_limit=resources.get("time_limit", "04:00:00") if gpu else "02:00:00",
            gpus=resources.get("gpus", "gpu:5090:1") if gpu else None,
            cpus=8,
            mem="64G",
            out_path=self.campaign / "allocations" / cell,
            env={
                "PROJECT_ISOLATED_COMMAND": "1",
                "PROJECT_REPO_ROOT": str(self.campaign),
                "PYTHONUNBUFFERED": "1",
            },
            extra_sbatch=[
                f"--output={self.campaign}/logs/%x-%j.out",
                "--exclude=rtx03",
            ],
        )
        result = self.runner.submit(spec, cluster="kander")
        if result.slurm_id is None or result.job.state == "failed":
            raise RuntimeError(f"submission failed {cell}: {result.job.reason}")
        self.state["jobs"][cell] = {"job_id": result.slurm_id, "phase": phase}
        self.save()
        print(json.dumps({"submitted": cell, "job_id": result.slurm_id}), flush=True)
        return int(result.slurm_id)

    def wait(self, cells: list[str]) -> None:
        """Wait for terminal states without performing any numerical analysis."""
        ids = [str(self.state["jobs"][cell]["job_id"]) for cell in cells]
        while ids:
            output = subprocess.check_output(
                [
                    "sacct",
                    "-X",
                    "-n",
                    "-P",
                    "-j",
                    ",".join(ids),
                    "--format=JobIDRaw,State%30",
                ],
                text=True,
            )
            states = {
                r.split("|")[0]: r.split("|")[1].split()[0].rstrip("+")
                for r in output.splitlines()
                if r
            }
            if all(i in states and states[i] not in ACTIVE for i in ids):
                self.state["last_wait"] = {"cells": cells, "states": states}
                self.save()
                bad = {i: states[i] for i in ids if states[i] != "COMPLETED"}
                if bad:
                    raise RuntimeError(
                        f"terminal infrastructure failures require explicit review: {bad}"
                    )
                return
            time.sleep(55)

    def base(self) -> dict[str, Any]:
        """Copy the frozen solver/physics contract for each job."""
        return copy.deepcopy(self.plan["base_payload"])

    def dataset(self, stage: str) -> dict[str, Any]:
        """Read the admitted prepared dataset identity, never regenerate it silently."""
        folder = self.campaign / "results" / f"prepare-{stage}"
        metadata = json.loads((folder / "dataset.json").read_text())
        return {
            "dataset_path": str(folder / "dataset.npz"),
            "dataset_metadata_path": str(folder / "dataset.json"),
            "dataset_sha256": metadata["dataset_sha256"],
        }

    def prepare(self, stage: str) -> None:
        """Shard fresh burns below4h, then assemble without numerical work on login."""
        if stage == "validation":
            groups = [
                [10000],
                list(range(10001, 10005)),
                list(range(10005, 10009)),
                list(range(10009, 10013)),
                list(range(10013, 10016)),
            ]
        else:
            groups = [list(range(start, start + 4)) for start in range(20000, 20032, 4)]
        cells = []
        for index, group in enumerate(groups):
            payload = self.base()
            payload["dataset_role"] = stage
            data = payload["run"]["dataset"]
            data["train_seeds"] = (
                list(range(8)) if stage == "validation" and index == 0 else [0]
            )
            data["test_seeds"] = group
            data["prefix_audit_seeds"] = data["train_seeds"] + group
            data["burn_in_cache_dir"] = str(self.campaign / "burn-in-cache")
            if stage == "test":
                payload["selection_sha256"] = hashlib.sha256(
                    (self.campaign / "selection.json").read_bytes()
                ).hexdigest()
            cell = f"prepare-{stage}-shard{index}"
            self.submit(cell, "prepare", payload)
            cells.append(cell)
            if stage == "validation" and index == 0:
                self.wait([cell])
                self.smoke(self.campaign / "results" / cell)
        self.wait(cells)
        merged = f"prepare-{stage}"
        payload = self.base()
        payload.update(
            shard_dirs=[str(self.campaign / "results" / cell) for cell in cells],
            dataset_role=stage,
            out=str(self.campaign / "results" / merged),
        )
        payload["run"]["dataset"].update(
            train_seeds=list(range(8)) if stage == "validation" else [0],
            test_seeds=[seed for group in groups for seed in group],
        )
        if stage == "test":
            payload["selection_sha256"] = hashlib.sha256(
                (self.campaign / "selection.json").read_bytes()
            ).hexdigest()
        self.submit(merged, "assemble", payload)
        self.wait([merged])
        self.dataset(stage)

    def smoke(self, anchor: Path) -> None:
        """Require real-solver one-update arm checks before the frozen search starts."""
        metadata = json.loads((anchor / "dataset.json").read_text())
        cells = []
        for arm in ("full", "stopped", "supervised"):
            payload = self.base()
            payload.update(
                dataset_path=str(anchor / "dataset.npz"),
                dataset_metadata_path=str(anchor / "dataset.json"),
                dataset_sha256=metadata["dataset_sha256"],
                arm=arm,
                model_seed=0,
            )
            payload["run"]["training"].update(
                lr=1e-5, unroll=8, max_updates=1, model_seeds=[0], check_grad=True
            )
            cell = f"smoke-{arm}"
            self.submit(cell, "train", payload)
            cells.append(cell)
        self.wait(cells)
        for cell in cells:
            outcome = json.loads(
                (self.campaign / "results" / cell / "outcome.json").read_text()
            )
            if not outcome.get("completed") or not outcome.get("admitted"):
                raise RuntimeError(f"real INS one-update smoke failed: {cell}")
        self.state["real_solver_smoke_passed"] = True
        self.save()

    def train(
        self,
        candidate: dict[str, Any],
        seed: int,
        stage: str,
        *,
        logical_arm: str | None = None,
        support: bool = False,
    ) -> str:
        """Fit one arm independently; warm starts refer to exact frozen supervision."""
        cell = f"{stage}-{candidate['candidate_id']}-s{seed}"
        payload = self.base()
        payload.update(
            self.dataset("validation"),
            arm=candidate["arm"],
            model_seed=seed,
            candidate=candidate,
        )
        payload["run"]["training"].update(
            lr=candidate["lr"],
            unroll=candidate["unroll"],
            max_updates=candidate["updates"],
            model_seeds=[seed],
            pretrain_updates=0,
            check_grad=True,
        )
        payload["run"]["training"].pop("curriculum", None)
        if candidate["pretrain"]:
            source = (
                self.campaign
                / "results"
                / f"{stage}-supervised-lr1e-05-h8-u1000-s{seed}"
            )
            model = source / "model.eqx"
            payload.update(
                initial_model_path=str(model),
                initial_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest()
                if model.exists()
                else None,
                pretrain_outcome_path=str(source / "outcome.json"),
            )
        self.submit(cell, "train", payload)
        row = {
            "candidate": candidate,
            "model_seed": seed,
            "path": str(self.campaign / "results" / cell),
        }
        group = "validation" if stage == "tune" else "training"
        if support:
            group = "support_training"
            self.report.setdefault(group, [])
        if group in {"training", "support_training"}:
            row["arm"] = logical_arm or candidate["arm"]
        if row not in self.report[group]:
            self.report[group].append(row)
            self.save()
        return cell

    def selection(self, stage: str) -> dict[str, Any]:
        """Run all selection arithmetic on an allocated CPU, then read its decision."""
        self.save()
        self.submit(stage, stage)
        self.wait([stage])
        filename = (
            "extensions.json" if stage == "select_extension" else "selection.json"
        )
        return json.loads((self.campaign / filename).read_text())

    def run(self) -> None:
        """Execute the bounded plan, persist selections, then evaluate fresh models/ICs."""
        self.submit("validation", "validate")
        self.wait(["validation"])
        self.prepare("validation")
        initial = self.plan["candidates"]
        cold = [
            self.train(c, s, "tune")
            for c in initial
            if not c["pretrain"]
            for s in (0, 1, 2)
        ]
        self.wait(cold)
        warm = [
            self.train(c, s, "tune")
            for c in initial
            if c["pretrain"]
            for s in (0, 1, 2)
        ]
        self.wait(warm)
        extended = []
        for rows in self.selection("select_extension").values():
            for selected in rows:
                candidate = {
                    **selected,
                    "updates": 3000,
                    "candidate_id": selected["candidate_id"].replace("u1000", "u3000"),
                }
                extended.extend(self.train(candidate, s, "tune") for s in (0, 1, 2))
        self.wait(extended)
        selected = self.selection("select_final")
        self.report["selected"] = selected
        self.save()
        # Prepare test references only after selection, while fresh training is independent.
        support = {
            "candidate_id": "supervised-lr1e-05-h8-u1000",
            "arm": "supervised",
            "lr": 1e-5,
            "unroll": 8,
            "updates": 1000,
            "pretrain": None,
        }
        support_cells = []
        if any(c["pretrain"] for c in selected.values()):
            support_cells = [
                self.train(support, s, "final", support=True) for s in range(8, 16)
            ]
            self.wait(support_cells)
        finals = [
            self.train(c, s, "final", logical_arm=arm)
            for arm, c in selected.items()
            for s in range(8, 16)
        ]
        self.prepare("test")
        self.wait(finals)
        evaluation_cells = []
        for arm, candidate in selected.items():
            for seed in range(8, 16):
                modeldir = (
                    self.campaign
                    / "results"
                    / f"final-{candidate['candidate_id']}-s{seed}"
                )
                for batch in range(8):
                    cell = f"test-{arm}-s{seed}-b{batch}"
                    payload = self.base()
                    payload.update(
                        self.dataset("test"),
                        arm=arm,
                        model_seed=seed,
                        candidate=candidate,
                        model_path=str(modeldir / "model.eqx"),
                        model_sha256=hashlib.sha256(
                            (modeldir / "model.eqx").read_bytes()
                        ).hexdigest(),
                        model_outcome_path=str(modeldir / "outcome.json"),
                        eval_indices=list(range(batch * 4, batch * 4 + 4)),
                        selection_sha256=hashlib.sha256(
                            (self.campaign / "selection.json").read_bytes()
                        ).hexdigest(),
                    )
                    self.submit(cell, "evaluate", payload)
                    evaluation_cells.append(cell)
                    row = {
                        "arm": arm,
                        "model_seed": seed,
                        "ic_seeds": list(range(20000 + batch * 4, 20004 + batch * 4)),
                        "path": str(self.campaign / "results" / cell),
                    }
                    if row not in self.report["test"]:
                        self.report["test"].append(row)
                        self.save()
        self.wait(evaluation_cells)
        self.submit("report", "report")
        self.wait(["report"])
        subprocess.run(
            [
                self.plan["controller_python"],
                str(self.campaign / "final_publish.py"),
                "--campaign",
                str(self.campaign),
            ],
            check=True,
        )
        self.state["completed"] = True
        self.save()


def main() -> None:
    """Run on the login node; Slurm owns every numerical operation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    lock = (args.campaign / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller = Controller(args.campaign)
    try:
        controller.run()
    except Exception as error:
        controller.state["failure"] = f"{type(error).__name__}: {error}"
        controller.report["failures"].append(controller.state["failure"])
        controller.save()
        try:
            controller.submit("report-failure", "report")
            controller.wait(["report-failure"])
            subprocess.run(
                [
                    controller.plan["controller_python"],
                    str(args.campaign / "final_publish.py"),
                    "--campaign",
                    str(args.campaign),
                ],
                check=True,
            )
        except Exception as publication_error:
            controller.state["failure_report_error"] = repr(publication_error)
            controller.save()
        raise


if __name__ == "__main__":
    main()
