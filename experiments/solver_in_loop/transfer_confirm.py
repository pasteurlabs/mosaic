"""Durable frozen-recipe transfer confirmation with exact training continuation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from experiments.solver_in_loop.final_controller import ACTIVE, Controller


def digest(path: Path) -> str:
    """Hash a checkpoint without loading numerical objects into the controller."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class ConfirmationController(Controller):
    """Prepare all40ICs once, finish exact fits, then evaluate disjoint held-out data."""

    def ready(self, pending: dict[str, Any]) -> list[str]:
        """Return finished cells while other independent allocations keep running."""
        ids = {cell: str(self.state["jobs"][cell]["job_id"]) for cell in pending}
        output = subprocess.check_output(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                ",".join(ids.values()),
                "--format=JobIDRaw,State%30",
            ],
            text=True,
        )
        states = {
            r.split("|")[0]: r.split("|")[1].split()[0].rstrip("+")
            for r in output.splitlines()
            if r
        }
        return [
            cell
            for cell, job in ids.items()
            if job in states and states[job] not in ACTIVE
        ]

    def data(self) -> dict[str, Any]:
        """Bind every fit/evaluation to one assembled reference file."""
        folder = self.campaign / "results" / "assembled"
        metadata = json.loads((folder / "dataset.json").read_text())
        return {
            "dataset_path": str(folder / "dataset.npz"),
            "dataset_metadata_path": str(folder / "dataset.json"),
            "dataset_sha256": metadata["dataset_sha256"],
        }

    def outcome(self, cell: str) -> dict[str, Any]:
        """Read only published scalar/identity metadata."""
        return json.loads(
            (self.campaign / "results" / cell / "outcome.json").read_text()
        )

    def fit(
        self, arm: str, seed: int, attempt: int, previous: str | None = None
    ) -> str:
        """Submit the unchanged recipe with an allocation boundary, never a restart."""
        recipe = self.plan["confirmation_recipes"][arm]
        cell = f"train-{arm}-s{seed}-part{attempt}"
        payload = self.base()
        payload.update(
            self.data(),
            arm=arm,
            model_seed=seed,
            evaluate_after_training=False,
            comparison_role="frozen_transfer_confirmation",
        )
        payload["run"]["training"].update(
            lr=recipe["lr"],
            unroll=recipe["unroll"],
            max_updates=recipe["updates"],
            model_seeds=[seed],
            check_grad=True,
        )
        payload["continuation"] = {
            "path": str(self.campaign / "checkpoints" / f"{arm}-s{seed}.zip"),
            "checkpoint_every": 100,
            "wall_limit_s": self.plan.get("continuation_wall_limit_s", 7200),
        }
        if previous:
            outcome = self.campaign / "results" / previous / "outcome.json"
            payload["continuation"].update(
                previous_outcome_path=str(outcome),
                previous_outcome_sha256=digest(outcome),
            )
        self.submit(cell, "train", payload)
        return cell

    def publish_report(self) -> None:
        """Render transfer evidence on a CPU allocation, including incomplete runs."""
        self.save()
        self.submit("transfer-report", "report")
        self.wait(["transfer-report"])
        report = json.loads((self.campaign / "report" / "report.json").read_text())
        if not report.get("report_complete"):
            raise RuntimeError("transfer report did not complete")
        self.state["report_complete"] = True
        self.state["conclusion"] = report["conclusion"]
        if not self.state.get("status", "").startswith("blocked_"):
            self.state["status"] = (
                "confirmation_inconclusive"
                if report.get("failures")
                else "confirmation_report_complete"
            )
        self.save()

    def run(self) -> None:
        """Execute registered transfer only; any failed admission keeps it inconclusive."""
        self.submit("validation", "validate")
        self.wait(["validation"])
        cells = []
        for seed in list(range(8)) + list(range(20000, 20032)):
            payload = self.base()
            payload.update(
                single_ic=True,
                generate_supervised_pairs=seed < 8,
                dataset_role="validation",
                comparison_role="training_reference"
                if seed < 8
                else "held_out_transfer_reference",
            )
            payload["run"]["dataset"].update(
                train_seeds=[seed],
                test_seeds=[],
                prefix_audit_seeds=[seed],
                burn_in_cache_dir=str(self.campaign / "burn-in-cache"),
            )
            cell = f"reference-ic{seed}"
            self.submit(cell, "prepare", payload)
            cells.append(cell)
        self.wait(cells)
        failed = [cell for cell in cells if not self.outcome(cell).get("admitted")]
        if failed:
            self.state.update(
                status="blocked_by_all_IC_admission", failed_reference_cells=failed
            )
            self.report["failures"] = [
                {
                    "cell": cell,
                    "failure": self.outcome(cell).get(
                        "failure", "reference admission failed"
                    ),
                }
                for cell in failed
            ]
            self.publish_report()
            return
        payload = self.base()
        payload.update(
            shard_dirs=[str(self.campaign / "results" / cell) for cell in cells]
        )
        payload["run"]["dataset"].update(
            train_seeds=list(range(8)), test_seeds=list(range(20000, 20032))
        )
        self.submit("assembled", "assemble", payload)
        self.wait(["assembled"])
        if not self.outcome("assembled").get("admitted"):
            raise RuntimeError("shared reference assembly failed")
        pending = {}
        for arm in ("full", "stopped", "supervised"):
            for seed in range(8, 16):
                cell = self.fit(arm, seed, 0)
                pending[cell] = (arm, seed, 0)
        completed = {}
        failures = []
        while pending:
            for cell in self.ready(pending):
                arm, seed, attempt = pending.pop(cell)
                try:
                    self.wait([cell])
                    result = self.outcome(cell)
                    if result.get("resume_required"):
                        if attempt + 1 >= 12:
                            raise RuntimeError(
                                "twelve-allocation safety cap reached before fixed update count"
                            )
                        next_cell = self.fit(arm, seed, attempt + 1, previous=cell)
                        pending[next_cell] = (arm, seed, attempt + 1)
                    elif result.get("completed") and result.get("admitted"):
                        completed[f"{arm}-s{seed}"] = cell
                    else:
                        raise RuntimeError(
                            result.get("failure", "training failed admission")
                        )
                except Exception as error:
                    failures.append(
                        {
                            "arm": arm,
                            "model_seed": seed,
                            "cell": cell,
                            "failure": repr(error),
                        }
                    )
                self.state.update(
                    completed_training=completed,
                    training_failures=failures,
                    pending_training=list(pending),
                )
                self.save()
            if pending:
                time.sleep(55)
        self.report.update(
            selected=self.plan["confirmation_recipes"],
            solver=self.plan["base_payload"]["solver"],
        )
        self.report["failures"] = failures
        self.report["training"] = []
        self.report["test"] = []
        evaluations = []
        for key, cell in completed.items():
            arm, seedtext = key.rsplit("-s", 1)
            seed = int(seedtext)
            folder = self.campaign / "results" / cell
            self.report["training"].append(
                {
                    "arm": arm,
                    "model_seed": seed,
                    "path": str(folder),
                    "candidate": self.plan["confirmation_recipes"][arm],
                }
            )
            for batch in range(8):
                payload = self.base()
                payload.update(
                    self.data(),
                    arm=arm,
                    model_seed=seed,
                    model_path=str(folder / "model.eqx"),
                    model_sha256=digest(folder / "model.eqx"),
                    model_outcome_path=str(folder / "outcome.json"),
                    eval_indices=list(range(4 * batch, 4 * batch + 4)),
                )
                name = f"evaluate-{arm}-s{seed}-b{batch}"
                self.submit(name, "evaluate", payload)
                evaluations.append(name)
                self.report["test"].append(
                    {
                        "arm": arm,
                        "model_seed": seed,
                        "ic_seeds": list(range(20000 + 4 * batch, 20004 + 4 * batch)),
                        "path": str(self.campaign / "results" / name),
                    }
                )
        self.save()
        self.wait(evaluations)
        for cell in evaluations:
            outcome = self.outcome(cell)
            if not outcome.get("completed") or not outcome.get("admitted"):
                failures.append(
                    {
                        "cell": cell,
                        "failure": outcome.get(
                            "failure", "evaluation failed admission"
                        ),
                    }
                )
        self.report["failures"] = failures
        self.state.update(
            status="confirmation_finished_requires_report",
            completed=not failures,
            evaluation_cells=evaluations,
        )
        self.publish_report()


def main() -> None:
    """Run metadata orchestration on login; every array operation stays in Slurm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    lock = (args.campaign / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller = ConfirmationController(args.campaign)
    try:
        controller.run()
    except Exception as error:
        controller.state.update(failure=repr(error), completed=False)
        controller.save()
        raise


if __name__ == "__main__":
    main()
