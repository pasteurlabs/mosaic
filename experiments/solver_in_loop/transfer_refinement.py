"""Bounded reference timestep convergence ladder without model or physics tuning."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path

from experiments.solver_in_loop.final_controller import Controller
from experiments.solver_in_loop.transfer_advance import launch


class RefinementController(Controller):
    """Retain each predeclared reference attempt, then dispatch admitted transfer."""

    def run(self) -> None:
        """Refine only reference dt; all coarse training recipes remain frozen."""
        receipt = json.loads((self.campaign / "validation.json").read_text())
        if (
            not receipt.get("passed")
            or receipt["source_sha256"] != self.plan["source_sha256"]
        ):
            raise RuntimeError("exact-source validation receipt required")
        solver = self.plan["refinement_solver"]
        spec = self.plan["solvers"][solver]
        self.state.setdefault("refinement_attempts", {})
        for factor, audit in self.plan["reference_ladder"]:
            payload = self.base()
            payload.update(
                solver=solver,
                image=spec["image"],
                image_sha256=spec["image_sha256"],
                dataset_role="validation",
            )
            payload["run"]["dataset"].update(
                train_seeds=[0],
                test_seeds=[20000],
                prefix_audit_seeds=[0, 20000],
                reference_temporal_factor=factor,
                reference_audit_temporal_factor=audit,
                burn_in_cache_dir=str(self.campaign / "burn-in-cache"),
            )
            payload["run"]["training"].update(
                lr=1e-4, unroll=16, max_updates=10, model_seeds=[8], check_grad=True
            )
            payload["run"]["training"].pop("curriculum", None)
            cell = f"reference-{solver}-t{factor}-audit{audit}"
            self.submit(cell, "prepare", payload)
            self.state["status"] = "reference_refinement_running"
            self.save()
            self.wait([cell])
            folder = self.campaign / "results" / cell
            outcome = json.loads((folder / "outcome.json").read_text())
            self.state["refinement_attempts"][cell] = outcome
            self.save()
            if not outcome.get("completed"):
                self.state["status"] = "blocked_by_reference_failure_requires_review"
                self.save()
                return
            if not outcome.get("admitted"):
                continue
            metadata = json.loads((folder / "dataset.json").read_text())
            payload.update(
                arm="full",
                model_seed=8,
                dataset_path=str(folder / "dataset.npz"),
                dataset_metadata_path=str(folder / "dataset.json"),
                dataset_sha256=metadata["dataset_sha256"],
            )
            probe = f"gradient-runtime-{solver}-t{factor}"
            self.submit(probe, "train", payload)
            self.wait([probe])
            gradient = json.loads(
                (self.campaign / "results" / probe / "outcome.json").read_text()
            )
            self.state["gradient_runtime_outcome"] = gradient
            if not gradient.get("completed") or not gradient.get("admitted"):
                self.state["status"] = "blocked_by_gradient_runtime_admission"
                self.save()
                return
            admitted_plan = copy.deepcopy(self.plan)
            admitted_plan["solvers"][solver].update(
                reference_temporal_factor=factor, audit_temporal_factor=audit
            )
            self.state["confirmation"] = launch(self.campaign, solver, admitted_plan)
            self.state["status"] = "all_IC_admission_dispatched"
            self.save()
            return
        self.state["status"] = "blocked_by_prespecified_refinement_ladder"
        self.save()


def main() -> None:
    """Metadata-only durable entry point; numerical work is delegated to Slurm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    lock = (args.campaign / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller = RefinementController(args.campaign)
    try:
        controller.run()
    except Exception as error:
        controller.state.update(
            status="infrastructure_failure_requires_review", failure=repr(error)
        )
        controller.save()
        raise


if __name__ == "__main__":
    main()
