"""Frozen cross-solver admission preflight; never select model hyperparameters."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

from experiments.solver_in_loop.final_controller import ACTIVE, Controller


class TransferController(Controller):
    """Run every registered reference gate, then admitted gradient/runtime probes."""

    def ready_references(self) -> Iterator[str]:
        """Yield finished reference cells without delaying fast solvers behind slow ones."""
        pending = set(self.plan["solvers"])
        while pending:
            ids = {
                solver: str(self.state["jobs"][f"reference-{solver}"]["job_id"])
                for solver in pending
            }
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
            ready = [
                solver
                for solver, job in ids.items()
                if job in states and states[job] not in ACTIVE
            ]
            for solver in sorted(ready):
                pending.remove(solver)
                yield solver
            if pending:
                time.sleep(55)

    def run(self) -> None:
        """Persist each solver's failure independently; preflight never starts fitting."""
        self.state.setdefault("solvers", {})
        for solver, specification in self.plan["solvers"].items():
            payload = copy.deepcopy(self.plan["base_payload"])
            payload.update(
                solver=solver,
                image=specification["image"],
                image_sha256=specification["image_sha256"],
                dataset_role="validation",
                experiment_purpose="transfer_admission_only",
            )
            payload["run"]["dataset"].update(
                train_seeds=[0],
                test_seeds=[20000],
                prefix_audit_seeds=[0, 20000],
                reference_temporal_factor=specification["reference_temporal_factor"],
                reference_audit_temporal_factor=specification["audit_temporal_factor"],
                burn_in_cache_dir=str(self.campaign / "burn-in-cache" / solver),
            )
            payload["run"]["training"].update(
                lr=1e-4, unroll=16, max_updates=10, model_seeds=[8], check_grad=True
            )
            payload["run"]["training"].pop("curriculum", None)
            self.submit(f"reference-{solver}", "prepare", payload)
            self.state["solvers"].setdefault(
                solver, {"reference_cell": f"reference-{solver}"}
            )
            self.save()
        probes = []
        for solver in self.ready_references():
            cell = f"reference-{solver}"
            state = self.state["solvers"][solver]
            try:
                self.wait([cell])
                folder = self.campaign / "results" / cell
                outcome = json.loads((folder / "outcome.json").read_text())
                metadata = json.loads((folder / "dataset.json").read_text())
                state["reference_admitted"] = bool(
                    outcome.get("admitted") and metadata.get("admitted")
                )
                if not state["reference_admitted"]:
                    state["status"] = "blocked_by_reference_admission"
                    state["reference_failure"] = outcome.get(
                        "failure", "reference or closure gate failed"
                    )
                    self.save()
                    continue
                payload = json.loads(
                    (self.campaign / "configs" / f"{cell}.json").read_text()
                )
                payload.update(
                    arm="full",
                    model_seed=8,
                    dataset_path=str(folder / "dataset.npz"),
                    dataset_metadata_path=str(folder / "dataset.json"),
                    dataset_sha256=metadata["dataset_sha256"],
                )
                probe = f"gradient-runtime-{solver}"
                self.submit(probe, "train", payload)
                state["probe_cell"] = probe
                state["status"] = "gradient_runtime_pending"
                probes.append((solver, probe))
            except Exception as error:
                state.update(
                    status="reference_preflight_failed",
                    failure=f"{type(error).__name__}: {error}",
                )
            self.save()
        for solver, probe in probes:
            state = self.state["solvers"][solver]
            try:
                self.wait([probe])
                outcome = json.loads(
                    (self.campaign / "results" / probe / "outcome.json").read_text()
                )
                state.update(
                    gradient_admitted=bool(outcome.get("admitted")),
                    gradient_relative_error=outcome.get("gradient_relative_error"),
                    training_wall_time_s=outcome.get("training_wall_time_s"),
                    optimizer_updates=outcome.get("optimizer_updates"),
                    status="preflight_passed_requires_all_IC_admission"
                    if outcome.get("admitted")
                    else "blocked_by_gradient_or_runtime_gate",
                )
            except Exception as error:
                state.update(
                    status="gradient_runtime_preflight_failed",
                    failure=f"{type(error).__name__}: {error}",
                )
            self.save()
        self.state["preflight_complete"] = True
        self.state["confirmation_started"] = False
        self.state["next_gate"] = (
            "Admit every training/test IC and validate full3000-update runtime before confirmation; no retuning."
        )
        self.save()


def main() -> None:
    """Run on login using frozen source; delegate all simulation to Slurm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    lock = (args.campaign / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller = TransferController(args.campaign)
    controller.state["transfer_controller_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    try:
        controller.run()
    except Exception as error:
        controller.state["failure"] = repr(error)
        controller.save()
        raise


if __name__ == "__main__":
    main()
