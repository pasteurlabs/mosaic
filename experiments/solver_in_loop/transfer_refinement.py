"""Bounded reference timestep convergence ladder without model or physics tuning."""

from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import hashlib
import json
import struct
import subprocess
import time
import zipfile
from pathlib import Path

from experiments.solver_in_loop.final_controller import Controller
from experiments.solver_in_loop.transfer_advance import launch


def verified_cache_files(payload: dict) -> dict[str, str]:
    """Check completed cache byte identities without loading numerical arrays."""

    def read_npy(archive: zipfile.ZipFile, name: str) -> tuple[dict, bytes]:
        raw = archive.read(name + ".npy")
        if raw[:6] != b"\x93NUMPY" or raw[6] not in (1, 2):
            raise ValueError("unsupported cache NPY header")
        width = 2 if raw[6] == 1 else 4
        size = struct.unpack("<H" if width == 2 else "<I", raw[8 : 8 + width])[0]
        return ast.literal_eval(raw[8 + width : 8 + width + size].decode()), raw[
            8 + width + size :
        ]

    def text(archive: zipfile.ZipFile, name: str) -> str:
        header, raw = read_npy(archive, name)
        if header["shape"] != () or not header["descr"].startswith("<U"):
            raise ValueError("invalid cache scalar metadata")
        return raw.decode("utf-32-le").rstrip("\0")

    dataset = payload["run"]["dataset"]
    expected_dt = payload["run"]["physics"]["dt"] / dataset["reference_temporal_factor"]
    found = {}
    for path in Path(dataset["burn_in_cache_dir"]).glob("*.npz"):
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(text(archive, "metadata"))
            key = hashlib.sha256(
                json.dumps(metadata, sort_keys=True).encode()
            ).hexdigest()
            header, state = read_npy(archive, "state")
            if (
                metadata["binding"] != payload["source_sha256"] + ":" + payload["image"]
                or metadata["physics"] != payload["run"]["physics"]
                or metadata["dt"] != expected_dt
            ):
                continue
            if (
                key != text(archive, "key")
                or path.stem != key
                or hashlib.sha256(state).hexdigest() != text(archive, "sha256")
                or list(header["shape"]) != metadata["shape"]
            ):
                raise ValueError("completed burn cache byte identity mismatch")
            found[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


class RefinementController(Controller):
    """Retain each predeclared reference attempt, then dispatch admitted transfer."""

    def submit_admission(self, cell: str, phase: str, payload: dict) -> str:
        """Apply recorded allocation-only overrides to preflight, never confirmation fits."""
        policy_path = self.campaign / "admission-resources.json"
        if not policy_path.exists():
            self.submit(cell, phase, payload)
            return cell
        policy = json.loads(policy_path.read_text())
        if policy["source_sha256"] != payload["source_sha256"]:
            raise RuntimeError("admission resource policy source differs")
        cell = policy.get("replacements", {}).get(cell, cell)
        previous = self.plan["resources"]
        try:
            self.plan["resources"] = policy["resources"]
            self.submit(cell, phase, payload)
        finally:
            self.plan["resources"] = previous
        self.state["admission_resources_policy_sha256"] = hashlib.sha256(
            policy_path.read_bytes()
        ).hexdigest()
        self.save()
        return cell

    def wait_reference(self, cell: str, payload: dict) -> str:
        """Permit one reviewed TIMEOUT retry only if a completed burn cache exists."""
        policy_path = self.campaign / "timeout-cache-retry-policy.json"
        records = self.state.setdefault("timeout_reference_retries", {})
        if cell in records:
            retry = records[cell]["retry_cell"]
            self.submit(retry, "prepare", payload)
            self.wait([retry])
            return retry
        try:
            self.wait([cell])
            return cell
        except RuntimeError:
            if not policy_path.exists():
                raise
            policy = json.loads(policy_path.read_text())
            if (
                policy["source_sha256"] != payload["source_sha256"]
                or policy["max_retries_per_cell"] != 1
            ):
                raise RuntimeError("invalid timeout retry authorization") from None
            job = str(self.state["jobs"][cell]["job_id"])
            state = subprocess.check_output(
                ["sacct", "-X", "-n", "-P", "-j", job, "--format=JobIDRaw,State%30"],
                text=True,
            )
            rows = dict(line.split("|")[:2] for line in state.splitlines() if line)
            if rows.get(job, "").split()[0] != "TIMEOUT":
                raise
            cached = verified_cache_files(payload)
            if not cached:
                raise RuntimeError(
                    "timeout produced no verified completed IC burn cache; not retrying"
                ) from None
            retry = cell + "-timeout-retry1"
            records[cell] = {
                "retry_cell": retry,
                "original_job_id": job,
                "original_state": "TIMEOUT",
                "verified_completed_cache": cached,
                "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            }
            self.save()
            self.submit(retry, "prepare", payload)
            self.wait([retry])
            return retry

    def run(self) -> None:
        """Refine only reference dt; all coarse training recipes remain frozen."""
        receipt = json.loads((self.campaign / "validation.json").read_text())
        if (
            not receipt.get("passed")
            or receipt["source_sha256"] != self.plan["source_sha256"]
        ):
            raise RuntimeError("exact-source validation receipt required")
        gate = self.plan.get("variant_gate")
        if gate:
            self.state["status"] = "waiting_for_verified_variant_gate"
            self.save()
            job = str(gate["job_id"])
            while True:
                output = subprocess.check_output(
                    [
                        "sacct",
                        "-X",
                        "-n",
                        "-P",
                        "-j",
                        job,
                        "--format=JobIDRaw,State%30",
                    ],
                    text=True,
                )
                states = dict(
                    line.split("|")[:2] for line in output.splitlines() if line
                )
                status = states.get(job, "PENDING").split()[0]
                if status == "COMPLETED":
                    break
                if status not in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
                    raise RuntimeError(f"variant gate did not complete: {status}")
                time.sleep(30)
            evidence = json.loads(Path(gate["outcome_path"]).read_text())
            for key, value in gate["expected_outcome"].items():
                if evidence.get(key) != value:
                    raise RuntimeError(f"variant gate mismatch: {key}")
            if not (
                evidence["training_primary_fd_error"] < 0.05
                and evidence["temporal_max_error"] <= 0.005
            ):
                raise RuntimeError("variant numerical admission failed")
            verification = json.loads(Path(gate["image_verification_path"]).read_text())
            for key, value in gate["expected_image_verification"].items():
                if verification.get(key) != value:
                    raise RuntimeError(f"variant image verification mismatch: {key}")
            self.state["variant_gate_receipt_sha256"] = hashlib.sha256(
                Path(gate["outcome_path"]).read_bytes()
            ).hexdigest()
            self.save()
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
            cell = self.submit_admission(cell, "prepare", payload)
            self.state["status"] = "reference_refinement_running"
            self.save()
            cell = self.wait_reference(cell, payload)
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
            probe = self.submit_admission(probe, "train", payload)
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
