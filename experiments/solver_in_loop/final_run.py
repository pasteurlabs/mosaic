"""One-arm INS corrector experiments using frozen data and existing core losses.

Preparation, training and checkpoint evaluation are separate jobs. The existing
three-arm benchmark is unchanged. A 3000-update fit starts from its declared
initialization, not a reset-Adam continuation of a 1000-update fit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract

from mosaic.benchmarks.problems import get_config
from mosaic.benchmarks.problems.navier_stokes_grid.corrector import (
    init_corrector,
    relative_l2,
)
from mosaic.benchmarks.problems.navier_stokes_grid.training_continuation import (
    TrainingContinuation,
    TrainingYield,
)

core = importlib.import_module(
    "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop"
)


def file_hash(path: str | Path) -> str:
    """Hash the exact saved artifact, streaming large dataset files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(payload: dict) -> dict:
    """Bind prepared data and models to the executed solver and physics."""
    return {
        key: payload[key]
        for key in ("solver", "source_sha256", "image", "image_sha256")
    } | {
        "physics": payload["run"]["physics"],
        "domain_extent": float(payload.get("domain_extent", 2 * np.pi)),
    }


def model_spec(training: dict) -> dict:
    """Record all architecture parameters required to deserialize a model."""
    return {
        "hidden_channels": int(training.get("hidden_channels", 32)),
        "kernel_size": int(training.get("kernel_size", 5)),
        "architecture": str(training.get("architecture", "periodic_residual_cnn")),
    }


def load_dataset(payload: dict) -> tuple[dict, dict]:
    """Reject changed arrays, source, image or physical configuration."""
    metadata_path = payload.get("dataset_metadata_path", payload.get("metadata_path"))
    metadata = json.loads(Path(metadata_path).read_text())
    actual = file_hash(payload["dataset_path"])
    if actual != payload["dataset_sha256"] or actual != metadata["dataset_sha256"]:
        raise ValueError("frozen dataset SHA256 mismatch")
    if metadata["identity"] != identity(payload):
        raise ValueError("frozen dataset source/image/physics identity mismatch")
    if not metadata["admitted"]:
        raise ValueError("frozen dataset failed admission")
    with np.load(payload["dataset_path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays, metadata


def prepare(t: Any, ctx: Any, payload: dict, out: Path) -> dict:
    """Generate one immutable split and run original reference/closure gates."""
    started = time.perf_counter()
    run = payload["run"]
    dataset = copy.deepcopy(run["dataset"])
    if payload.get("dataset_role", "validation") == "test" and not payload.get(
        "selection_sha256"
    ):
        raise ValueError("test preparation requires a frozen selection identity")
    train_seeds = list(map(int, dataset["train_seeds"]))
    eval_seeds = list(map(int, dataset["test_seeds"]))
    all_seeds = train_seeds + eval_seeds
    single_ic = bool(payload.get("single_ic", False))
    if single_ic:
        if len(train_seeds) != 1 or eval_seeds:
            raise ValueError(
                "single-IC preparation requires one seed and no evaluation list"
            )
    elif len(set(all_seeds)) != len(all_seeds) or not train_seeds or not eval_seeds:
        raise ValueError("nonempty disjoint unique training/evaluation ICs required")
    # Audit every supplied IC with the original same-solver helper.
    dataset["prefix_audit_seeds"] = all_seeds
    train, train_rollouts, references, dataset_hash, audit = (
        core._make_solver_self_reference_datasets(
            t,
            ctx,
            dataset=dataset,
            evaluation=run["evaluation"],
            training=run["training"],
        )
    )
    velocity_scale = float(np.sqrt(np.mean(train**2)) + 1e-8)
    frame_steps = int(ctx.phys["steps"])
    (
        native_rollouts,
        native_errors,
        semigroup_errors,
        long_closure,
        native_final_errors,
    ) = [], [], [], [], []
    for reference in np.concatenate([train_rollouts, references], axis=0):
        native, errors = core._evaluate_rollout(
            t,
            ctx,
            None,
            reference,
            frame_steps=frame_steps,
            velocity_scale=velocity_scale,
            corrected=False,
        )
        native_rollouts.append(native)
        native_errors.append(errors)
        first, state = core._solver_advance(
            t, ctx, jnp.asarray(reference[0]), frame_steps=frame_steps
        )
        repeated, _ = core._solver_advance(
            t, ctx, first, frame_steps=frame_steps, native_state=state
        )
        uninterrupted, _ = core._solver_advance(
            t, ctx, jnp.asarray(reference[0]), frame_steps=2 * frame_steps
        )
        semigroup_errors.append(
            relative_l2(np.asarray(repeated), np.asarray(uninterrupted))
        )
        final, _ = core._solver_advance(
            t,
            ctx,
            jnp.asarray(reference[0]),
            frame_steps=frame_steps * (len(reference) - 1),
        )
        long_closure.append(relative_l2(native[-1], np.asarray(final)))
        native_final_errors.append(relative_l2(np.asarray(final), reference[-1]))
    native_errors = np.asarray(native_errors)
    native_rollouts = np.asarray(native_rollouts)
    median_tol = float(dataset.get("semigroup_median_tolerance", 0.005))
    p95_tol = float(dataset.get("semigroup_p95_tolerance", 0.01))
    closure_tol = float(dataset.get("long_closure_tolerance", p95_tol))
    closure_passed = bool(
        np.all(np.isfinite(native_rollouts))
        and np.median(semigroup_errors) <= median_tol
        and np.percentile(semigroup_errors, 95) <= p95_tol
        and np.percentile(long_closure, 95) <= closure_tol
    )
    accuracy_passed = core._passes_reference_accuracy_gate(
        "solver_self_refined",
        first_interval_error=float(np.percentile(native_errors[:, 1], 95)),
        first_interval_tolerance=float(
            run["evaluation"].get("first_interval_error_tolerance", 0.2)
        ),
        native_long_error=float(np.percentile(native_final_errors, 95)),
        native_long_tolerance=float(
            run["evaluation"].get("native_long_error_tolerance", 1.0)
        ),
    )
    pairs_started = time.perf_counter()
    if payload.get("generate_supervised_pairs", True):
        pairs = core._make_supervised_inputs(t, ctx, train, frame_steps=frame_steps)
        pair_wall = time.perf_counter() - pairs_started
    elif single_ic:
        pairs = np.empty((0, *train.shape[1:]), dtype=train.dtype)
        pair_wall = 0.0
    else:
        raise ValueError(
            "only single-IC held-out preparation may skip supervised pairs"
        )
    seen_count = min(
        int(run["evaluation"].get("seen_ic_trajectories", len(eval_seeds))),
        len(train_seeds),
    )
    horizon = min(train.shape[1] - 1, native_errors.shape[1] - 1)
    # Match the original benchmark's squared mean-native-error normalization.
    loss_scale = max(
        float(
            np.mean(np.mean(native_errors[:seen_count], axis=0)[1 : horizon + 1] ** 2)
        ),
        float(run["training"].get("loss_scale_floor", 1e-6)),
    )
    if run["training"].get("loss_normalization", "target_energy") == "target_energy":
        loss_scale = 1.0
    elif run["training"].get("loss_normalization") != "solver_baseline":
        raise ValueError("unknown loss normalization")
    np.savez_compressed(
        out / "dataset.npz",
        train=train,
        train_rollouts=train_rollouts,
        reference=references,
        supervised_inputs=pairs,
        native_rollouts=native_rollouts[len(train_seeds) :],
        native_errors=native_errors[len(train_seeds) :],
        train_native_errors=native_errors[: len(train_seeds)],
        train_native_rollouts=native_rollouts[: len(train_seeds)],
        train_seeds=np.asarray(train_seeds),
        eval_seeds=np.asarray(eval_seeds),
    )
    metadata = {
        "identity": identity(payload),
        "dataset_sha256": file_hash(out / "dataset.npz"),
        "dataset_hash": dataset_hash,
        "dataset_role": payload.get("dataset_role", "validation"),
        "single_ic": single_ic,
        "selection_sha256": payload.get("selection_sha256"),
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "reference_audit": audit,
        "native_state_threading": core._supports_native_state(t),
        "velocity_scale": velocity_scale,
        "training_loss_scale": loss_scale,
        "semigroup_errors": semigroup_errors,
        "long_closure_errors": long_closure,
        "native_final_errors": native_final_errors,
        "native_first_interval_errors": native_errors[:, 1].tolist(),
        "semigroup_median_tolerance": median_tol,
        "semigroup_p95_tolerance": p95_tol,
        "long_closure_tolerance": closure_tol,
        "closure_passed": closure_passed,
        "accuracy_gate_passed": bool(accuracy_passed),
        "supervised_dataset_wall_time_s": pair_wall,
        "dataset_preparation_wall_time_s": time.perf_counter() - started,
        "completed": True,
        "admitted": bool(
            audit["eligible_for_corrector_training"]
            and closure_passed
            and accuracy_passed
            and np.all(np.isfinite(pairs))
        ),
    }
    (out / "dataset.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def assemble(payload: dict, out: Path) -> dict:
    """Join admitted reference shards in a predetermined IC order on a CPU job.

    Training arrays and normalization come only from the declared first shard.
    Repeated training trajectories in other shards are not consumed. Their
    exact agreement with the anchor's corresponding training arrays is checked;
    a mismatch is retained as a failed assembly, never averaged away.
    """
    started = time.perf_counter()
    directories = [Path(path) for path in payload["shard_dirs"]]
    if not directories:
        raise ValueError("assembly requires at least one shard")
    desired = list(map(int, payload["run"]["dataset"]["test_seeds"]))
    wanted_train = list(map(int, payload["run"]["dataset"]["train_seeds"]))
    if len(set(desired)) != len(desired):
        raise ValueError("duplicate requested evaluation seeds")
    shards = []
    for directory in directories:
        metadata = json.loads((directory / "dataset.json").read_text())
        configured = {
            **payload,
            "dataset_path": str(directory / "dataset.npz"),
            "dataset_metadata_path": str(directory / "dataset.json"),
            "dataset_sha256": metadata["dataset_sha256"],
        }
        arrays, metadata = load_dataset(configured)
        if metadata["dataset_role"] != payload.get("dataset_role", "validation"):
            raise ValueError("dataset shard role mismatch")
        if metadata.get("selection_sha256") != payload.get("selection_sha256"):
            raise ValueError("dataset shard selection mismatch")
        shards.append((arrays, metadata))
    anchor, anchor_meta = shards[0]
    if list(anchor["train_seeds"]) != wanted_train:
        raise ValueError("anchor training seeds mismatch")
    seed_to_train = {int(seed): i for i, seed in enumerate(anchor["train_seeds"])}
    seed_to_reference = {}
    for arrays, _metadata in shards:
        for i, seed in enumerate(arrays["train_seeds"]):
            if int(seed) not in seed_to_train:
                raise ValueError("shard training seed absent from anchor")
            ai = seed_to_train[int(seed)]
            for key in ["train", "train_rollouts", "supervised_inputs"]:
                if not np.array_equal(arrays[key][i], anchor[key][ai]):
                    raise ValueError(
                        f"shared training arrays differ across shards: {int(seed)} {key}"
                    )
        for i, seed in enumerate(arrays["eval_seeds"]):
            seed = int(seed)
            if seed in seed_to_reference:
                raise ValueError("duplicate evaluation IC across shards")
            seed_to_reference[seed] = (arrays, i)
    if set(seed_to_reference) != set(desired):
        raise ValueError("missing or unexpected evaluation ICs in assembly")
    combined = {
        key: anchor[key]
        for key in [
            "train",
            "train_rollouts",
            "supervised_inputs",
            "train_native_errors",
            "train_seeds",
        ]
    }
    for key in ["reference", "native_rollouts", "native_errors"]:
        combined[key] = np.stack(
            [
                seed_to_reference[seed][0][key][seed_to_reference[seed][1]]
                for seed in desired
            ]
        )
    combined["eval_seeds"] = np.asarray(desired)
    np.savez_compressed(out / "dataset.npz", **combined)
    metadata = {
        **anchor_meta,
        "eval_seeds": desired,
        "dataset_sha256": file_hash(out / "dataset.npz"),
        "shards": [
            {
                "directory": str(directory),
                "dataset_sha256": item[1]["dataset_sha256"],
                "metadata_sha256": file_hash(directory / "dataset.json"),
                "eval_seeds": item[1]["eval_seeds"],
            }
            for directory, item in zip(directories, shards, strict=True)
        ],
        "reference_audits_by_shard": [item[1]["reference_audit"] for item in shards],
        "dataset_preparation_wall_time_s": sum(
            item[1]["dataset_preparation_wall_time_s"] for item in shards
        ),
        "supervised_dataset_wall_time_s": anchor_meta["supervised_dataset_wall_time_s"],
        "assembly_wall_time_s": time.perf_counter() - started,
        "shared_training_arrays_identical": True,
        "completed": True,
        "admitted": True,
    }
    # These gate arrays are per-shard (including their repeated training ICs),
    # not misleadingly presented as only the anchor's complete evaluation set.
    for key in [
        "semigroup_errors",
        "long_closure_errors",
        "native_final_errors",
        "native_first_interval_errors",
    ]:
        metadata[key + "_by_shard"] = [item[1][key] for item in shards]
        metadata.pop(key, None)
    metadata.pop("reference_audit", None)
    metadata.pop("dataset_hash", None)
    (out / "dataset.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def deserialize_model(path: str, sha: str, metadata: dict) -> Any:
    """Deserialize only the declared immutable checkpoint."""
    if file_hash(path) != sha or metadata["model_sha256"] != sha:
        raise ValueError("checkpoint SHA256 mismatch")
    model = init_corrector(
        jax.random.PRNGKey(int(metadata["model_seed"])), **metadata["model_spec"]
    )
    return eqx.tree_deserialise_leaves(path, model)


def evaluate_model(
    t: Any,
    ctx: Any,
    model: Any,
    arrays: dict,
    metadata: dict,
    model_metadata: dict,
    indices: list[int],
) -> tuple[dict, dict]:
    """Evaluate all requested ICs without model selection or training."""
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(i < 0 or i >= len(arrays["reference"]) for i in indices)
    ):
        raise ValueError("evaluation indices must be nonempty, unique and in range")
    started = time.perf_counter()
    references = arrays["reference"][indices]
    evaluated = core._evaluate_reference_set(
        t,
        ctx,
        model,
        references,
        frame_steps=int(ctx.phys["steps"]),
        velocity_scale=model_metadata["velocity_scale"],
        corrected=True,
    )
    errors = evaluated.errors
    finite = bool(
        np.all(np.isfinite(errors)) and np.all(np.isfinite(evaluated.first_rollout))
    )
    fields = {
        "error_corrected": errors,
        "error_uncorrected": arrays["native_errors"][indices],
        "evaluation_seeds": arrays["eval_seeds"][indices],
        "reference_rollout": references[0],
        "rollout_corrected": evaluated.first_rollout,
        "rollout_uncorrected": arrays["native_rollouts"][indices[0]],
        "evaluation_times": np.arange(references.shape[1])
        * int(ctx.phys["steps"])
        * float(ctx.phys["dt"]),
    }
    metrics = {
        "evaluation_indices": indices,
        "evaluation_seeds": fields["evaluation_seeds"].tolist(),
        "error_by_ic": errors.tolist(),
        "mean_rollout_error": float(np.mean(errors[:, 1:])),
        "final_rollout_error": float(np.mean(errors[:, -1])),
        "uncorrected_mean_rollout_error": float(
            np.mean(fields["error_uncorrected"][:, 1:])
        ),
        "evaluation_wall_time_s": time.perf_counter() - started,
        "evaluation_finite": finite,
        "evaluation_dataset_sha256": metadata["dataset_sha256"],
    }
    return metrics, fields


def train(t: Any, ctx: Any, payload: dict, out: Path) -> dict:
    """Fit one arm from scratch or a verified supervised start, then validate."""
    arrays, metadata = load_dataset(payload)
    if metadata["dataset_role"] != "validation":
        raise ValueError("training requires development data, never held-out test data")
    training = dict(payload["run"]["training"])
    arm = payload["arm"]
    if arm not in {"full", "stopped", "supervised"}:
        raise ValueError("unknown training arm")
    model_seed = int(payload["model_seed"])
    initial_model = None
    pretrain_cost = 0.0
    pretrain_sha = None
    if payload.get("initial_model_path"):
        previous = json.loads(Path(payload["pretrain_outcome_path"]).read_text())
        if (
            previous["identity"] != identity(payload)
            or previous["training_dataset_sha256"] != metadata["dataset_sha256"]
            or previous["model_seed"] != model_seed
            or previous["arm"] != "supervised"
            or previous["model_spec"] != model_spec(training)
            or not previous["completed"]
            or not previous["admitted"]
            or float(previous["training"]["lr"]) != 1e-5
            or int(previous["training"]["max_updates"]) != 1000
            or int(previous["training"]["unroll"]) != 8
            or previous["training"]["seed"] != int(training.get("seed", 2026))
        ):
            raise ValueError("supervised warm-start provenance mismatch")
        initial_model = deserialize_model(
            payload["initial_model_path"], payload["initial_model_sha256"], previous
        )
        pretrain_cost = float(previous["training_wall_time_s"])
        pretrain_sha = previous["model_sha256"]
    training["seed"] = int(training.get("seed", 2026))
    # Identical model seed and sampling schedule; supervision uses fixed pairs.
    if arm == "supervised":
        training["loss_mode"] = "mean"
        training["solver_loss_weight"] = 0.0
    gradient_checks = []
    continuation = None
    prior_training_wall = 0.0
    continuation_config = payload.get("continuation")
    if continuation_config:
        continuation = TrainingContinuation(
            path=Path(continuation_config["path"]),
            identity={
                **identity(payload),
                "dataset_sha256": metadata["dataset_sha256"],
            },
            checkpoint_every=int(continuation_config.get("checkpoint_every", 100)),
            max_updates_per_allocation=continuation_config.get(
                "max_updates_per_allocation"
            ),
            wall_limit_s=continuation_config.get("wall_limit_s"),
        )
        if continuation.path.exists():
            previous_path = Path(continuation_config["previous_outcome_path"])
            if (
                file_hash(previous_path)
                != continuation_config["previous_outcome_sha256"]
            ):
                raise ValueError("continuation accounting outcome hash mismatch")
            previous = json.loads(previous_path.read_text())
            if (
                not previous.get("resume_required")
                or previous["identity"] != identity(payload)
                or previous["arm"] != arm
                or previous["model_seed"] != model_seed
                or previous["training"] != training
                or previous["training_dataset_sha256"] != metadata["dataset_sha256"]
                or previous["checkpoint_sha256"] != file_hash(continuation.path)
            ):
                raise ValueError("continuation accounting/protocol provenance mismatch")
            prior_training_wall = float(previous["training_wall_time_s"])
    started = time.perf_counter()
    try:
        model, losses, grads, update_times, fd_error, completed = core._train_corrector(
            t,
            ctx,
            arrays["train"],
            frame_steps=int(ctx.phys["steps"]),
            training=training,
            velocity_scale=metadata["velocity_scale"],
            loss_scale=metadata["training_loss_scale"],
            differentiate_solver=arm == "full",
            model_seed=model_seed,
            supervised_inputs=arrays["supervised_inputs"]
            if arm == "supervised"
            else None,
            initial_model=initial_model,
            fd_checks=gradient_checks,
            **({"continuation": continuation} if continuation is not None else {}),
        )
    except TrainingYield as yielded:
        allocation_wall = time.perf_counter() - started
        return {
            "phase": "train",
            "identity": identity(payload),
            "arm": arm,
            "model_seed": model_seed,
            "training": training,
            "training_dataset_sha256": metadata["dataset_sha256"],
            "completed": False,
            "admitted": False,
            "resume_required": True,
            "optimizer_updates": yielded.updates,
            "checkpoint_sha256": file_hash(continuation.path),
            "allocation_training_wall_time_s": allocation_wall,
            "training_wall_time_s": prior_training_wall + allocation_wall,
            "continuation_scope": "Completed-update boundary; unchanged model, Adam, RNG and FD state",
        }
    jax.block_until_ready(eqx.filter(model, eqx.is_array))
    allocation_training_wall = time.perf_counter() - started
    training_wall = prior_training_wall + allocation_training_wall
    eqx.tree_serialise_leaves(out / "model.eqx", model)
    gradient_passed = bool(
        arm != "full"
        or (fd_error is not None and np.isfinite(fd_error) and fd_error < 0.05)
    )
    result = {
        "phase": "train",
        "identity": identity(payload),
        "arm": arm,
        "model_seed": model_seed,
        "model_spec": model_spec(training),
        "training": training,
        "model_sha256": file_hash(out / "model.eqx"),
        "initial_model_sha256": pretrain_sha,
        "training_dataset_sha256": metadata["dataset_sha256"],
        "velocity_scale": metadata["velocity_scale"],
        "training_loss_scale": metadata["training_loss_scale"],
        "training_wall_time_s": training_wall,
        "allocation_training_wall_time_s": allocation_training_wall,
        "resume_required": False,
        "pretraining_wall_time_s": pretrain_cost,
        "supervised_dataset_wall_time_s": metadata["supervised_dataset_wall_time_s"],
        "common_dataset_preparation_wall_time_s": metadata[
            "dataset_preparation_wall_time_s"
        ],
        "method_training_including_pretrain_and_pairs_s": training_wall
        + pretrain_cost
        + (
            metadata["supervised_dataset_wall_time_s"]
            if arm == "supervised" or pretrain_sha
            else 0
        ),
        "optimizer_updates": len(losses),
        "gradient_checks": gradient_checks,
        "gradient_relative_error": fd_error,
        "gradient_passed": gradient_passed,
        "completed": bool(completed and len(losses) == int(training["max_updates"])),
        "admitted": False,
        "fit_scope": "independent fit; Adam resets only at declared initialization",
    }
    fields = {
        "loss": np.asarray(losses),
        "grad_norm": np.asarray(grads),
        "update_time_s": np.asarray(update_times),
    }
    # Persist training evidence before potentially failing rollout evaluation.
    (out / "outcome.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out / "fields.npz", **fields)
    if not payload.get("evaluate_after_training", True):
        result["evaluation_deferred"] = True
        result["evaluation_seeds"] = []
        result["prepared_evaluation_seeds"] = metadata["eval_seeds"]
        result["admitted"] = bool(
            result["completed"] and metadata["admitted"] and gradient_passed
        )
        return result
    metrics, evaluation_fields = evaluate_model(
        t, ctx, model, arrays, metadata, result, list(range(len(arrays["reference"])))
    )
    result.update(metrics)
    result["admitted"] = bool(
        result["completed"]
        and metadata["admitted"]
        and gradient_passed
        and metrics["evaluation_finite"]
    )
    fields.update(evaluation_fields)
    np.savez_compressed(out / "fields.npz", **fields)
    return result


def evaluate(t: Any, ctx: Any, payload: dict, out: Path) -> dict:
    """Evaluate a frozen chosen checkpoint on a fixed batch of held-out ICs."""
    arrays, metadata = load_dataset(payload)
    previous = json.loads(Path(payload["model_outcome_path"]).read_text())
    if previous["identity"] != identity(payload) or not previous["admitted"]:
        raise ValueError("evaluation model identity/admission mismatch")
    if metadata["dataset_role"] == "test" and metadata.get(
        "selection_sha256"
    ) != payload.get("selection_sha256"):
        raise ValueError("test selection identity mismatch")
    if set(metadata["eval_seeds"]) & set(previous["evaluation_seeds"]):
        raise ValueError("held-out evaluation overlaps development ICs")
    model = deserialize_model(payload["model_path"], payload["model_sha256"], previous)
    metrics, fields = evaluate_model(
        t, ctx, model, arrays, metadata, previous, list(payload["eval_indices"])
    )
    np.savez_compressed(out / "fields.npz", **fields)
    return {
        **metrics,
        "phase": "evaluate",
        "arm": previous["arm"],
        "model_seed": previous["model_seed"],
        "identity": identity(payload),
        "model_sha256": previous["model_sha256"],
        "training_dataset_sha256": previous["training_dataset_sha256"],
        "selection_sha256": payload.get("selection_sha256"),
        "completed": True,
        "admitted": bool(metadata["admitted"] and metrics["evaluation_finite"]),
    }


def main() -> None:
    """Cluster entry point, preserving failed outcomes and partial evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "assemble", "train", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text())
    phase = args.phase or payload["phase"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocol.json").write_text(json.dumps(payload, indent=2))
    started = time.perf_counter()
    try:
        if phase == "assemble":
            result = assemble(payload, args.out)
            result["job_wall_time_s"] = time.perf_counter() - started
            (args.out / "outcome.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2), flush=True)
            return
        if not args.url:
            raise ValueError("solver phases require --url")
        if jax.default_backend() != "gpu":
            raise RuntimeError("solver experiments require an allocated GPU")
        cfg = get_config("ns-grid")
        spec = next(
            s for s in cfg.solvers if s.key == payload["solver"].replace("-", "_")
        )
        ctx = SimpleNamespace(
            name=spec.name,
            make_inputs=cfg.make_inputs,
            phys=payload["run"]["physics"],
            domain_extent=float(payload.get("domain_extent", 2 * np.pi)),
            run=payload["run"],
            output_key="result",
        )
        os.environ["MOSAIC_REFERENCE_CACHE_ID"] = (
            payload["source_sha256"] + ":" + payload["image"]
        )
        with Tesseract.from_url(args.url, timeout=(30, 1200)) as solver:
            result = {"prepare": prepare, "train": train, "evaluate": evaluate}[phase](
                solver, ctx, payload, args.out
            )
    except Exception as exc:
        import traceback

        traceback.print_exc()
        partial = args.out / "outcome.json"
        result = json.loads(partial.read_text()) if partial.exists() else {}
        result.update(
            {
                "completed": False,
                "admitted": False,
                "failure": f"{type(exc).__name__}: {exc}",
            }
        )
    result["job_wall_time_s"] = time.perf_counter() - started
    (args.out / "outcome.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
