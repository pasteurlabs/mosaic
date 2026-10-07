"""Diagnostic-only PICT forward/AD parity and precision checks on saved fields."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    """Compare exact adapter modes; preserve all directions and precision failures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.dataset.read_bytes()).hexdigest() == args.dataset_sha256
    spec = importlib.util.spec_from_file_location("pict_probe_adapter", args.adapter)
    api = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = api
    spec.loader.exec_module(api)
    assert torch.cuda.is_available()
    with np.load(args.dataset, allow_pickle=False) as archive:
        train = archive["train"].copy()
    rng = np.random.RandomState(2026)
    trajectory = int(rng.randint(len(train)))
    start = int(rng.randint(train.shape[1] - 16))
    records = []
    for frame in (start, train.shape[1] - 1):
        original = train[trajectory, frame]
        n = original.shape[0]
        scale = n / (2 * np.pi)
        for dtype in (torch.float32, torch.float64):
            for steps in (1, 4):
                row = {"frame": frame, "steps": steps, "dtype": str(dtype)}
                try:

                    def advance(
                        value: np.ndarray,
                        differentiable: bool,
                        dtype: torch.dtype = dtype,
                        scale: float = scale,
                        steps: int = steps,
                        n: int = n,
                    ):
                        initial = api._v0_to_pict(
                            value, api._DEVICE, dtype, requires_grad=differentiable
                        )
                        viscosity = torch.tensor(
                            [0.001 * scale], dtype=dtype, device="cpu"
                        )
                        result, _drag, _domain = api._run_pict(
                            initial,
                            viscosity,
                            0.01 * scale,
                            steps,
                            n,
                            2,
                            dtype,
                            differentiable,
                            phys_scale=scale,
                        )
                        return api._pict_to_v0(result, n, 2), initial, result

                    forward, _, _ = advance(original, False)
                    taped, initial, raw_result = advance(original, True)
                    forward_np = forward.detach().cpu().numpy().astype(np.float64)
                    taped_np = taped.detach().cpu().numpy().astype(np.float64)
                    row["forward_tape_max_absolute"] = float(
                        np.max(np.abs(forward_np - taped_np))
                    )
                    row["forward_tape_relative_l2"] = float(
                        np.linalg.norm(forward_np - taped_np)
                        / np.linalg.norm(forward_np)
                    )
                    cotangent = np.random.default_rng(117).normal(size=original.shape)
                    cotangent /= np.linalg.norm(cotangent)
                    gradient = torch.autograd.grad(
                        raw_result,
                        initial,
                        grad_outputs=api._v0_to_pict(cotangent, api._DEVICE, dtype),
                    )[0]
                    grad_np = (
                        api._pict_to_v0(gradient, n, 2)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    row["gradient_finite"] = bool(np.isfinite(grad_np).all())
                    checks = []
                    for seed in (9, 10, 11):
                        direction = np.random.default_rng(seed).normal(
                            size=original.shape
                        )
                        direction /= np.linalg.norm(direction)
                        ad = float(np.sum(grad_np * direction))
                        for epsilon in (0.003, 0.001, 0.0003, 0.0001):
                            np_dtype = (
                                np.float32 if dtype == torch.float32 else np.float64
                            )
                            plus, _, _ = advance(
                                (original + epsilon * direction).astype(np_dtype), False
                            )
                            minus, _, _ = advance(
                                (original - epsilon * direction).astype(np_dtype), False
                            )
                            difference = plus.detach().cpu().numpy().astype(
                                np.float64
                            ) - minus.detach().cpu().numpy().astype(np.float64)
                            fd = float(np.sum(difference * cotangent) / (2 * epsilon))
                            checks.append(
                                {
                                    "direction_seed": seed,
                                    "epsilon": epsilon,
                                    "autodiff": ad,
                                    "finite_difference": fd,
                                    "absolute_error": abs(ad - fd),
                                    "relative_error": abs(ad - fd)
                                    / (abs(ad) + abs(fd) + 1e-30),
                                }
                            )
                    row["checks"] = checks
                except Exception as exc:
                    import traceback

                    traceback.print_exc()
                    row["failure"] = f"{type(exc).__name__}: {exc}"
                records.append(row)
                print(json.dumps(row), flush=True)
                args.out.write_text(
                    json.dumps(
                        {
                            "adapter_sha256": hashlib.sha256(
                                args.adapter.read_bytes()
                            ).hexdigest(),
                            "dataset_sha256": args.dataset_sha256,
                            "gpu": torch.cuda.get_device_name(),
                            "torch_version": torch.__version__,
                            "first_window_start": start,
                            "records": records,
                        },
                        indent=2,
                    )
                )


if __name__ == "__main__":
    main()
