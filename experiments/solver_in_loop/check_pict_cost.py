"""Profile PICT setup, forward replay and adjoint costs inside its GPU image."""

from __future__ import annotations

import argparse
import cProfile
import importlib.util
import json
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Time synchronized phases separately from uninstrumented warm RPC work."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    modules = {"original": _load(Path("/tesseract/tesseract_api.py"), "pict_original")}
    if args.candidate:
        modules["candidate"] = _load(args.candidate, "pict_candidate")
    rng = np.random.default_rng(116)
    n = 64
    x, y = np.meshgrid(
        np.arange(n) * 2 * np.pi / n, np.arange(n) * 2 * np.pi / n, indexing="ij"
    )
    velocity = np.stack([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)], axis=-1)[
        :, :, None, :
    ].astype(np.float32)
    cotangent = rng.normal(size=velocity.shape).astype(np.float32)
    parameters = {
        "v0": velocity,
        "viscosity": np.array([0.001], np.float32),
        "dt": np.array([0.02], np.float32),
        "steps": 4,
        "domain_extent": 2 * np.pi,
    }
    inputs = {name: api.InputSchema(**parameters) for name, api in modules.items()}

    def evaluate(name: str, mode: str) -> dict:
        api = modules[name]
        if mode == "apply":
            return api.apply(inputs[name])
        return api.vector_jacobian_product(
            inputs[name], {"v0"}, {"result"}, {"result": cotangent}
        )

    # Alternate both image implementations using identical fields, before any
    # instrumentation. Synchronize boundaries to measure completed GPU work.
    timings = {name: {mode: [] for mode in ["apply", "vjp"]} for name in modules}
    outputs = {}
    for sample in range(9):
        order = list(modules) if sample % 2 == 0 else list(reversed(modules))
        for name in order:
            for mode in ["apply", "vjp"]:
                torch.cuda.synchronize()
                started = time.perf_counter()
                outputs[name, mode] = evaluate(name, mode)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if sample:
                    timings[name][mode].append(elapsed)
                print(
                    json.dumps(
                        {
                            "sample": sample,
                            "implementation": name,
                            "mode": mode,
                            "seconds": elapsed,
                        }
                    ),
                    flush=True,
                )
    parity = {}
    if "candidate" in modules:
        for mode in ["apply", "vjp"]:
            for key, value in outputs["original", mode].items():
                candidate = outputs["candidate", mode][key]
                parity[f"{mode}/{key}"] = {
                    "passed": bool(np.allclose(candidate, value, rtol=1e-5, atol=1e-6)),
                    "max_absolute_difference": float(
                        np.max(np.abs(np.asarray(candidate) - np.asarray(value)))
                    ),
                }

    if args.audit and "candidate" in modules:
        for seed, size, dt, nu, steps in [
            (0, 32, 0.01, 0.001, 1),
            (1, 64, 0.02, 0.001, 4),
            (2, 64, 0.01, 0.002, 4),
            (3, 32, 0.02, 0.001, 4),
        ]:
            # Smooth random stream functions produce periodic divergence-free ICs.
            random = np.random.default_rng(seed)
            k = np.fft.fftfreq(size) * size
            kx, ky = np.meshgrid(k, k, indexing="ij")
            spectrum = np.fft.fft2(random.normal(size=(size, size))) * np.exp(
                -0.5 * ((np.sqrt(kx**2 + ky**2) - 4) / 0.5) ** 2
            )
            u = np.fft.ifft2(1j * ky * spectrum).real
            v = np.fft.ifft2(-1j * kx * spectrum).real
            field = np.stack([u, v], axis=-1)[:, :, None, :]
            field = (field * (0.5 / np.sqrt(np.mean(field**2)))).astype(np.float32)
            ct = random.normal(size=field.shape).astype(np.float32)
            values = {}
            for name, candidate_api in modules.items():
                request = candidate_api.InputSchema(
                    v0=field,
                    viscosity=np.array([nu], np.float32),
                    dt=np.array([dt], np.float32),
                    steps=steps,
                    domain_extent=2 * np.pi,
                )
                values[name] = {
                    "apply": candidate_api.apply(request),
                    "vjp": candidate_api.vector_jacobian_product(
                        request, {"v0", "viscosity"}, {"result"}, {"result": ct}
                    ),
                }
            for mode in ["apply", "vjp"]:
                for key, value in values["original"][mode].items():
                    candidate = values["candidate"][mode][key]
                    parity[
                        f"seed{seed}-n{size}-dt{dt}-nu{nu}-steps{steps}/{mode}/{key}"
                    ] = {
                        "passed": bool(
                            np.allclose(candidate, value, rtol=1e-5, atol=1e-6)
                        ),
                        "max_absolute_difference": float(
                            np.max(np.abs(np.asarray(candidate) - np.asarray(value)))
                        ),
                    }
    phases = defaultdict(list)
    api = modules["original"]
    originals = []

    def instrument(owner: Any, attribute: str, label: str) -> None:
        original = getattr(owner, attribute)
        originals.append((owner, attribute, original))

        def measured(*values: Any, **kwargs: Any) -> Any:
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = original(*values, **kwargs)
            torch.cuda.synchronize()
            phases[label].append(time.perf_counter() - started)
            return result

        setattr(owner, attribute, measured)

    instrument(api, "_make_domain", "domain_factory")
    instrument(api, "_run_pict", "complete_forward_setup_and_solve")
    instrument(api.PISOtorch_simulation.Simulation, "run", "simulation_run")
    instrument(torch.autograd, "grad", "adjoint")
    for mode in ["apply", "vjp"]:
        for _ in range(3):
            evaluate("original", mode)
    for owner, attribute, original in reversed(originals):
        setattr(owner, attribute, original)
    profiler = cProfile.Profile()
    profiler.enable()
    evaluate("original", "apply")
    evaluate("original", "vjp")
    profiler.disable()
    stats = pstats.Stats(profiler)
    hot = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:40]
    result = {
        "N": n,
        "steps": 4,
        "warmup_samples": 1,
        "timings": timings,
        "candidate_checks_passed": bool(parity)
        and all(row["passed"] for row in parity.values()),
        "max_absolute_differences": parity,
        "synchronized_phase_seconds": dict(phases),
        "profile": [
            {
                "location": str(key),
                "calls": value[1],
                "self_s": value[2],
                "cumulative_s": value[3],
            }
            for key, value in hot
        ],
    }
    args.out.write_text(json.dumps(result, indent=2))
    profiler.dump_stats(str(args.out.with_suffix(".prof")))
    print(json.dumps(result), flush=True)
    if args.candidate and not result["candidate_checks_passed"]:
        raise AssertionError("Candidate parity failed; raw diagnostics retained")


if __name__ == "__main__":
    main()
