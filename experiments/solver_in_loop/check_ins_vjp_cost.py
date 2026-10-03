"""Check requested-gradient equivalence and warm INS recurrent-VJP timing on Slurm."""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np


def load_api(
    path: Path, name: str, julia_source: Path | None = None
) -> types.ModuleType:
    """Load independent Julia modules so both implementations share one process."""
    source = path.read_text().replace('newmodule("ins_ns")', f'newmodule("{name}")')
    if julia_source is not None:
        source = source.replace(
            'jl.include(str(_JULIA_SOURCE / "ns_solver.jl"))',
            f"jl.include({str(julia_source)!r})",
        )
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 -- trusted solver sources, isolated module names
    return module


def main() -> None:
    """Compare outputs, all requested cotangents and the velocity-only fast path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    original = load_api(Path("/tesseract/tesseract_api.py"), "ins_original")
    candidate = load_api(
        args.candidate / "tesseract_api.py",
        "ins_candidate",
        args.candidate / "ns_solver.jl",
    )
    rng = np.random.default_rng(116)
    results = []
    for dimensions, n in [(2, 64), (2, 192), (3, 8)]:
        shape = (n, n, 1, 2) if dimensions == 2 else (n, n, n, 3)
        initial = (rng.normal(size=shape) * 0.02).astype(np.float32)
        initial_out, _, initial_state = original._run(
            initial, 0.001, 0.01, 1, 2 * np.pi, return_state=True
        )
        for continuation in [False, True]:
            velocity = initial_out if continuation else initial
            state = initial_state if continuation else None
            steps = 4 if continuation else 1
            parameters = {
                "v0": velocity,
                "viscosity": np.array([0.001], dtype=np.float32),
                "dt": np.array([0.01], dtype=np.float32),
                "steps": steps,
                "domain_extent": 2 * np.pi,
                "state": state,
                "return_state": True,
            }
            old_inputs = original.InputSchema(**parameters)
            new_inputs = candidate.InputSchema(**parameters)
            old_forward, new_forward = (
                original.apply(old_inputs),
                candidate.apply(new_inputs),
            )
            for key in ["result", "state"]:
                np.testing.assert_allclose(
                    np.asarray(new_forward[key]),
                    np.asarray(old_forward[key]),
                    rtol=1e-6,
                    atol=1e-7,
                )
            cotangents = {
                key: rng.normal(size=shape).astype(np.float32)
                for key in ["result", "state"]
            }
            requested = {"v0", "dt", "viscosity"} | (
                {"state"} if continuation else set()
            )
            expected = original.vector_jacobian_product(
                old_inputs, requested, set(cotangents), cotangents
            )
            complete = candidate.vector_jacobian_product(
                new_inputs, requested, set(cotangents), cotangents
            )
            for key in requested:
                np.testing.assert_allclose(
                    complete[key], expected[key], rtol=1e-5, atol=1e-6
                )
            requested.remove("viscosity")
            fast = candidate.vector_jacobian_product(
                new_inputs, requested, set(cotangents), cotangents
            )
            assert set(fast) == requested
            for key in requested:
                np.testing.assert_array_equal(fast[key], complete[key])
            timings = {"original": [], "candidate": []}
            # Interleave warmed calls to reduce order/thermal bias.
            for _ in range(5):
                for name, api, inputs in [
                    ("original", original, old_inputs),
                    ("candidate", candidate, new_inputs),
                ]:
                    started = time.perf_counter()
                    api.vector_jacobian_product(
                        inputs, requested, set(cotangents), cotangents
                    )
                    timings[name].append(time.perf_counter() - started)
            row = {
                "dimensions": dimensions,
                "N": n,
                "continuation": continuation,
                "steps": steps,
                "requested": sorted(requested),
                "seconds": timings,
                "median_speedup": float(
                    np.median(timings["original"]) / np.median(timings["candidate"])
                ),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    args.out.write_text(
        json.dumps({"checks_passed": True, "comparisons": results}, indent=2)
    )


if __name__ == "__main__":
    main()
