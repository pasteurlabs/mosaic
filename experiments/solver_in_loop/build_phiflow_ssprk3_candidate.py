"""Build an explicit experimental PhiFlow adapter copy; never edit its source.

Only the obstacle-free periodic branch changes. The external forcing remains
Strang split, so this is not a third-order claim for the complete forced map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_candidate(source: str) -> str:
    """Replace periodic Euler stepping with projected combined-RHS SSPRK3."""
    anchor = """    else:
        # Use explicit CG pressure solver to prevent numerical divergence in 3D"""
    replacement = """    elif periodic and not obstacles:
        # EXPERIMENTAL SSPRK3 variant. Preserve the frozen Euler image separately.
        # Advection and diffusion increments use the same stage input. Reusing
        # the old sequential advection-then-diffusion map would add a dt**2
        # cross term and would not be an Euler stage of this semi-discrete RHS.
        def euler_stage(face_arr: jnp.ndarray) -> jnp.ndarray:
            vel = faces_to_staggered(face_arr)
            advection = staggered_to_faces(advect.differential(vel, vel))
            diffused = staggered_to_faces(diffuse.explicit(vel, viscosity, dt))
            return project_periodic_faces(diffused + dt * advection)

        def step(face_arr: jnp.ndarray, _: None) -> tuple[jnp.ndarray, None]:
            base = project_periodic_faces(face_arr)
            stage1 = euler_stage(base)
            stage2 = 0.75 * base + 0.25 * euler_stage(stage1)
            final = (1.0 / 3.0) * base + (2.0 / 3.0) * euler_stage(stage2)
            return final, None

    else:
        # Use explicit CG pressure solver to prevent numerical divergence in 3D"""
    if source.count(anchor) != 1:
        raise ValueError("expected exactly one original PhiFlow Euler branch")
    return source.replace(anchor, replacement)


def main() -> None:
    """Write the candidate and exact source hashes for a separate experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    original = args.source.read_bytes()
    candidate = build_candidate(original.decode()).encode()
    args.out.write_bytes(candidate)
    metadata = {
        "base_adapter_sha256": hashlib.sha256(original).hexdigest(),
        "candidate_adapter_sha256": hashlib.sha256(candidate).hexdigest(),
        "variant": "periodic_projected_combined_rhs_ssprk3",
        "default_image_changed": False,
        "scope": "experimental adapter copy; periodic obstacle-free flow only; unchanged external Strang forcing",
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
