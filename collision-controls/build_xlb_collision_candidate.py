"""Build a diagnostic XLB velocity-bound variant; preserve original image."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    """Apply identical static substep selection in forward and derivative paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--velocity-bound", type=float, default=8.0)
    parser.add_argument("--collision", choices=["original", "bgk"], default="original")
    parser.add_argument("--reassociate", action="store_true")
    args = parser.parse_args()
    original = args.base.read_bytes()
    expected = "4a7a9bf8bb0943ee4bb5c8fad0c9a1e02d3c8539d62eb82a97fac15a46673377"
    if hashlib.sha256(original).hexdigest() != expected:
        raise ValueError("Unexpected original XLB adapter")
    if not 1 <= args.velocity_bound <= 16:
        raise ValueError("Diagnostic velocity bound must be between1 and16")
    source = original.decode()
    for old, new in [
        ("_u_max_conservative = 1.0", f"_u_max_conservative = {args.velocity_bound!r}"),
        (
            "_Ma_full_run = 1.0 * _scale_concrete",
            f"_Ma_full_run = {args.velocity_bound!r} * _scale_concrete",
        ),
    ]:
        if source.count(old) != 1:
            raise ValueError(f"Expected one source anchor: {old}")
        source = source.replace(old, new)
    if args.collision == "bgk":
        for old, new in [
            (
                '_collision_kind = "kbc" if _needs_kbc else "bgk"',
                '_collision_kind = "bgk"',
            ),
            ('_ck = "kbc" if _needs_kbc_concrete else "bgk"', '_ck = "bgk"'),
        ]:
            if source.count(old) != 1:
                raise ValueError(f"Expected one collision source anchor: {old}")
            source = source.replace(old, new)
    if args.reassociate:
        old = "f0 = f_previous + xlb_eq(rho0, u_corrected) - xlb_eq(rho0, u_previous)"
        assert source.count(old) == 1
        source = source.replace(
            old,
            "f0 = f_previous + (xlb_eq(rho0, u_corrected) - xlb_eq(rho0, u_previous))",
        )
    compile(source, str(args.out), "exec")
    args.out.write_text(source)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "base_adapter_sha256": expected,
                "candidate_adapter_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "static_physical_velocity_bound": args.velocity_bound,
                "collision_variant": args.collision,
                "reassociated_equilibrium_increment": args.reassociate,
                "activation_environment": {"XLB_SUB_K_DISABLE": "0"},
                "scope": (
                    "Diagnostic numerical variant; identical forward/derivative substep selection. "
                    "No admission override. Original image unchanged."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
