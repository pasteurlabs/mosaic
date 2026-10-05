"""Create an explicitly hashed experimental adapter without editing Euler source."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    """Copy and patch only the declared input adapter into a new candidate file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--centered-projection", action="store_true")
    parser.add_argument("--skew-advection", action="store_true")
    args = parser.parse_args()
    raw = args.base.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.expected_sha256
    source = raw.decode()
    overlay_path = Path(__file__).with_name("warp_rk3_overlay.py")
    overlay = overlay_path.read_text()
    if args.centered_projection:
        overlay += (
            "\n"
            + Path(__file__)
            .with_name("warp_centered_projection_overlay.py")
            .read_text()
        )
    if args.skew_advection:
        if not args.centered_projection:
            raise ValueError("skew variant requires matching centered projection")
        overlay = overlay.replace(
            'base["tentative_vel_2d_kernel"]', 'base["skew_tentative_vel_2d_kernel"]'
        )
        overlay += "\n" + Path(__file__).with_name("warp_skew_overlay.py").read_text()
    anchor = "# Schema definitions\n"
    assert source.count(anchor) == 1
    source = source.replace(anchor, overlay + "\n\n" + anchor)
    old = "        result = ns2d_solve_forward(\n"
    assert source.count(old) == 1
    source = source.replace(
        old,
        "        result = (ns2d_ssprk3_forward if _rk3_enabled() else ns2d_solve_forward)(\n",
    )
    old = "        ) = ns2d_solve_tape(\n"
    assert source.count(old) == 1
    source = source.replace(
        old, "        ) = (ns2d_ssprk3_tape if _rk3_enabled() else ns2d_solve_tape)(\n"
    )
    old = '    """warp-ns supports only fully-periodic flows."""\n'
    assert source.count(old) == 1
    source = source.replace(
        old,
        old
        + "    if _rk3_enabled() and _is_3d(np.asarray(inputs.v0)):\n"
        + '        raise NotImplementedError("experimental SSPRK3 is 2D only")\n',
    )
    compile(source, str(args.out), "exec")
    args.out.write_text(source)
    metadata = {
        "base_sha256": args.expected_sha256,
        "overlay_sha256": hashlib.sha256(overlay.encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "advection": "centered skew-symmetric"
        if args.skew_advection
        else "centered convective",
        "activation": "MOSAIC_WARP_INTEGRATOR=ssprk3",
        "default": "euler",
        "projection": (
            "matched centered D/G symbol; joint null modes preserved"
            if args.centered_projection
            else "unchanged continuous-k2 with centered divergence/gradient"
        ),
        "scope": "experimental 2D stage integrator; no full-system third-order claim",
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
