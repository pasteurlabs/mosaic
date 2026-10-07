"""Build a hashed PICT precision diagnostic without modifying the original adapter."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    """Keep canonical float32 RPC outputs while varying internal solve precision."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dtype", required=True, choices=["float32", "float64"])
    parser.add_argument("--tolerance", required=True, type=float)
    parser.add_argument("--double-fallback", action="store_true")
    args = parser.parse_args()
    original = args.base.read_bytes()
    expected = "e0eb8b50cdeb00d70d442e2305f1dcc9bfcc47e814c2b6ce1504e8df391ededf"
    if hashlib.sha256(original).hexdigest() != expected:
        raise ValueError("Unexpected original PICT adapter")
    if not 0 < args.tolerance < 1e-4:
        raise ValueError("Invalid diagnostic tolerance")
    source = original.decode()
    assert source.count("    dtype = torch.float32\n") == 2
    source = source.replace(
        "    dtype = torch.float32\n", f"    dtype = torch.{args.dtype}\n"
    )
    anchor = "        corrector_steps=4,\n"
    assert source.count(anchor) == 1
    source = source.replace(
        anchor,
        anchor + f"        advection_tol={args.tolerance!r},\n"
        f"        pressure_tol={args.tolerance!r},\n"
        f"        solver_double_fallback={args.double_fallback!r},\n",
    )
    source = source.replace(
        "out_np = _pict_to_v0(result_t, N, ndim).detach().cpu().numpy()",
        "out_np = _pict_to_v0(result_t, N, ndim).detach().cpu().numpy().astype(np.float32)",
    )
    source = source.replace(
        "_pict_to_v0(g, N, ndim).cpu().numpy()",
        "_pict_to_v0(g, N, ndim).cpu().numpy().astype(np.float32)",
    )
    source = source.replace(
        "g.detach().cpu().numpy()\n                    if g is not None",
        "g.detach().cpu().numpy().astype(np.float32)\n                    if g is not None",
    )
    compile(source, str(args.out), "exec")
    args.out.write_text(source)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "base_adapter_sha256": expected,
                "candidate_adapter_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "internal_dtype": args.dtype,
                "advection_tolerance": args.tolerance,
                "pressure_tolerance": args.tolerance,
                "double_fallback": args.double_fallback,
                "rpc_dtype": "float32",
                "scope": "diagnostic-only forward+adjoint consistent precision variant; original image unchanged",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
