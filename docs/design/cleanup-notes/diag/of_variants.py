"""Build one OpenFOAM image per channel-setting variant (from main) and run forward/cylinder.

Usage: python of_variants.py <variant>...   (variants listed in VARIANTS)
"""

import os
import subprocess
import sys
import time
from pathlib import Path

MAIN = Path.home() / "localfiles/mosaic"
WT = Path.home() / "localfiles/worktrees/mosaic-ofdiag"
API = "mosaic/tesseracts/navier-stokes-grid/openfoam/tesseract_api.py"
P_GAMG = """        solver          GAMG;
        smoother        DICGaussSeidel;
        tolerance       1e-08;
        relTol          0.01;"""
U_TOL = """        smoother        symGaussSeidel;
        tolerance       1e-08;
        relTol          0;
    }
    UFinal { $U; }"""
VARIANTS = {
    "base": [],
    "u-tol-1e-5": [(U_TOL, U_TOL.replace("1e-08", "1e-05"))],
    "p-gamg-1e-6": [(P_GAMG, P_GAMG.replace("1e-08", "1e-06").replace("0.01", "0.1"))],
    "p-pcg": [
        (
            P_GAMG,
            """        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-08;
        relTol          0.01;""",
        )
    ],
    "nonorth-0": [
        ("    nNonOrthogonalCorrectors 1;\n}", "    nNonOrthogonalCorrectors 0;\n}")
    ],
    "no-grading": [
        ("_RADIAL_GRADING = 0.5", "_RADIAL_GRADING = 1.0"),
        ("_WAKE_GRADING = 10.0", "_WAKE_GRADING = 1.0"),
    ],
    "serial": [("_CELLS_PER_PROCESS = 10_000", "_CELLS_PER_PROCESS = 10**12")],
}
for v in sys.argv[1:]:
    subprocess.run(["git", "-C", str(WT), "checkout", "-q", "--", API], check=True)
    src = (WT / API).read_text()
    for a, b in VARIANTS[v]:
        assert src.count(a) == 1, (v, a)
        src = src.replace(a, b)
    (WT / API).write_text(src)
    env = {**os.environ, "TMPDIR": "/nvme/scratch/tmp"}
    subprocess.run(
        [
            str(MAIN / ".venv/bin/tesseract"),
            "build",
            "-t",
            v,
            "mosaic/tesseracts/navier-stokes-grid/openfoam",
        ],
        cwd=WT,
        check=True,
        env=env,
        capture_output=True,
    )
    subprocess.run(
        [
            "docker",
            "tag",
            f"openfoam_navier_stokes_grid:{v}",
            "openfoam_navier_stokes_grid:latest",
        ],
        check=True,
    )
    out = f"/nvme/scratch/results/of-{v}"
    t = time.time()
    subprocess.run(
        [
            str(MAIN / ".venv/bin/mosaic"),
            "run",
            "-p",
            "ns-grid",
            "-e",
            "forward/cylinder",
            "-s",
            "OpenFOAM",
            "--no-build",
            "--no-plots",
            "-o",
            out,
        ],
        cwd=MAIN,
        env=env,
        capture_output=True,
    )
    print(v, f"total {time.time() - t:.0f}s", flush=True)
subprocess.run(["git", "-C", str(WT), "checkout", "-q", "--", API], check=True)
subprocess.run(
    [
        "docker",
        "tag",
        "openfoam_navier_stokes_grid:main",
        "openfoam_navier_stokes_grid:latest",
    ],
    check=True,
)
