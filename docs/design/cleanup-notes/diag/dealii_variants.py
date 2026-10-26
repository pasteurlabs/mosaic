"""Build deal.II variant images from main (tags: ds-uncap, ds-cg, dh-cap, dh-tol8)."""

import os
import subprocess
import sys
from pathlib import Path

MAIN = Path.home() / "localfiles/mosaic"
WT = Path.home() / "localfiles/worktrees/mosaic-dealiidiag"
S = "mosaic/tesseracts/structural-mesh/dealii-structural"
H = "mosaic/tesseracts/thermal-mesh/dealii-heat"
SRC = {S: f"{S}/src/struct_solver.cpp", H: f"{H}/src/heat_solver.cpp"}
V = {
    "ds-uncap": (S, [("  MultithreadInfo::set_thread_limit(1);\n", "")]),
    "ds-cg": (
        S,
        [
            (
                "#include <deal.II/lac/sparse_direct.h>\n",
                "#include <deal.II/lac/sparse_direct.h>\n#include <deal.II/lac/solver_cg.h>\n#include <deal.II/lac/precondition.h>\n",
            ),
            (
                """  SparseDirectUMFPACK direct;
  direct.initialize(system_matrix);
  direct.vmult(solution, system_rhs);""",
                """  SolverControl solver_control(50000, 1e-12 * system_rhs.l2_norm());
  SolverCG<Vector<double>> cg(solver_control);
  PreconditionSSOR<SparseMatrix<double>> precond;
  precond.initialize(system_matrix, 1.2);
  cg.solve(system_matrix, solution, system_rhs, precond);""",
            ),
        ],
    ),
    "dh-cap": (
        H,
        [
            (
                "#include <deal.II/base/quadrature_lib.h>\n",
                "#include <deal.II/base/multithread_info.h>\n#include <deal.II/base/quadrature_lib.h>\n",
            ),
            (
                "int main(int argc, char* argv[]) {\n",
                "int main(int argc, char* argv[]) {\n    MultithreadInfo::set_thread_limit(1);\n",
            ),
        ],
    ),
    "dh-tol8": (
        H,
        [
            (
                "SolverControl solver_control(50000, 1e-12 * system_rhs.l2_norm());",
                "SolverControl solver_control(50000, 1e-8 * system_rhs.l2_norm());",
            )
        ],
    ),
}
env = {**os.environ, "TMPDIR": "/nvme/scratch/tmp"}
for v in sys.argv[1:]:
    t, reps = V[v]
    subprocess.run(["git", "-C", str(WT), "checkout", "-q", "--", "."], check=True)
    p = WT / SRC[t]
    s = p.read_text()
    for a, b in reps:
        assert s.count(a) == 1, (v, a)
        s = s.replace(a, b)
    p.write_text(s)
    r = subprocess.run(
        [str(MAIN / ".venv/bin/tesseract"), "build", "-t", v, t],
        cwd=WT,
        env=env,
        capture_output=True,
        text=True,
    )
    print(
        v,
        r.returncode,
        r.stdout.strip().splitlines()[-1:] or r.stderr[-1500:],
        flush=True,
    )
subprocess.run(["git", "-C", str(WT), "checkout", "-q", "--", "."], check=True)
