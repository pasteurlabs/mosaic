# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make PICT's CUDA error handlers throw instead of exiting the process.

PISOtorch checks every CUDA, cuBLAS and cuSPARSE call with a handler that
prints the error and calls ``exit(10)``. An out-of-memory error in its solvers
therefore kills the Tesseract server, and the client only sees the connection
drop. This rewrites each handler to throw ``std::runtime_error`` with the same
message, which reaches Python as a ``RuntimeError`` (and the harness as OOM
when the message says so).

Usage: ``python patch_cuda_errors.py /opt/pict/extensions``. Fails if the
source no longer contains exactly the handlers this was written against.
"""

import re
import sys
from pathlib import Path

# File -> number of handlers it defines, at PICT commit 72a6e86.
EXPECTED = {
    "cg_solver_kernel.cu": 3,
    "bicgstab_solver_kernel.cu": 3,
    "dispatch.h": 1,
    "PISO_multiblock_cuda_kernel.cu": 1,
}

# How each status type names its error. cuBLAS and cuSPARSE names such as
# CUBLAS_STATUS_ALLOC_FAILED identify out-of-memory errors.
ERROR_NAME = {
    "cudaError_t": "cudaGetErrorString(err)",
    "cublasStatus_t": "cublasGetStatusName(err)",
    "cusparseStatus_t": "cusparseGetErrorName(err)",
}

# mosaic-intervention: cuda-errors-raise
HANDLER = re.compile(
    r"(static void \w+\(const char\* file, unsigned line, const char\* statement, "
    r"(\w+) err\) \{\s*if \(err == \w+\) return;)\s*"
    r"std::cerr << .*? << std::endl;\s*exit\(10\);",
    re.DOTALL,
)


def _throwing_body(match: re.Match) -> str:
    head, status_type = match.group(1), match.group(2)
    name = ERROR_NAME[status_type]
    return (
        f"{head}\n"
        "  std::ostringstream msg;\n"
        f'  msg << statement << " returned " << {name} << "("\n'
        '      << err << ") at " << file << ":" << line;\n'
        "  std::cerr << msg.str() << std::endl;\n"
        "  throw std::runtime_error(msg.str());"
    )


def patch(extensions: Path) -> None:
    """Rewrite the handlers in PISOtorch's ``extensions`` directory in place."""
    for name, count in EXPECTED.items():
        path = extensions / name
        source = path.read_text()
        patched, n = HANDLER.subn(_throwing_body, source)
        if n != count or "exit(10)" in patched:
            raise SystemExit(f"{path}: expected {count} handlers, patched {n}")
        path.write_text("#include <sstream>\n#include <stdexcept>\n" + patched)
        print(f"{path}: {n} handlers now throw")


# mosaic-intervention-end: cuda-errors-raise

if __name__ == "__main__":
    patch(Path(sys.argv[1]))
