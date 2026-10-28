"""Solve a captured PICT pressure system with PICT's GPU solvers and with SciPy.

Usage: python standalone.py pressure_system.npz
"""

import sys

import numpy as np
import PISOtorch
import PISOtorch_diff
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

z = np.load(sys.argv[1])
n = len(z["row"]) - 1
A = sp.csr_matrix((z["value"].astype(np.float64), z["index"], z["row"]), shape=(n, n))
b = z["rhs"].reshape(-1).astype(np.float64)
asym = abs(A - A.T).max() / abs(A).max()
diag = A.diagonal()
print(
    f"n={n} nnz={A.nnz} max|A-A^T|/max|A|={asym:.2e} diag range [{diag.min():.3g}, {diag.max():.3g}]"
)
print(
    f"row sums: max|.|={abs(np.asarray(A.sum(1))).max():.2e}; sum(b)={b.sum():.3e}, |b|={np.linalg.norm(b):.3e}"
)
x = spla.spsolve(A.tocsc(), b)
print(
    f"scipy spsolve: finite={np.isfinite(x).all()} rel residual={np.linalg.norm(A @ x - b) / np.linalg.norm(b):.2e}"
)
try:
    ev = spla.eigs(A, k=6, sigma=0, return_eigenvectors=False)
    print("eigenvalues nearest 0:", np.sort_complex(ev))
except Exception as e:
    print("eigs failed:", e)

dev = torch.device("cuda")
for dtype in (torch.float32, torch.float64):
    for bicg in (False, True):
        mat = PISOtorch.CSRmatrix(
            torch.tensor(z["value"], dtype=dtype, device=dev),
            torch.tensor(z["index"], device=dev),
            torch.tensor(z["row"], device=dev),
        )
        rhs = torch.tensor(z["rhs"], dtype=dtype, device=dev)
        xg = torch.zeros_like(rhs)
        try:
            info = PISOtorch_diff._linear_solve_wrapper(
                mat,
                rhs,
                xg,
                torch.IntTensor([5000]),
                PISOtorch_diff._get_solver_tolerance_torch(1e-6, dtype),
                PISOtorch.ConvergenceCriterion.NORM2_NORMALIZED,
                bicg,
                False,
                0,
                False,
                False,
                False,
                double_fallback=False,
                debug_out=False,
                BiCG_with_preconditioner=True,
                BiCG_precondition_fallback=True,
            )
        except PISOtorch_diff.LinsolveError as e:
            info = str(e).splitlines()[-1].strip()
        print(
            f"PICT GPU {'BiCGStab' if bicg else 'CG'} {str(dtype)[6:]}:"
            f" finite={bool(torch.isfinite(xg).all())} info={info}"
        )
