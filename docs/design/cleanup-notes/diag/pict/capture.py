import pickle
import sys

sys.path.insert(0, "/tesseract")
import numpy as np
import PISOtorch_simulation as ps
import tesseract_api as m

orig_ls = ps.Simulation.linear_solve


def ls(self, csrMat, rhs, x=None, use_BiCG=False, tol=None, max_iter=None, *a, **kw):
    snap = {
        "value": csrMat.value.detach().cpu().numpy(),
        "index": csrMat.index.cpu().numpy(),
        "row": csrMat.row.cpu().numpy(),
        "rhs": rhs.detach().cpu().numpy(),
        "x0": None if x is None else x.detach().cpu().numpy(),
        "tol": np.float64(tol if tol is not None else np.nan),
    }
    if not use_BiCG:
        dom = self.domain
        snap["block_names"] = np.array(
            [dom.getBlock(i).name for i in range(dom.getNumBlocks())]
        )
        snap["block_shapes"] = np.array(
            [
                tuple(dom.getBlock(i).pressure.shape[2:])
                for i in range(dom.getNumBlocks())
            ]
        )
        np.savez_compressed(
            f"/out/pressure_system_{sys.argv[1]}.npz",
            **{k: v for k, v in snap.items() if v is not None},
        )
        print(
            "saved pressure system: rows",
            len(snap["row"]) - 1,
            "nnz",
            len(snap["value"]),
            "dtype",
            snap["value"].dtype,
            "x0 given",
            snap["x0"] is not None,
            "tol",
            tol,
        )
    return orig_ls(self, csrMat, rhs, x, use_BiCG, tol, max_iter, *a, **kw)


ps.Simulation.linear_solve = ls
orig = m._simulate


def patched(*a, **kw):
    kw["warm_start_tol"] = None
    kw["non_orthogonal"] = sys.argv[1] == "true"
    return orig(*a, **kw)


m._simulate = patched
with open("/data/inputs.pkl", "rb") as f:
    inp = m.InputSchema.model_validate(pickle.load(f))
try:
    m.apply(inp)
except RuntimeError as e:
    print("failed as expected:", str(e).splitlines()[0][:100])
