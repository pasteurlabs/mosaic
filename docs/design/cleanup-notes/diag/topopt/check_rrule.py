import pickle
import sys

import numpy as np

sys.path.insert(0, "/tesseract")
import tesseract_api as api

jl = api._get_jl()
jl.seval("using Zygote")
jl.seval("""
function _grad_pair(rho_in, pts_in, cells_in, dm_in, cm_in, cv_in, E, nu, xmin)
    _, _, comp = get_solver_mesh(Float64.(Matrix(pts_in)), Int64.(Matrix(cells_in)), Int32.(vec(dm_in)),
                                 Int32.(vec(cm_in)), Float64.(Matrix(cv_in)), E, nu, xmin)
    rho = Float64.(vec(rho_in))
    comp(PseudoDensities(rho))
    field = copy(comp.grad)
    zyg = Zygote.gradient(r -> comp(PseudoDensities(r)), rho)[1]
    return field, zyg
end
""")
for nx, raw in pickle.load(open("/data/inputs.pkl", "rb")).items():
    inp = api.InputSchema.model_validate(raw)
    pts, cells, rho, dm, vm, vv = api._unpack(inp)
    f, z = jl._grad_pair(
        *(api._to_julia(a) for a in (rho, pts, cells, dm, vm, vv)),
        float(inp.E),
        float(inp.nu),
        float(inp.xmin),
    )
    f, z = np.asarray(f), np.asarray(z)
    print(
        f"nx={nx}: n={len(f)} max|field - zygote| = {np.max(np.abs(f - z)):.3e}, bit-identical = {np.array_equal(f, z)}"
    )
