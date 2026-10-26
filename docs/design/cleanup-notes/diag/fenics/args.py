import pickle, sys
sys.path.insert(0, "/tesseract")
import tesseract_api as m
import fenics_adjoint.types.compat as c
inner = c.linalg_solve
def show(*a, **k):
    print("adjoint linalg_solve via", inner.__name__, "extra args:", [x for x in a[3:]], "kwargs:", k)
    return inner(*a, **k)
c.linalg_solve = show
inp = m.InputSchema.model_validate(pickle.load(open("/data/inputs.pkl", "rb")))
m.vector_jacobian_product(inp, {"rho"}, {"compliance"}, {"compliance": 1.0})
