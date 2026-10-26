import pickle, sys
sys.path.insert(0, "/tesseract")
import tesseract_api as m
import fenics_adjoint.types.compat as c
calls = {"n": 0}
inner = c.linalg_solve
def counting(*a, **k):
    calls["n"] += 1
    return inner(*a, **k)
c.linalg_solve = counting
inp = m.InputSchema.model_validate(pickle.load(open("/data/inputs.pkl", "rb")))
m.apply(inp)
print("after apply:", calls["n"])
m.vector_jacobian_product(inp, {"rho"}, {"compliance"}, {"compliance": 1.0})
print("after vjp:", calls["n"], "| patched function is Mosaic's:", inner.__name__)
