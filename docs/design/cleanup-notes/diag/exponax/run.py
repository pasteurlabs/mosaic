"""Effect of exponax's dealiasing_fraction on the ns-3d-grid TGV runs.

Replicates the 3D path of mosaic's exponax tesseract (NavierStokesVelocity,
ETDRK2, drag 0, lax.scan) and compares dealiasing_fraction = 1.0 (current
mosaic setting) with 2/3 (exponax default) against a float64 N=128 run with a
4x smaller time step.
"""

import json
import sys

import jax

MODE = sys.argv[
    3
]  # "truth" (float64, saves npz) or "main" (float32, as in the tesseract)
if MODE == "truth":
    jax.config.update("jax_enable_x64", True)
import exponax as ex
import jax.numpy as jnp
import numpy as np

L = 2 * np.pi
REF_DIR = sys.argv[1]  # mosaic references dir
FIELDS_DIR = sys.argv[2]  # run results dir (forward/)


def tgv(N, dtype):
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    v = np.stack(
        [np.sin(X) * np.cos(Y) * np.cos(Z), -np.cos(X) * np.sin(Y) * np.cos(Z), 0 * X],
        axis=-1,
    ).astype(np.float32)  # mosaic IC is float32
    return jnp.asarray(v, dtype=dtype)


def run(N, nu, dt, steps, frac, dtype=jnp.float32):
    st = ex.stepper.NavierStokesVelocity(
        3, L, N, dt, diffusivity=nu, drag=0.0, order=2, dealiasing_fraction=frac
    )
    u0 = jnp.moveaxis(tgv(N, dtype), -1, 0)

    def f(c, _):
        return st(c).astype(c.dtype), None

    out, _ = jax.jit(lambda u: jax.lax.scan(f, u, None, length=steps))(u0)
    return np.asarray(jnp.moveaxis(out, 0, -1), dtype=np.float64)


def down(field, n):
    n0 = field.shape[0]
    fh = np.fft.fftn(field, axes=(0, 1, 2))
    h = n // 2
    idx = np.concatenate([np.arange(h), np.arange(n0 - h, n0)])
    fh = fh[np.ix_(idx, idx, idx, range(3))]
    return np.real(np.fft.ifftn(fh, axes=(0, 1, 2)) * (n / n0) ** 3)


def err(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b**2)))


def shell_energy_above(field, kc):
    """Fraction of kinetic energy in modes with max|k_i| > kc."""
    n = field.shape[0]
    fh = np.fft.fftn(field, axes=(0, 1, 2))
    k = np.abs(np.fft.fftfreq(n) * n)
    K = np.maximum.reduce(np.meshgrid(k, k, k, indexing="ij"))
    e = np.sum(np.abs(fh) ** 2, axis=-1)
    return float(e[K > kc].sum() / e.sum())


if MODE == "truth":
    tr = {
        f"ag_{i}": down(run(128, nu, 0.0025, 200, 2 / 3, jnp.float64), 16)
        for i, nu in enumerate([0.001, 0.01, 0.05])
    }
    tr["bl_hi"] = run(128, 0.05, 0.0025, 512, 2 / 3, jnp.float64)
    np.savez("/nvme/scratch/dealias_exp/truth.npz", **tr)
    sys.exit(0)
TR = np.load("/nvme/scratch/dealias_exp/truth.npz")

FR = {"f1": 1.0, "f23": 2 / 3}
out = {"agreement": {}, "baseline": {}}

# forward/agreement: N=16 grid, dt=0.01, 50 steps (T=0.5); reference at N=64.
ref_ag = np.load(f"{REF_DIR}/forward_agreement.npz")
fl_ag = np.load(f"{FIELDS_DIR}/agreement/fields.npz")
solvers = [str(s) for s in fl_ag["solver_names"]]
for i, nu in enumerate([0.001, 0.01, 0.05]):
    truth = TR[f"ag_{i}"]
    r = {}
    hi = {k: run(64, nu, 0.01, 50, f) for k, f in FR.items()}
    ref = {k: down(v, 16) for k, v in hi.items()}
    r["E_frac_above_k20_at_N64_f1"] = shell_energy_above(hi["f1"], 20)
    r["E_frac_above_k8_at_N64_f1"] = shell_energy_above(hi["f1"], 7)
    r["ref_f1_vs_checked_in"] = err(ref["f1"], ref_ag[f"reference_{i}"])
    r["ref_f1_vs_truth"] = err(ref["f1"], truth)
    r["ref_f23_vs_truth"] = err(ref["f23"], truth)
    r["ref_f1_vs_ref_f23"] = err(ref["f1"], ref["f23"])
    for k, f in FR.items():
        e16 = run(16, nu, 0.01, 50, f)
        r[f"exponax16_{k}_vs_ref_f1"] = err(e16, ref["f1"])
        r[f"exponax16_{k}_vs_ref_f23"] = err(e16, ref["f23"])
        r[f"exponax16_{k}_vs_truth"] = err(e16, truth)
    r["exponax16_stored_vs_my_f1"] = err(
        fl_ag[f"Exponax_{i}"], run(16, nu, 0.01, 50, 1.0)
    )
    r["peers"] = {
        s: {
            "vs_ref_f1": err(fl_ag[f"{s}_{i}"], ref["f1"]),
            "vs_ref_f23": err(fl_ag[f"{s}_{i}"], ref["f23"]),
            "vs_truth": err(fl_ag[f"{s}_{i}"], truth),
        }
        for s in solvers
    }
    out["agreement"][str(nu)] = r
    print(json.dumps({nu: r}, indent=1), flush=True)

# forward/baseline: nu=0.05, T=1.28, fixed CFL (dt=0.04 at N=32).
ref_bl = np.load(f"{REF_DIR}/forward_baseline.npz")
fl_bl = np.load(f"{FIELDS_DIR}/baseline/fields.npz")
nu = 0.05
hi = {k: run(64, nu, 0.02, 64, f) for k, f in FR.items()}
truth_hi = TR["bl_hi"]
out["baseline"]["E_frac_above_k20_at_N64_f1"] = shell_energy_above(hi["f1"], 20)
for i, n in enumerate([8, 16, 32]):
    truth = down(truth_hi, n)
    ref = {k: down(v, n) for k, v in hi.items()}
    r = {
        "ref_f1_vs_checked_in": err(ref["f1"], ref_bl[f"reference_{i}"]),
        "ref_f1_vs_truth": err(ref["f1"], truth),
        "ref_f23_vs_truth": err(ref["f23"], truth),
        "ref_f1_vs_ref_f23": err(ref["f1"], ref["f23"]),
    }
    dt, steps = 0.04 * 32 / n, 32 * n // 32
    for k, f in FR.items():
        en = run(n, nu, dt, steps, f)
        r[f"exponax{n}_{k}_vs_ref_f1"] = err(en, ref["f1"])
        r[f"exponax{n}_{k}_vs_ref_f23"] = err(en, ref["f23"])
        r[f"exponax{n}_{k}_vs_truth"] = err(en, truth)
    r["exponax_stored_vs_my_f1"] = err(
        fl_bl[f"Exponax_{i}"], run(n, nu, dt, steps, 1.0)
    )
    r["peers"] = {
        s: {
            "vs_ref_f1": err(fl_bl[f"{s}_{i}"], ref["f1"]),
            "vs_ref_f23": err(fl_bl[f"{s}_{i}"], ref["f23"]),
            "vs_truth": err(fl_bl[f"{s}_{i}"], truth),
        }
        for s in solvers
    }
    out["baseline"][str(n)] = r
    print(json.dumps({n: r}, indent=1), flush=True)

json.dump(out, open("/nvme/scratch/dealias_exp/results.json", "w"), indent=1)
