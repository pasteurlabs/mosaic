# PhiFlow metrics of the 1e-7 arm against another tolerance arm (default tol1e-5), for the
# experiments tol-fast.sh runs. Usage: tol-compare.py [arm]
import glob
import json
import os
import sys

A, B = (
    "/nvme/scratch/ab/tol1e-7",
    f"/nvme/scratch/ab/{sys.argv[1] if len(sys.argv) > 1 else 'tol1e-5'}",
)
SKIP = ("time", "_s", "ram", "vram", "trials")


def phiflow(path):
    with open(path) as fh:
        d = json.load(fh)
    return {
        r.get("sweep_value"): r["metrics"]
        for r in d["results"]
        if r["solver"] == "PhiFlow"
    }


def scalars(m):
    return {
        k: v
        for k, v in m.items()
        if isinstance(v, (int, float)) and not any(s in k for s in SKIP)
    }


# Experiments tol-fast.sh runs in both arms (the 1e-7 arm ran the ns-grid ones in its full run).
RUN = (
    "ns-3d-grid/optimization/recovery_constant_ic_bfgs_proj",
    "ns-3d-grid/forward",
    "ns-3d-grid/gradient/fd_check",
    "ns-grid/forward/baseline",
    "ns-grid/forward/agreement",
    "ns-grid/forward/physical_laws",
    "ns-grid/forward/tgv_nu_sweep",
    "ns-grid/gradient/fd_check",
    "ns-grid/gradient/analytic_check",
)

for fb in sorted(glob.glob(f"{B}/**/result.json", recursive=True)):
    rel = os.path.relpath(fb, B)
    if not rel.startswith(RUN) or not os.path.exists(f"{A}/{rel}"):
        continue
    ra, rb = phiflow(f"{A}/{rel}"), phiflow(fb)
    for sv in sorted(set(ra) & set(rb), key=str):
        ma, mb = scalars(ra[sv]), scalars(rb[sv])
        for k in sorted(set(ma) & set(mb)):
            x, y = ma[k], mb[k]
            rd = abs(y - x) / max(abs(x), 1e-300)
            flag = "" if rd < 1e-3 else "  <--"
            print(
                f"{os.path.dirname(rel):55s} {sv!s:>8} {k:28s} {x:11.4g} {y:11.4g} {rd:8.1e}{flag}"
            )
