"""Per-sweep-point mean times for cost experiments, side by side for several result dirs.

Usage: cost_summary.py <problem> <experiment-subpath> <dir>...
"""
import json, sys
prob, exp, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = {}
for d in dirs:
    try: r = json.load(open(f"{d}/{prob}/{exp}/result.json"))
    except Exception as e: print(d, "missing"); continue
    for x in r["results"]:
        m = x["metrics"]; t = m.get("mean")
        rows.setdefault(x["sweep_value"], {})[d.split("/")[-1]] = t
print(exp, "| " + " | ".join(d.split("/")[-1] for d in dirs))
for k in sorted(rows, key=lambda v: (v is None, v)):
    print(f"  {k}: " + " | ".join(f"{rows[k].get(d.split('/')[-1]):.4g}" if rows[k].get(d.split('/')[-1]) is not None else "-" for d in dirs))
