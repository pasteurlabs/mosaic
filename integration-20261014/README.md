# Main integration checks for buoyancy

Source `74069820dfd6d3f01ee98991630e7fa4da23067e`: 128 focused tests passed on Slurm (job 3052902). The only subsequent source change was a three-line chart/test adjustment to use a readable linear error axis, source `855420fd199c240738391c2ce9b55e4cd351dfb4`: all four chart tests passed (job 3053610).

All five original saved training results were rendered through the merged Altair code, producing 20 PNGs and 20 Vega-Lite sidecars. Final plots are under `plots/<solver>/`; receipt hashes identify every file. No numerical training was repeated. The underlying results are the four-case Slurm campaign at source `a2339b9`, with artifacts at commit `d334f4d41630df4b965a48267c505a72ff20bf39`, folder `data_mix_256`.

These supplemental Slurm tests used the pip-resolved environments recorded here. Canonical Docker execution and the frozen production environment are validated separately by GitHub Actions; these receipts do not certify that path.
