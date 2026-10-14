# Canonical container CI validation

[Benchmark run 38561389132](https://github.com/pasteurlabs/mosaic/actions/runs/38561389132) passed, including documentation rendering. PR head: `855420fd199c240738391c2ce9b55e4cd351dfb4`; tested merge: `9d9f5c26370283e94e6f217d8874c83629f08273`.

The formal buoyancy report contains 20 successful cells: five solvers across forward, gradient, cost, and optimization. No buoyancy anomalies, failures, missing or stale results. Other domains in the report are baseline overlays, not reruns.

All five correctors completed 256 updates (64 per training case), passed reference, recurrence and neural gradient checks, and reproduced the prior selected checkpoint and rounded held-out losses. `summary.json` records actual times and comparisons. Full optimization benchmark time per solver: INS.jl 609 s, Oceananigans 441 s, WaterLily 317 s, JAX-CFD 227 s, PhiFlow 1512 s. These exclude image build/download and environment setup. CPU optimization command took about 23 minutes; GPU about 29 minutes. Julia backends ran on CPU, JAX-CFD and PhiFlow on GPU.

The shared finite-difference suite retains its existing median-direction criterion; JAX-CFD has a near-zero directional component with a large relative discrepancy. Passing the aggregate check does not certify every direction. Neither acceptance thresholds nor training recipe were changed for this validation.

`status-report.md` is the official combined report. `detailed_metric_review.json` retains per-suite evidence; provenance, workflow metadata and artifact checksums accompany it. One model seed and held-out trajectory establish bounded benchmark feasibility, not general superiority over supervised training.
