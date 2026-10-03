### Final bounded neural-corrector comparison

**Inconclusive: incomplete results or failed admission; no superiority claim.**

Research stops here; failure is not evidence of equivalence.

- RuntimeError: terminal infrastructure failures require explicit review: {'2873094': 'FAILED'}
- matched_stopped/model13/[20008, 20009, 20010, 20011]: [Errno 2] No such file or directory: '/data/personal/andrinr/runner/results/mosaic/pr116-ins-final-v2-20261003/results/test-matched_stopped-s13-b2/outcome.json'
- matched_stopped: 4 missing model/IC pairs

| Selected recipe | Learning rate | Unroll | Updates | Supervised start |
|---|---:|---:|---:|---|
| Full solver gradients | 0.0001 | 16 | 3000 | None |
| Matched stopped gradients | 0.0001 | 16 | 3000 | None |
| Tuned stopped gradients | 0.0001 | 16 | 3000 | None |
| Tuned supervision | 1e-05 | 8 | 1000 | None |

Recorded tuning fit time: 60.18 GPU-hours; shared preparation, diagnostics and evaluation are additional.

[Frozen protocol, every candidate, failures, costs and numerical results](ARTIFACT_TREE). Earlier negative correction and neural-control results remain retained in this PR; direct control optimization is a separate non-neural result.
