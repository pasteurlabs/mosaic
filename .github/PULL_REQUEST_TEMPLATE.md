<!--
Please use a PR title that conforms to *conventional commits*: "<commit_type>: Describe your change"; for example: "fix: prevent race condition". Some other commit types are: fix, feat, ci, doc, refactor...
For a full list of commit types visit https://www.conventionalcommits.org/en/v1.0.0/
-->

## Summary

<!-- What does this PR do? 1-3 sentences. -->

## Type of change

- [ ] New solver
- [ ] Solver tuning
- [ ] New domain
- [ ] Harness or infrastructure
- [ ] Documentation
- [ ] Bug fix

### For solver PRs

- [ ] `metadata.mosaic` in `tesseract_config.yaml` sets at least `name`, `backend` and `scheme`
- [ ] `tesseract build mosaic/tesseracts/<domain>/<solver>` succeeds
- [ ] `mosaic run -p <problem> --suites forward -s "<Display Name>" --debug` completes
- [ ] The intervention record (`abstraction`, `interventions`, `deviations`) is filled in, with markers around L2 and L3 code

### For domain PRs

- [ ] `mosaic validate-domain <problem>` passes

<!--
A maintainer applies a benchmark label, and CI posts a comment comparing the results against `main`.
See https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/contributing.html#benchmark-labels
-->

## Notes for reviewers

<!-- Anything reviewers should know: exclusions, anomalies, GPU requirements, etc. -->
