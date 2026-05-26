<!--
Please use a PR title that conforms to *conventional commits*: "<commit_type>: Describe your change"; for example: "fix: prevent race condition". Some other commit types are: fix, feat, ci, doc, refactor...
For a full list of commit types visit https://www.conventionalcommits.org/en/v1.0.0/
-->

## Summary

<!-- What does this PR do? 1-3 sentences. -->

## Type of change

- [ ] New solver backend
- [ ] Solver tuning / improvement
- [ ] New benchmark domain
- [ ] Harness / infrastructure change
- [ ] Documentation
- [ ] Bug fix

## Checklist

- [ ] `ruff check --fix && ruff format` passes
- [ ] `pytest` passes (unit tests, no Docker required)

### For new solvers

- [ ] `tesseract_config.yaml` has a `metadata.mosaic:` block with at least `name` and `backend`
- [ ] `tesseract build mosaic/tesseracts/<domain>/<solver>` succeeds
- [ ] `mosaic run -p <domain> --suites forward -s <solver> --debug` completes
- [ ] `mosaic status -p <domain> -f` output pasted below
- [ ] Exclusions and explained anomalies documented (if any)

### For solver tuning

- [ ] Before/after `mosaic status --format json` snapshots compared
- [ ] No regressions to other solvers

## Status output

<!-- Paste the output of `mosaic status -p <domain> -f` here -->

```

```

## Notes

<!-- Anything reviewers should know: exclusions, anomalies, GPU requirements, etc. -->
