## Summary

Reshape the codebase around three intents:

- **Make a problem readable as a single declaration.** Before, each problem was a scatter of module-level dicts (`MAKE_IC`, `EXPERIMENTS`, `PLOT_DESCRIPTIONS`, `EXCLUSIONS`, …) that had to stay in sync by convention. Now one `Problem()` instance is built declaratively via `add_experiment(...)` / `add_ic(...)` / `exclude(...)` calls — each experiment's config, plot, status checks, and exclusions live next to each other and nowhere else.

- **Push the per-suite loops into the framework, not the problems.** The forward/gradient/cost runners used to be near-duplicated ~500-line loops in each problem package. They're now a single generic driver plus a `@kernel` decorator; a problem only supplies the one-step kernel and an aggregate. Adding a problem stops meaning "copy the harness."

- **Draw a clean line between framework code and problem code.** Shared infrastructure lives under `benchmarks/core/`; anything that's specific to one physical problem (optimisers, ICs, input factories, schemas) lives under that problem's package or under `tesseracts/`. Editing one problem no longer risks touching framework primitives.

Supporting work that falls out of the above: templates and the `mosaic new-domain` scaffold produce the new shape directly, and the test suite was rebuilt around the new abstractions (kernel/harness unit tests + an end-to-end dummy-tesseract integration test) so the API can't silently drift.

## Test plan

- [ ] `pytest mosaic/tests/ -q`
- [ ] `mosaic new-domain my-flow --from-template ns-periodic` then `mosaic status my-flow`
- [ ] Spot-check a real run on one in-tree problem (`mosaic run ns-grid forward -e baseline`)
