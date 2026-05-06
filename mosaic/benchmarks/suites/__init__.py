"""Suite registry — maps suite names to their module paths.

Each suite module must expose either:
  - ``_EXPERIMENTS`` (dict) and ``_plot_fns`` (callable returning dict)
  - ``get_experiments(cfg)`` and ``get_plot_fns(cfg)`` for dynamic suites

Adding a new suite: add one entry here and create the module.
"""

SUITE_REGISTRY: dict[str, str] = {
    "ics": "mosaic.benchmarks.suites.ics",
    "forward": "mosaic.benchmarks.suites.forward",
    "cost": "mosaic.benchmarks.suites.cost",
    "gradient": "mosaic.benchmarks.suites.gradient",
    "optimization": "mosaic.benchmarks.suites.optimization",
}
