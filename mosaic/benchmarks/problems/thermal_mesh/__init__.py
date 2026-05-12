"""Quasi-2D steady heat-conduction compliance / source-identification benchmark.

The problem definition is split across five modules:

- :mod:`.ics`         — IC generators (``_zero_source``, ``_uniform``,
                        ``_random``, ``_gaussian_source``, ``_two_gaussians``)
                        and the ``MAKE_IC`` registry.
- :mod:`.physics`     — mesh / BC builders, the reference FEM solve for
                        inverse-recovery ground truth, the
                        ``build_make_inputs`` factory, the
                        ``_density_to_2d`` helper, and ``DIAGNOSTICS``.
- :mod:`.experiments` — per-suite ``_*_DEFAULTS`` dicts and the assembled
                        ``EXPERIMENTS`` registry.
- :mod:`.plots`       — assembled ``PLOT_FNS`` registry.
- :mod:`.config`      — solver discovery and the final
                        :class:`~mosaic.benchmarks.core.config.Problem`
                        instance assembled from the modules above.
"""

from .config import CONFIG

__all__ = ["CONFIG"]
