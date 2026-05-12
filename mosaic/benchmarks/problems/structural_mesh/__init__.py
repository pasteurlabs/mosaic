"""3D linear-elasticity SIMP topology optimisation on a cantilever beam.

The problem definition is split across five modules:

- :mod:`.ics`         — IC generators (``_uniform``, ``_random``,
                        ``_two_density_bumps``) and the ``MAKE_IC`` registry.
- :mod:`.physics`     — mesh and BC builders, input factory
                        (``build_make_inputs``), the ``_infer_mesh_dims``
                        helper, and the ``DIAGNOSTICS`` registry.
- :mod:`.experiments` — per-suite ``_*_DEFAULTS`` dicts and the assembled
                        ``EXPERIMENTS`` registry.
- :mod:`.plots`       — the ``_density_to_2d`` field projection and the
                        assembled ``PLOT_FNS`` registry.
- :mod:`.config`      — solver discovery and the final
                        :class:`~mosaic.benchmarks.core.config.Problem`
                        instance assembled from the modules above.
"""

from .config import CONFIG

__all__ = ["CONFIG"]
