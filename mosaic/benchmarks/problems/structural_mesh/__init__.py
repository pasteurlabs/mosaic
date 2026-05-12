"""3D linear-elasticity SIMP topology optimisation on a cantilever beam.

The problem definition is split across four modules:

- :mod:`.ics`         — IC generators (``_uniform``, ``_random``,
                        ``_two_density_bumps``) and the ``MAKE_IC`` registry.
- :mod:`.physics`     — mesh and BC builders, input factory
                        (``build_make_inputs``), the ``_infer_mesh_dims``
                        helper, the ``_density_to_2d`` field projection, and
                        the ``DIAGNOSTICS`` registry.
- :mod:`.experiments` — per-experiment run-lists and the assembled
                        ``EXPERIMENTS`` / ``PLOT_FNS`` registries.
- :mod:`.config`      — solver discovery and the final
                        :class:`~mosaic.benchmarks.core.config.Problem`
                        instance assembled from the modules above.
"""

from .config import CONFIG

__all__ = ["CONFIG"]
