"""3D incompressible Navier-Stokes on a triply-periodic grid.

The problem definition is split across five modules:

- :mod:`.ics`         — IC generators (``_tgv3d``, ``_abc_flow``,
                        ``_rand_div_free_3d``), the ``MAKE_IC`` registry, and
                        the ``_tgv3d_analytic`` reference solution.
- :mod:`.physics`     — input factory (``build_make_inputs``) and diagnostic
                        functions (``_divergence_rms``, ``_kinetic_energy``,
                        ``_energy_spectrum``).
- :mod:`.experiments` — per-suite ``_*_DEFAULTS`` dicts and the assembled
                        ``EXPERIMENTS`` registry.
- :mod:`.plots`       — assembled ``PLOT_FNS`` registry incl. the
                        ``_extra/gradient/jacobian_svd_comparison`` callback
                        and the ``_field_to_2d`` slice helper.
- :mod:`.config`      — solver discovery, exclusions, and the final
                        :class:`~mosaic.benchmarks.core.config.Problem`
                        instance assembled from the modules above.
"""

from .config import CONFIG

__all__ = ["CONFIG"]
