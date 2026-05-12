"""2D incompressible Navier-Stokes on a periodic grid.

The problem definition is split across five modules:

- :mod:`.ics`         — IC generators (``_multimode``, ``_tgv``, ``_uniform_flow``,
                        ``_flat_inflow``), the ``MAKE_IC`` registry, and the
                        ``_tgv_analytic`` reference solution.
- :mod:`.physics`     — input factory (``build_make_inputs``) and diagnostic
                        functions (``_divergence_rms``, ``_kinetic_energy``,
                        ``_energy_spectrum``).
- :mod:`.experiments` — per-suite ``_*_DEFAULTS`` dicts and the assembled
                        ``EXPERIMENTS`` registry.
- :mod:`.plots`       — assembled ``PLOT_FNS`` registry incl. the
                        ``_extra/gradient/jacobian_svd_comparison`` callback.
- :mod:`.config`      — solver discovery, exclusions, and the final
                        :class:`~mosaic.benchmarks.core.config.Problem`
                        instance assembled from the modules above.
"""

from .config import CONFIG

__all__ = ["CONFIG"]
