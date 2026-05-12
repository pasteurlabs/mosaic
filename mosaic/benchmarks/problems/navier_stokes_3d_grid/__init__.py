"""3D incompressible Navier-Stokes on a triply-periodic grid.

The problem definition is split across four modules:

- :mod:`.ics`         — IC generators (``_tgv3d``, ``_abc_flow``,
                        ``_rand_div_free_3d``), the ``MAKE_IC`` registry, and
                        the ``_tgv3d_analytic`` reference solution.
- :mod:`.physics`     — input factory (``build_make_inputs``) and diagnostic
                        functions (``_divergence_rms``, ``_kinetic_energy``,
                        ``_energy_spectrum``).
- :mod:`.experiments` — per-suite run-lists plus the assembled
                        ``EXPERIMENTS`` / ``PLOT_FNS`` registries built via
                        the :class:`Problem` builder pattern.
- :mod:`.config`      — solver discovery, exclusions, the ``_field_to_2d``
                        slice helper, and the final
                        :class:`~mosaic.benchmarks.core.config.Problem`
                        instance assembled from the modules above.
"""

from .config import CONFIG

__all__ = ["CONFIG"]
