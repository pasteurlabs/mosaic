"""Registry of available problem configs, keyed by CLI name.

Problem modules are auto-discovered: any ``*.py`` file in this directory
(excluding ``__init__.py`` and private ``_``-prefixed files) that defines a
module-level ``CONFIG`` attribute of type :class:`ProblemConfig` is registered
automatically.  No manual import or registry entry is needed.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from mosaic.benchmarks.core.config import ProblemConfig

log = logging.getLogger(__name__)

_PROBLEMS_DIR = Path(__file__).parent


def _registry() -> dict[str, ProblemConfig]:
    registry: dict[str, ProblemConfig] = {}
    for path in sorted(_PROBLEMS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = f"mosaic.benchmarks.problems.{path.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            log.warning("skipping %s: %s", module_name, exc)
            continue
        cfg = getattr(mod, "CONFIG", None)
        if isinstance(cfg, ProblemConfig):
            registry[cfg.name] = cfg
    return registry


def get_config(name: str) -> ProblemConfig:
    reg = _registry()
    if name not in reg:
        raise ValueError(f"Unknown problem {name!r}. Choose from: {list(reg)}")
    return reg[name]


PROBLEMS: list[str] = list(_registry().keys())
