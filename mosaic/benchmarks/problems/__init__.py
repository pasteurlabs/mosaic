"""Registry of available problem configs, keyed by CLI name.

Problem modules are auto-discovered. Each problem lives either as a top-level
``<name>.py`` file or as a subpackage ``<name>/`` containing an
``__init__.py`` that re-exports ``problem``. The discovery scan picks up both
shapes — anything defining a module-level ``problem`` of type
:class:`Problem` is registered automatically.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from mosaic.benchmarks.core.config import Problem

log = logging.getLogger(__name__)

_PROBLEMS_DIR = Path(__file__).parent


def _candidate_module_names() -> list[str]:
    """Yield problem module names (single-file *or* package) in this dir."""
    names: list[str] = []
    for entry in sorted(_PROBLEMS_DIR.iterdir()):
        if entry.name.startswith("_") or entry.name == "__pycache__":
            continue
        if entry.is_file() and entry.suffix == ".py":
            names.append(entry.stem)
        elif entry.is_dir() and (entry / "__init__.py").exists():
            names.append(entry.name)
    return names


def _registry() -> dict[str, Problem]:
    registry: dict[str, Problem] = {}
    for stem in _candidate_module_names():
        module_name = f"mosaic.benchmarks.problems.{stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            log.warning("skipping %s: %s", module_name, exc)
            continue
        cfg = getattr(mod, "problem", None)
        if isinstance(cfg, Problem):
            registry[cfg.name] = cfg
    return registry


def get_config(name: str) -> Problem:
    reg = _registry()
    if name not in reg:
        raise ValueError(f"Unknown problem {name!r}. Choose from: {list(reg)}")
    return reg[name]


PROBLEMS: list[str] = list(_registry().keys())
