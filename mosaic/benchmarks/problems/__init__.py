# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry of available problem configs, keyed by CLI name.

Problem modules are auto-discovered. Each problem lives either as a top-level
``<name>.py`` file (with ``problem = Problem(...)`` at module scope) or as a
subpackage ``<name>/`` whose ``config.py`` defines ``problem``. The discovery
scan picks up both shapes — anything defining a module-level ``problem`` of
type :class:`Problem` is registered automatically.
"""

from __future__ import annotations

import functools
import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from mosaic.benchmarks.core.config import Problem

if TYPE_CHECKING:
    PROBLEMS: list[str]

log = logging.getLogger(__name__)

_PROBLEMS_DIR = Path(__file__).parent


def _candidate_module_paths() -> list[str]:
    """Yield fully-qualified module paths to scan for ``problem``.

    Two shapes are supported:
      * Single-file:  ``problems/<name>.py``         → ``…problems.<name>``
      * Package:      ``problems/<name>/config.py``  → ``…problems.<name>.config``

    The package shape uses ``<name>/config.py`` as the canonical entry rather
    than ``<name>/__init__.py``, so per-package ``__init__.py`` files can stay
    empty (or hold only a docstring) without breaking discovery.
    """
    paths: list[str] = []
    for entry in sorted(_PROBLEMS_DIR.iterdir()):
        if entry.name.startswith("_") or entry.name == "__pycache__":
            continue
        if entry.is_file() and entry.suffix == ".py":
            paths.append(f"mosaic.benchmarks.problems.{entry.stem}")
        elif entry.is_dir():
            # Canonical: ``<pkg>/config.py``. Fall back to ``<pkg>/__init__.py``
            # for packages that haven't been split into config.py yet.
            if (entry / "config.py").exists():
                paths.append(f"mosaic.benchmarks.problems.{entry.name}.config")
            elif (entry / "__init__.py").exists():
                paths.append(f"mosaic.benchmarks.problems.{entry.name}")
    return paths


@functools.cache
def _registry() -> dict[str, Problem]:
    registry: dict[str, Problem] = {}
    for module_name in _candidate_module_paths():
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
    """Return the Problem config registered under *name*."""
    reg = _registry()
    if name not in reg:
        raise ValueError(f"Unknown problem {name!r}. Choose from: {list(reg)}")
    return reg[name]


def problem_names() -> list[str]:
    """Return the CLI names of all registered problems."""
    return list(_registry())


def __getattr__(name: str) -> list[str]:
    # ``PROBLEMS`` is built on first access rather than at import time, because
    # building the registry imports every problem config and with it JAX and
    # the plotting stack, which made every CLI invocation slow (#58).
    if name == "PROBLEMS":
        return problem_names()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
