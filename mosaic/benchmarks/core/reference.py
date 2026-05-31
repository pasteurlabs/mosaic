# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precomputed reference solutions for consensus-based experiments.

Mesh-domain forward experiments (structural-mesh, thermal-mesh) lack
an analytic reference — they derive a trimmed-mean consensus across
all available solvers at runtime. This creates a coupling: single-solver
runs can't produce valid results because there aren't enough peers to
form the consensus.

This module provides checked-in reference NPZ files that decouple
solvers. Each NPZ stores the trimmed-mean reference fields for one
experiment's sweep, keyed by sweep index. At experiment time,
:func:`load_reference` returns the precomputed reference; if missing,
the caller falls back to runtime consensus as before.

The ``mosaic reference`` CLI command generates these NPZ files by
running all solvers, computing the trimmed mean of their outputs, and
writing the result under ``problems/<domain>/references/``.

File layout::

    mosaic/benchmarks/problems/structural_mesh/references/
        forward_baseline.npz      # reference_{i} for each sweep value
        forward_agreement.npz
    mosaic/benchmarks/problems/thermal_mesh/references/
        forward_baseline.npz
        forward_agreement.npz
        forward_source_baseline.npz
        forward_source_linearity.npz
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Root of the problems package — references live alongside problem configs.
_PROBLEMS_DIR: Path = Path(__file__).resolve().parents[1] / "problems"


def _domain_slug_to_package(domain: str) -> str:
    """Convert a CLI domain slug (e.g. ``structural-mesh``) to the Python package name."""
    return domain.replace("-", "_")


def _reference_dir(domain: str) -> Path:
    """Return the ``references/`` directory for a domain."""
    return _PROBLEMS_DIR / _domain_slug_to_package(domain) / "references"


def _reference_filename(exp_key: str) -> str:
    """Convert an experiment key like ``forward/agreement`` to ``forward_agreement.npz``."""
    return exp_key.replace("/", "_") + ".npz"


def load_reference(domain: str, exp_key: str, sweep_index: int) -> np.ndarray | None:
    """Load a precomputed reference field for one sweep value.

    Returns ``None`` if the reference file doesn't exist or doesn't
    contain an entry for the requested sweep index.
    """
    path = _reference_dir(domain) / _reference_filename(exp_key)
    if not path.exists():
        return None
    try:
        with np.load(str(path), allow_pickle=False) as data:
            key = f"reference_{sweep_index}"
            if key not in data:
                return None
            return np.asarray(data[key])
    except Exception:
        log.warning("Failed to load reference from %s", path, exc_info=True)
        return None


def save_reference(
    domain: str,
    exp_key: str,
    references: dict[int, np.ndarray],
    sweep_values: list | None = None,
) -> Path:
    """Save precomputed reference fields for one experiment.

    Parameters
    ----------
    domain : str
        Domain slug (e.g. ``structural-mesh``).
    exp_key : str
        Experiment key (e.g. ``forward/agreement``).
    references : dict[int, np.ndarray]
        Mapping from sweep index to reference array.
    sweep_values : list, optional
        Sweep values to store for provenance.

    Returns:
    -------
    Path
        The written NPZ file path.
    """
    ref_dir = _reference_dir(domain)
    ref_dir.mkdir(parents=True, exist_ok=True)
    path = ref_dir / _reference_filename(exp_key)

    arrays: dict[str, np.ndarray] = {}
    for idx, arr in references.items():
        arrays[f"reference_{idx}"] = np.asarray(arr)
    if sweep_values is not None:
        arrays["sweep_values"] = np.array([float(v) for v in sweep_values])

    np.savez(path, **arrays)
    log.info("Saved reference: %s (%d sweep values)", path, len(references))
    return path


def reference_exists(domain: str, exp_key: str) -> bool:
    """Check whether a precomputed reference NPZ exists for the experiment."""
    return (_reference_dir(domain) / _reference_filename(exp_key)).exists()


def extract_references_from_fields(
    fields_path: Path | str,
    n_sweep_values: int,
) -> dict[int, np.ndarray]:
    """Extract consensus references from a fields.npz snapshot.

    The forward/agreement experiments store consensus references as
    ``consensus_{i}`` arrays in ``fields.npz``. This function reads
    them out, suitable for saving as a standalone reference NPZ.

    Returns a dict mapping sweep index to the consensus array. Missing
    indices are omitted.
    """
    fields_path = Path(fields_path)
    if not fields_path.exists():
        return {}
    refs: dict[int, np.ndarray] = {}
    try:
        with np.load(str(fields_path), allow_pickle=False) as data:
            for i in range(n_sweep_values):
                key = f"consensus_{i}"
                if key in data:
                    refs[i] = np.asarray(data[key])
    except Exception:
        log.warning("Failed to read fields from %s", fields_path, exc_info=True)
    return refs


# ── Experiments that use consensus references ──────────────────────────────

# Domain → list of experiment keys that rely on consensus references.
# NS domains have analytic or designated-solver references and are not listed.
CONSENSUS_EXPERIMENTS: dict[str, list[str]] = {
    "structural-mesh": [
        "forward/baseline",
        "forward/agreement",
    ],
    "thermal-mesh": [
        "forward/baseline",
        "forward/agreement",
        "forward/source_baseline",
        "forward/source_linearity",
    ],
}
