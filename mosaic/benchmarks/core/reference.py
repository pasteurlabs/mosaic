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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Root of the problems package — references live alongside problem configs.
_PROBLEMS_DIR: Path = Path(__file__).resolve().parents[1] / "problems"


# Domains whose CLI slug does not map to the package name by ``-`` → ``_``.
_PACKAGE_ALIASES: dict[str, str] = {
    "ns-3d-grid": "navier_stokes_3d_grid",
}


def _domain_slug_to_package(domain: str) -> str:
    """Convert a CLI domain slug (e.g. ``structural-mesh``) to the Python package name."""
    if domain in _PACKAGE_ALIASES:
        return _PACKAGE_ALIASES[domain]
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


# ── Experiments that need precomputed references ───────────────────────────

# Domain → list of experiment keys that need a checked-in reference NPZ.
# Reasons an experiment lands here:
#   - consensus-based (mesh domains): no analytic solution, needs trimmed-mean
#     across all solvers — but single-solver CI jobs have only one peer;
#   - designated reference solver on different hardware: e.g. the cylinder
#     experiment uses OpenFOAM (CPU) as reference, but GPU-only solvers can't
#     see OpenFOAM's output during their aggregate pass.
PRECOMPUTED_EXPERIMENTS: dict[str, list[str]] = {
    "ns-grid": [
        "forward/cylinder",
    ],
    # The 3D TGV forward/agreement reference is a *converged* spectral run,
    # not the linearized analytic decay (issue #123). Its strategy lives in
    # CONVERGED_REFERENCES below.
    "ns-3d-grid": [
        "forward/agreement",
    ],
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

# Backward-compat alias
CONSENSUS_EXPERIMENTS = PRECOMPUTED_EXPERIMENTS


@dataclass(frozen=True)
class ConvergedSpec:
    """How to build a *converged* reference for one experiment.

    A converged reference is a single high-fidelity solver run at a resolution
    well above the benchmark grid, spectrally truncated back down — the true
    continuous solution sampled on the benchmark grid, as opposed to a
    trimmed-mean *consensus* across peers (which shares their common-mode bias).

    Attributes:
        solver: Name of the reference solver (must be a spectral, periodic
            solver so spectral downsampling is exact).
        high_n: Grid resolution to integrate at before downsampling.
    """

    solver: str
    high_n: int


# (domain, exp_key) → ConvergedSpec for experiments whose reference is a
# converged spectral run rather than a peer consensus. Experiments in
# PRECOMPUTED_EXPERIMENTS but absent here default to the consensus strategy.
CONVERGED_REFERENCES: dict[tuple[str, str], ConvergedSpec] = {
    # 3D TGV forward accuracy: the linearized analytic decay is only a
    # short-horizon approximation, so score against a converged spectral run
    # instead (issue #123). Exponax is spectral and enforces incompressibility
    # to machine precision; at N=64 the field is band-limited well below the
    # N=16 Nyquist cutoff, so truncation to the benchmark grid is exact.
    ("ns-3d-grid", "forward/agreement"): ConvergedSpec(solver="Exponax", high_n=64),
}


def converged_spec(domain: str, exp_key: str) -> ConvergedSpec | None:
    """Return the converged-reference spec for an experiment, or None."""
    return CONVERGED_REFERENCES.get((domain, exp_key))


def spectral_downsample(field: np.ndarray, n_target: int) -> np.ndarray:
    """Spectrally truncate a periodic field to ``n_target`` points per axis.

    ``field`` has shape ``(N, N, N, C)`` (or ``(N, N, C)``) on a uniform
    periodic grid. Truncation keeps the ``n_target`` lowest-|k| Fourier modes
    per spatial axis; it is exact when the field carries no energy above the
    target Nyquist cutoff. Returns the same layout at ``n_target`` per axis.
    """
    spatial = field.ndim - 1
    axes = tuple(range(spatial))
    n = field.shape[0]
    if n_target > n:
        raise ValueError(f"n_target={n_target} exceeds source resolution {n}")
    fh = np.fft.fftn(field, axes=axes)
    h = n_target // 2
    idx = np.concatenate([np.arange(h), np.arange(n - h, n)])
    fh = fh[np.ix_(*([idx] * spatial), range(field.shape[-1]))]
    out = np.fft.ifftn(fh, axes=axes) * (n_target / n) ** spatial
    return np.real(out)


def is_precomputed_experiment(domain: str, exp_key: str) -> bool:
    """Whether ``domain``/``exp_key`` is served by a checked-in reference.

    Designated precomputed experiments prefer their reference NPZ over any
    runtime reference selection (analytic or consensus) — the checked-in
    field is the intended ground truth for the accuracy metric.
    """
    return exp_key in PRECOMPUTED_EXPERIMENTS.get(domain, [])
