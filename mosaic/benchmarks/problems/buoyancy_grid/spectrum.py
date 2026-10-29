# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate structural null directions from non-null Jacobian conditioning."""

import math
from collections.abc import Sequence
from itertools import pairwise


def spectrum_diagnostics(
    values: Sequence[float], n: int, tolerance: float = 1e-6
) -> dict:
    """Summarize a full packed-state spectrum; tolerance is relative to sigma_1.

    Closed-wall MAC velocities have (N-1)^2 divergence-free degrees of freedom;
    temperature adds N^2. The non-null ratio describes the map on the orthogonal
    complement of its numerical kernel, not a separately projected Jacobian.
    """
    dimension = 3 * n * n
    expected = (n - 1) ** 2 + n * n
    if n < 2 or len(values) != dimension:
        raise ValueError("Expected a complete 3*N*N buoyancy spectrum")
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Singular values must be finite and nonnegative")
    if any(a < b for a, b in pairwise(values)):
        raise ValueError("Singular values must be descending")
    active = [v for v in values if v > tolerance * values[0]]
    return {
        "dimension": dimension,
        "expected_rank": expected,
        "structural_nullity": dimension - expected,
        "numerical_rank": len(active),
        "relative_cutoff": tolerance,
        "condition_non_null": values[0] / active[-1] if active else None,
        "condition_full": values[0] / values[-1] if values[-1] > 0 else None,
    }
