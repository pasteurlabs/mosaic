# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for trimmed_mean.

trimmed_mean computes the ensemble reference baseline that every solver's
error is measured against. These tests verify the two code paths (n<=2
plain mean, n>2 quantile trim) and boundary conditions.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from mosaic.benchmarks.core.utils import trimmed_mean


class TestTrimmedMean:
    def test_single_array_passthrough(self):
        a = jnp.array([1.0, 2.0, 3.0])
        result = trimmed_mean([a])
        assert jnp.allclose(result, a)

    def test_two_arrays_plain_mean(self):
        a = jnp.array([0.0, 4.0])
        b = jnp.array([2.0, 6.0])
        result = trimmed_mean([a, b])
        assert jnp.allclose(result, jnp.array([1.0, 5.0]))

    def test_three_or_more_trims_outlier(self):
        # 4 arrays where one is an extreme outlier. The trimmed mean
        # should give a result closer to the non-outlier values.
        normal = jnp.array([1.0, 1.0, 1.0])
        arrays = [normal, normal, normal, jnp.array([100.0, 100.0, 100.0])]
        result = trimmed_mean(arrays)
        plain = jnp.stack(arrays).mean(axis=0)
        # Trimmed result should be closer to 1.0 than the plain mean.
        assert float(jnp.abs(result - 1.0).max()) < float(jnp.abs(plain - 1.0).max())

    def test_identical_arrays(self):
        a = jnp.array([5.0, 10.0])
        result = trimmed_mean([a, a, a, a])
        assert jnp.allclose(result, a)

    def test_five_arrays_excludes_extreme(self):
        # With 5 arrays, the 5th and 95th percentile should bracket the
        # central values and exclude the min/max.
        arrays = [jnp.array([float(i)]) for i in [1, 2, 3, 4, 100]]
        result = trimmed_mean(arrays)
        # The outlier (100) should be trimmed; result should be < plain mean.
        plain = jnp.stack(arrays).mean(axis=0)
        assert float(result[0]) < float(plain[0])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            trimmed_mean([])
