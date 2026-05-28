# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _NumpyEncoder and _strict_float_safe.

Every result.json flows through this encoder. These tests guard against
silent emission of non-strict JSON tokens (Infinity, NaN) and verify that
numpy/JAX types are correctly coerced to native Python types.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.benchmarks.core.io import _NumpyEncoder, _strict_float_safe


def _dumps(obj: object) -> str:
    return json.dumps(obj, cls=_NumpyEncoder, indent=2)


def _roundtrip(obj: object) -> object:
    return json.loads(_dumps(obj))


# -- _strict_float_safe -------------------------------------------------------


class TestStrictFloatSafe:
    def test_nan_becomes_none(self):
        assert _strict_float_safe(float("nan")) is None

    def test_inf_becomes_none(self):
        assert _strict_float_safe(float("inf")) is None

    def test_neg_inf_becomes_none(self):
        assert _strict_float_safe(float("-inf")) is None

    def test_finite_float_unchanged(self):
        assert _strict_float_safe(3.14) == 3.14

    def test_nested_dict(self):
        data = {"a": {"b": float("nan"), "c": 1.0}}
        result = _strict_float_safe(data)
        assert result == {"a": {"b": None, "c": 1.0}}

    def test_nested_list(self):
        data = [1.0, float("inf"), 3.0]
        result = _strict_float_safe(data)
        assert result == [1.0, None, 3.0]

    def test_tuple_preserved_as_tuple(self):
        data = (1.0, float("nan"))
        result = _strict_float_safe(data)
        assert isinstance(result, tuple)
        assert result == (1.0, None)

    def test_int_unchanged(self):
        assert _strict_float_safe(42) == 42

    def test_string_unchanged(self):
        assert _strict_float_safe("hello") == "hello"


# -- _NumpyEncoder via json.dumps ---------------------------------------------


class TestNumpyEncoderNonFinite:
    """Non-finite floats must become JSON null, never bare Infinity/NaN tokens."""

    def test_top_level_nan(self):
        assert _roundtrip(float("nan")) is None

    def test_top_level_inf(self):
        assert _roundtrip(float("inf")) is None

    def test_top_level_neg_inf(self):
        assert _roundtrip(float("-inf")) is None

    def test_nan_in_nested_dict(self):
        result = _roundtrip({"a": {"b": float("nan"), "c": 1.0}})
        assert result == {"a": {"b": None, "c": 1.0}}

    def test_nan_in_list(self):
        result = _roundtrip([1.0, float("nan"), 3.0])
        assert result == [1.0, None, 3.0]

    def test_output_is_valid_json(self):
        """The serialised string must be parseable by a strict JSON parser."""
        payload = _dumps({"x": float("nan"), "y": float("inf"), "z": 1.0})
        parsed = json.loads(payload)
        assert parsed["x"] is None
        assert parsed["y"] is None
        assert parsed["z"] == 1.0


class TestNumpyEncoderArrays:
    def test_numpy_array_finite(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert _roundtrip(arr) == [1.0, 2.0, 3.0]

    def test_numpy_array_with_nan(self):
        arr = np.array([1.0, np.nan, 3.0])
        assert _roundtrip(arr) == [1.0, None, 3.0]

    def test_numpy_array_with_inf(self):
        arr = np.array([1.0, np.inf, -np.inf])
        assert _roundtrip(arr) == [1.0, None, None]

    def test_jax_array_finite(self):
        arr = jnp.array([1.0, 2.0])
        assert _roundtrip(arr) == [1.0, 2.0]

    def test_jax_array_with_nan(self):
        arr = jnp.array([np.nan, 2.0])
        result = _roundtrip(arr)
        assert result[0] is None
        assert result[1] == 2.0

    def test_2d_numpy_array(self):
        arr = np.array([[1.0, 2.0], [3.0, np.nan]])
        result = _roundtrip(arr)
        assert result == [[1.0, 2.0], [3.0, None]]


class TestNumpyEncoderScalars:
    def test_numpy_float64(self):
        assert _roundtrip(np.float64(3.14)) == pytest.approx(3.14)

    def test_numpy_float32(self):
        assert _roundtrip(np.float32(2.5)) == pytest.approx(2.5)

    def test_numpy_float32_inf(self):
        assert _roundtrip(np.float32(np.inf)) is None

    def test_numpy_float64_nan(self):
        assert _roundtrip(np.float64(np.nan)) is None

    def test_numpy_int64(self):
        assert _roundtrip(np.int64(42)) == 42

    def test_numpy_int32(self):
        assert _roundtrip(np.int32(-7)) == -7

    def test_numpy_bool_true(self):
        assert _roundtrip(np.bool_(True)) is True

    def test_numpy_bool_false(self):
        assert _roundtrip(np.bool_(False)) is False


class TestNumpyEncoderSpecialTypes:
    def test_set_becomes_sorted_list(self):
        assert _roundtrip({3, 1, 2}) == [1, 2, 3]

    def test_callable_stringified(self):
        def my_func():
            pass

        result = _roundtrip(my_func)
        assert isinstance(result, str)
        assert "my_func" in result

    def test_class_stringified(self):
        result = _roundtrip(int)
        assert isinstance(result, str)
        assert "int" in result


class TestNumpyEncoderFinitePassthrough:
    """Normal finite data must survive a roundtrip without mutation."""

    def test_plain_dict(self):
        data = {"status": "ok", "mean_s": 0.123, "trials_s": [0.1, 0.12, 0.15]}
        assert _roundtrip(data) == data

    def test_nested_structure(self):
        data = {"by_solver": {"XLB": {"mean_s": 1.5, "std_s": 0.01}}}
        assert _roundtrip(data) == data

    def test_integers_and_strings(self):
        data = {"n_trials": 5, "solver": "XLB", "ok": True}
        assert _roundtrip(data) == data
