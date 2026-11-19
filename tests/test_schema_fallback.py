"""Private-schema robustness: positional fallback when checkpoint paths rename."""
from __future__ import annotations

import numpy as np
import pytest

from rsi.platform.plans import _ordered_leaves, flatten_row


SPECS = [[["key", "features"], ["index", index]] for index in range(3)]


def test_strict_paths_still_work():
    values = flatten_row(SPECS, {"features": [1.0, 2.0, 3.0]})
    assert np.allclose(values, [1.0, 2.0, 3.0])


def test_renamed_schema_falls_back_to_sorted_leaves():
    inputs = {"probe": {"sensor_0000": 1.0, "sensor_0001": 2.0, "sensor_0002": 3.0}}
    values = flatten_row(SPECS, inputs)
    assert np.allclose(values, [1.0, 2.0, 3.0])
    assert _ordered_leaves(inputs) == [1.0, 2.0, 3.0]


def test_nested_extra_wrapper_is_flattened_in_order():
    inputs = {"batch": {"probe": {"sensor_0000": 4.0, "sensor_0001": 5.0, "sensor_0002": 6.0}}}
    assert np.allclose(flatten_row(SPECS, inputs), [4.0, 5.0, 6.0])


def test_too_few_features_raises_instead_of_silently_padding():
    inputs = {"probe": {"sensor_0000": 1.0, "sensor_0001": 2.0}}
    with pytest.raises(ValueError):
        flatten_row(SPECS, inputs)
