"""Explicit replacements for legacy truncation and initial-value recipes."""

import numpy as np
import polars as pl
import pytest

from metis.data import SimulationDataset
from metis.preprocessing import initialize_static_inputs, truncate_simulations


def fixture_bank():
    frame = pl.DataFrame(
        [
            {
                "scenario": group,
                "rep": rep,
                "t": t,
                "initial": value,
                "source": value + t,
                "other": 100 + t,
            }
            for group in range(3)
            for rep, value in [(0, 10.0 + group * 20), (1, 20.0 + group * 20)]
            for t in [0, 1, 2]
        ]
    )
    return SimulationDataset.from_frame(
        frame, inputs=["initial"], outputs=["source", "other"], groups=["scenario"]
    )


def test_truncate_then_recalculate_matches_frozen_legacy_reference():
    original = fixture_bank()
    before = original.fingerprint
    result = initialize_static_inputs(
        truncate_simulations(original, start_time=1), mapping={"initial": "source"}
    )
    # Frozen reference: each original run had source = initial + t, so selecting
    # t=1,2 must copy initial+1 into the static input without mixing replicates.
    np.testing.assert_array_equal(result.inputs[:, 0], [11, 21, 31, 41, 51, 61])
    np.testing.assert_array_equal(
        result.outputs[:, :, 0],
        [[11, 12], [21, 22], [31, 32], [41, 42], [51, 52], [61, 62]],
    )
    np.testing.assert_array_equal(result.outputs[:, :, 1], np.tile([101, 102], (6, 1)))
    np.testing.assert_array_equal(result.times, [1, 2])
    assert result.group_keys == original.group_keys
    assert result.replicate_ids == original.replicate_ids
    assert original.fingerprint == before != result.fingerprint
    split = result.split()
    assert set(split.train.group_keys).isdisjoint(split.validation.group_keys)


def test_boundary_between_times_does_not_interpolate_or_reset_grid():
    result = truncate_simulations(fixture_bank(), start_time=0.5)
    np.testing.assert_array_equal(result.times, [1, 2])
    np.testing.assert_array_equal(result.inputs[:, 0], [10, 20, 30, 40, 50, 60])


def test_new_static_input_is_explicit_and_preserves_original_scenarios():
    original = fixture_bank()
    result = initialize_static_inputs(original, mapping={"new_initial": "other"})
    assert result.input_names == ("initial", "new_initial")
    np.testing.assert_array_equal(result.inputs[:, 1], 100)
    np.testing.assert_array_equal(result.outputs, original.outputs)
    assert result.group_keys == original.group_keys


@pytest.mark.parametrize("boundary", [True, float("nan"), float("inf"), "1", 9])
def test_invalid_or_empty_truncation_rejected(boundary):
    with pytest.raises(ValueError):
        truncate_simulations(fixture_bank(), start_time=boundary)


@pytest.mark.parametrize(
    "mapping", [{}, {"source": "other"}, {"initial": "missing"}, {"": "source"}]
)
def test_invalid_initial_value_mapping_rejected(mapping):
    with pytest.raises(ValueError):
        initialize_static_inputs(fixture_bank(), mapping=mapping)


def test_frame_boundary_remains_strict_about_duplicate_runs():
    frame = pl.DataFrame({"x": [1, 1], "rep": [0, 0], "t": [0, 0], "y": [2, 3]})
    with pytest.raises(ValueError, match="Duplicate"):
        SimulationDataset.from_frame(frame, inputs=["x"], outputs=["y"])
