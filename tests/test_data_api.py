"""Data correctness at the public simulation-bank boundary."""

import json

import numpy as np
import polars as pl
import pytest

from metis.data import DatasetSplit, SimulationDataset, Standardizer


def bank(n_groups=8, n_reps=2):
    return pl.DataFrame(
        [
            {
                "scenario": f"s{group}",
                "beta": group * 0.1,
                "fixed": 7.0,
                "rep": rep,
                "t": t,
                "incidence": group + rep * 0.2 + t,
                "zero": 0.0,
            }
            for group in range(n_groups)
            for rep in range(n_reps)
            for t in [0, 2, 7]
        ]
    )


def load(frame):
    return SimulationDataset.from_frame(
        frame,
        inputs=["beta", "fixed"],
        outputs=["incidence", "zero"],
        groups=["scenario"],
    )


def test_frame_and_parquet_preserve_trajectories_and_metadata(tmp_path):
    frame = bank().sample(fraction=1, shuffle=True, seed=7)
    data = load(frame)
    path = tmp_path / "simulations.parquet"
    frame.write_parquet(path)
    restored = SimulationDataset.from_parquet(
        path,
        inputs=["beta", "fixed"],
        outputs=["incidence", "zero"],
        groups=["scenario"],
    )
    assert data.inputs.shape == (16, 2)
    assert data.outputs.shape == (16, 3, 2)
    assert data.inputs.dtype == data.outputs.dtype == np.float32
    np.testing.assert_array_equal(data.times, [0, 2, 7])
    np.testing.assert_allclose(data.outputs[1, :, 0], [0.2, 2.2, 7.2])
    assert data.group_keys[:2] == (("s0",), ("s0",))
    assert data.replicate_ids[:2] == (0, 1)
    assert data.fingerprint == restored.fingerprint == load(bank()).fingerprint
    assert len(data.fingerprint) == 64
    with pytest.raises(ValueError, match="read-only"):
        data.inputs[0, 0] = 100


def test_duplicate_run_rows_are_not_silently_dropped():
    frame = bank()
    with pytest.raises(ValueError, match="Duplicate group/replicate/time"):
        load(pl.concat([frame, frame.head(1)]))


def test_pandas_three_dataframe_loads_with_exact_schema_and_trajectories():
    pandas = pytest.importorskip("pandas", minversion="3.0")
    frame = pandas.DataFrame(bank().to_dict(as_series=False))
    assert type(frame).__module__.split(".")[0] == "pandas"
    actual = load(frame)
    expected = load(bank())
    np.testing.assert_array_equal(actual.inputs, expected.inputs)
    np.testing.assert_array_equal(actual.outputs, expected.outputs)
    assert actual.group_keys == expected.group_keys
    assert actual.fingerprint == expected.fingerprint


def test_pandas_duplicate_column_names_rejected_before_conversion():
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(bank().to_dict(as_series=False))
    # Repeated names must fail at the table boundary, not collapse or produce
    # an incidental pandas "DataFrame has no tolist" attribute error.
    frame = pandas.concat([frame, frame[["beta"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate column names"):
        load(frame)


def test_unequal_lengths_cannot_silently_mix_runs():
    # Total length remains divisible by three: older reshape-only loading could
    # merge parts of the two realizations and produce plausible array shapes.
    frame = bank(1, 2).filter(~((pl.col("rep") == 0) & (pl.col("t") == 7)))
    extra = frame.filter((pl.col("rep") == 1) & (pl.col("t") == 7)).with_columns(
        pl.lit(9).alias("t")
    )
    with pytest.raises(ValueError, match="time grid or length"):
        load(pl.concat([frame, extra], how="vertical_relaxed"))


def test_equal_lengths_with_different_time_grids_are_rejected():
    frame = bank().with_columns(
        pl.when((pl.col("rep") == 1) & (pl.col("t") == 7))
        .then(8)
        .otherwise(pl.col("t"))
        .alias("t")
    )
    with pytest.raises(ValueError, match="time grid or length"):
        load(frame)


def test_static_inputs_are_checked_within_realization():
    frame = bank().with_columns(
        pl.when(pl.col("t") == 7).then(5.0).otherwise(pl.col("fixed")).alias("fixed")
    )
    with pytest.raises(ValueError, match="inputs must be static"):
        load(frame)


@pytest.mark.parametrize(
    "column,value",
    [("beta", np.nan), ("incidence", np.inf), ("fixed", None), ("scenario", None)],
)
def test_nonfinite_or_missing_values_rejected(column, value):
    frame = bank().with_columns(pl.lit(value).alias(column))
    with pytest.raises(ValueError, match="missing|NaN|infinite"):
        load(frame)


def test_numeric_strings_are_rejected_instead_of_silently_coerced():
    with pytest.raises(ValueError, match="numeric"):
        load(bank().with_columns(pl.col("beta").cast(pl.String)))


def test_missing_and_overlapping_columns_explain_failure():
    with pytest.raises(ValueError, match="missing required columns"):
        load(bank().drop("fixed"))
    with pytest.raises(ValueError, match="distinct"):
        SimulationDataset.from_frame(bank(), inputs=["beta"], outputs=["beta"])
    with pytest.raises(ValueError, match="duplicate"):
        SimulationDataset.from_frame(
            bank(), inputs=["beta", "beta"], outputs=["incidence"]
        )


def test_split_keeps_scenarios_together_and_is_reproducible():
    data = load(bank(11, 3))
    first = data.split(seed=12)
    second = data.split(seed=12)
    groups = [set(d.group_keys) for d in (first.train, first.validation, first.test)]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert set.union(*groups) == set(data.group_keys)
    assert len(first.train) + len(first.validation) + len(first.test) == len(data)
    assert first.train.fingerprint == second.train.fingerprint
    assert first.validation.fingerprint == second.validation.fingerprint
    assert first.test.fingerprint == second.test.fingerprint


def test_small_split_never_returns_empty_training_or_validation():
    split = load(bank(2, 1)).split(validation_fraction=0.01, test_fraction=0)
    assert len(split.train) == len(split.validation) == 1
    assert split.test is None
    with pytest.raises(ValueError, match="at least 3 distinct scenario groups"):
        load(bank(2)).split()
    with pytest.raises(ValueError, match="at least 2 distinct scenario groups"):
        load(bank(1)).split(test_fraction=0)
    # Fractions near boundaries still reserve training and both requested sets.
    split = load(bank(3)).split(validation_fraction=0.01, test_fraction=0.98)
    assert all(
        len(set(d.group_keys)) == 1 for d in (split.train, split.validation, split.test)
    )


@pytest.mark.parametrize(
    "validation,test", [(-0.1, 0.1), (0.5, 0.5), (np.nan, 0), (0, np.inf)]
)
def test_invalid_split_fractions_rejected(validation, test):
    with pytest.raises(ValueError, match="fractions"):
        load(bank()).split(validation_fraction=validation, test_fraction=test)


def test_explicit_splits_reject_group_leakage_even_with_different_replicates():
    data = load(bank())
    with pytest.raises(ValueError, match="overlap"):
        DatasetSplit.from_datasets(data.subset([0]), data.subset([1]))
    split = DatasetSplit.from_datasets(
        data.subset([0, 1]), data.subset([2, 3]), data.subset([4, 5])
    )
    assert len(split.train) == 2
    with pytest.raises(ValueError, match="overlap"):
        DatasetSplit.from_datasets(split.train, split.validation, split.validation)


def test_explicit_splits_check_schema_and_times():
    data = load(bank())
    other = SimulationDataset.from_arrays(
        [[4.0, 7.0]],
        [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]],
        input_names=["beta", "fixed"],
        output_names=["incidence", "zero"],
        times=[0, 2, 8],
        group_keys=[("different",)],
    )
    with pytest.raises(ValueError, match="schema and time grid"):
        DatasetSplit.from_datasets(data, other)


def test_arrays_default_to_input_scenario_groups_and_validate():
    data = SimulationDataset.from_arrays(
        [[1], [1], [2]],
        np.arange(6).reshape(3, 2, 1),
        input_names=["theta"],
        output_names=["y"],
    )
    assert data.group_keys == ((1,), (1,), (2,))
    assert data.replicate_ids == (0, 1, 2)
    split = data.split(test_fraction=0)
    assert set(split.train.group_keys).isdisjoint(split.validation.group_keys)
    np.testing.assert_array_equal(data.times, [0, 1])
    with pytest.raises(ValueError, match="strictly increasing"):
        SimulationDataset.from_arrays(
            [[1]], [[[1], [2]]], input_names=["x"], output_names=["y"], times=[1, 1]
        )


def test_standardizer_safe_constants_and_serializable_inverse():
    values = np.array([[1.0, 5.0], [3.0, 5.0]], dtype=np.float32)
    fitted = Standardizer.fit(values)
    np.testing.assert_array_equal(fitted.constant_mask, [False, True])
    np.testing.assert_array_equal(fitted.transform(values), [[-1, 0], [1, 0]])
    # A value varying in a column constant in training is not discarded.
    assert fitted.transform([[2, 9]])[0, 1] == 4
    restored = Standardizer.from_dict(json.loads(json.dumps(fitted.to_dict())))
    np.testing.assert_allclose(
        restored.inverse_transform(restored.transform(values)), values
    )
    one = Standardizer.fit(np.array([[[0.0], [8.0]]]))
    assert one.constant_mask.all()
    assert np.isfinite(one.transform([[[0.0], [8.0]]])).all()


def test_scaling_uses_only_arrays_passed_to_fit():
    train = np.array([[0.0], [2.0]])
    held_out = np.array([[1000.0]])
    fitted = Standardizer.fit(train)
    assert fitted.mean.item() == 1
    assert fitted.scale.item() == 1
    assert fitted.transform(held_out).item() == 999


def test_standardizer_supports_trajectory_samples_with_leading_batch_axes():
    training = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    fitted = Standardizer.fit(training)
    draws = np.arange(90, dtype=np.float32).reshape(3, 5, 3, 2)
    np.testing.assert_allclose(
        fitted.inverse_transform(fitted.transform(draws)), draws, atol=1e-5
    )
    with pytest.raises(ValueError, match="trailing shape"):
        fitted.transform(np.zeros((3, 5, 2, 3)))


def test_legacy_loader_validates_boundaries_and_constants():
    from dataloader import make_tensors

    frame = bank(2, 1).with_columns(
        pl.col("scenario").str.slice(1).cast(pl.Int64).alias("param_index")
    )
    x, y, _, _ = make_tensors(frame, ["beta", "fixed"], ["incidence", "zero"], 3)
    assert np.isfinite(x.numpy()).all() and np.isfinite(y.numpy()).all()
    np.testing.assert_array_equal(x[:, 1], 0)
    np.testing.assert_array_equal(y[:, 1], 0)
    broken = frame.filter(~((pl.col("param_index") == 0) & (pl.col("t") == 7)))
    extra = frame.filter(
        (pl.col("param_index") == 1) & (pl.col("t") == 7)
    ).with_columns(pl.lit(9).alias("t"))
    with pytest.raises(ValueError, match="unequal trajectory lengths"):
        make_tensors(
            pl.concat([broken, extra], how="vertical_relaxed"),
            ["beta", "fixed"],
            ["incidence", "zero"],
            3,
        )


def test_legacy_splitting_fits_training_only_and_handles_two_groups(tmp_path):
    import torch
    from dataloader import prepare_splits

    frame = bank(2, 1)
    path = tmp_path / "bank.parquet"
    frame.write_parquet(path)
    train, validation, input_stats, output_stats = prepare_splits(
        path,
        0.01,
        42,
        ["scenario"],
        ["beta", "fixed"],
        ["incidence", "zero"],
        3,
        None,
        None,
    )
    assert len(train) == len(validation) == 3
    assert input_stats.means[0, 0] == pytest.approx(train["beta"][0])
    np.testing.assert_allclose(
        output_stats.means[0], train.sort("t")["incidence"].to_numpy()
    )
    assert torch.equal(input_stats.stds, torch.ones_like(input_stats.stds))
    assert torch.equal(output_stats.stds, torch.ones_like(output_stats.stds))
    assert input_stats.means[0, 0] != pytest.approx(frame["beta"].mean())
