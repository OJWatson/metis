import numpy as np
import polars as pl
import pytest

from metis.data import SimulationDataset
from metis.design import assess_coverage, suggest_simulations, _nearest


def training(inputs, names=None):
    inputs = np.asarray(inputs)
    return SimulationDataset.from_arrays(
        inputs,
        np.zeros((len(inputs), 1, 1)),
        input_names=names or [f"p{i}" for i in range(inputs.shape[1])],
        output_names=["output"],
    )


def test_scaled_euclidean_distance_constant_dimensions_and_bounds():
    data = training([[0, 5], [10, 5]])
    report = assess_coverage(data, [[5, 5], [10, 7], [0, 5]])
    assert report["nearest_training_distance"].to_list() == pytest.approx([0.5, 2, 0])
    assert report["outside_training_bounds"].to_list() == [False, True, False]
    assert report.select(data.input_names).rows() == [(5, 5), (10, 7), (0, 5)]


def test_inside_marginal_bounds_can_have_large_joint_distance():
    data = training([[0, 0], [1, 1]])
    report = assess_coverage(data, [[0, 1]])
    assert report["outside_training_bounds"].item() is False
    assert report["nearest_training_distance"].item() == 1


def test_greedy_selects_supplied_rows_in_priority_order_and_updates_distance():
    data = training([[0], [1]])
    candidates = pl.DataFrame(
        {"extra_label": ["a", "b", "c", "d"], "p0": [2.5, 0.5, 3.0, 0.0]}
    )
    result = suggest_simulations(data, candidates, n=3)
    assert result["p0"].to_list() == [3, 2.5, 0.5]
    assert result["priority"].to_list() == [1, 2, 3]
    assert result["nearest_training_distance"].to_list() == [2, 1.5, 0.5]
    assert result["selection_distance"].to_list() == [2, 0.5, 0.5]
    assert set(result["p0"]) <= set(candidates["p0"])
    assert all("not model uncertainty" in text for text in result["interpretation"])
    assert len(result["p0"].unique()) == 3


def test_candidates_already_simulated_are_not_selected():
    data = training([[0], [1]])
    assert suggest_simulations(data, [[0], [0.5], [1]], n=1)["p0"].item() == 0.5
    with pytest.raises(ValueError, match="not already simulated"):
        suggest_simulations(data, [[0], [0.5], [1]], n=2)


@pytest.mark.parametrize(
    "candidates",
    [
        [[0.5], [0.5]],
        [[float("nan")]],
        [[float("inf")]],
        [],
        [[1, 2]],
        {"missing": 3},
        [{"p0": 0}, {"missing": 1}],
        [[None]],
    ],
)
def test_invalid_candidates_are_rejected(candidates):
    with pytest.raises(ValueError):
        suggest_simulations(training([[0], [1]]), candidates, n=1)


@pytest.mark.parametrize("n", [0, -1, True, 1.5, 4])
def test_invalid_selection_counts_are_rejected(n):
    with pytest.raises(ValueError):
        suggest_simulations(training([[0], [1]]), [[0.25], [0.5], [0.75]], n=n)


def test_named_candidate_inputs_are_reordered_to_training_schema():
    data = training([[0, 0], [1, 2]], names=["a", "b"])
    for candidates in (
        {"b": 1, "a": 0.5},
        [{"b": 1, "a": 0.5}],
        pl.DataFrame({"b": [1], "a": [0.5]}),
    ):
        report = assess_coverage(data, candidates)
        assert report.select(["a", "b"]).row(0) == (0.5, 1)
        assert report["nearest_training_distance"].item() == pytest.approx(np.sqrt(0.5))


def test_pandas_candidates_validate_names_and_reorder_without_optional_arrow():
    pd = pytest.importorskip("pandas")
    data = training([[0, 0], [1, 2]], names=["a", "b"])
    candidates = pd.DataFrame({"b": [1], "extra": [999], "a": [0.5]})
    report = assess_coverage(data, candidates)
    assert report.select(["a", "b"]).row(0) == (0.5, 1)
    assert report["nearest_training_distance"].item() == pytest.approx(np.sqrt(0.5))
    selected = suggest_simulations(data, candidates, n=1)
    assert selected.select(["a", "b"]).row(0) == (0.5, 1)
    with pytest.raises(ValueError, match="Missing candidate inputs"):
        assess_coverage(data, pd.DataFrame({"wrong_a": [0.5], "wrong_b": [1]}))
    with pytest.raises(ValueError, match="Duplicate candidate input columns"):
        assess_coverage(data, pd.DataFrame([[0.5, 1, 2]], columns=["a", "b", "b"]))


def test_reserved_input_name_fails_instead_of_overwriting_input():
    data = training([[0], [1]], names=["priority"])
    with pytest.raises(ValueError, match="conflict"):
        assess_coverage(data, [[0.5]])


def test_blocked_nearest_considers_all_points_and_matches_brute_force():
    rng = np.random.default_rng(42)
    existing = rng.normal(size=(1030, 3))
    candidates = np.concatenate([rng.normal(size=(260, 3)), existing[-1:]])
    result = _nearest(candidates, existing)
    expected = np.linalg.norm(candidates[:, None] - existing[None, :], axis=-1).min(
        axis=1
    )
    np.testing.assert_allclose(result, expected, atol=1e-15, rtol=1e-14)
    assert result[-1] == 0


def test_distances_do_not_underflow_for_distinct_candidates():
    data = training([[0], [1]])
    result = suggest_simulations(data, [[1e-200]], n=1)
    assert result["selection_distance"].item() == pytest.approx(1e-200, abs=0)
