"""Analytic diagnostics tests: these assert scientific quantities, not training loss."""

import json

import numpy as np
import pytest

from metis.data import SimulationDataset
from metis.evaluation import evaluate


class FixedSurrogate:
    def __init__(self, data, predictions=None, draws=None, model_name=None):
        self.input_names = data.input_names
        self.output_names = data.output_names
        self.times = data.times
        self.model_name = model_name or ("itarflow" if draws is not None else "mlp")
        self.training_metadata = {
            "training_fingerprint": "training-fixture",
            "validation_fingerprint": "validation-fixture",
            "train_group_keys": [["training-only"]],
            "validation_group_keys": [["validation-only"]],
        }
        self.predictions = predictions
        self.draws = draws
        self.calls = []

    def predict(self, inputs):
        self.calls.append(len(inputs))
        return self.predictions[inputs[:, 0].astype(int)].copy()

    def sample(self, inputs, **kwargs):
        self.calls.append((len(inputs), kwargs))
        return self.draws[inputs[:, 0].astype(int)].copy()


def bank(
    inputs, outputs, input_names=("parameter",), output_names=("count",), **kwargs
):
    return SimulationDataset.from_arrays(
        np.asarray(inputs),
        np.asarray(outputs),
        input_names=input_names,
        output_names=output_names,
        **kwargs,
    )


def test_original_unit_metrics_and_replicate_aggregation():
    data = bank([[0], [0], [1]], [[[0], [2]], [[2], [4]], [[10], [14]]])
    model = FixedSurrogate(data, predictions=np.array([[[2], [2]], [[11], [11]]]))
    report = evaluate(model, data, batch_size=2)
    assert report.metrics["rmse"] == pytest.approx(np.sqrt(3))
    assert report.metrics["mae"] == pytest.approx(4 / 3)
    # Group mean SSEs are 1 and 5; groups receive equal weight here.
    assert report.metrics["replicate_mean_rmse"] == pytest.approx(np.sqrt(3))
    assert report.by_group["n_replicates"].to_list() == [2, 1]
    assert report.by_group["replicate_mean_rmse"].to_list() == pytest.approx(
        [1, np.sqrt(5)]
    )
    assert model.calls == [2, 1]
    assert report.worst_groups(1)["input:parameter"].item() == 1
    assert report.metadata["dataset_fingerprint"] == data.fingerprint
    assert report.metadata["training_fingerprint"] == "training-fixture"
    assert report.metadata["model_fingerprint"] is None
    assert report.metadata["model_fingerprint_status"].startswith("unavailable:")
    assert report.metadata["resolved_model_options"] is None
    assert "original output units" in report.metadata["units"]
    assert report.lower is None and report.upper is None


def test_aggregate_rmse_weights_values_in_their_original_units():
    data = bank([[0]], [[[0, 0]]], output_names=("small", "large"))
    report = evaluate(FixedSurrogate(data, predictions=np.array([[[3, 4]]])), data)
    assert report.by_output["rmse"].to_list() == [3, 4]
    assert report.metrics["rmse"] == pytest.approx(np.sqrt((9 + 16) / 2))
    assert report.metrics["mae"] == 3.5


def test_group_input_names_cannot_overwrite_metrics():
    data = bank([[0]], [[[2]]], input_names=("rmse",))
    report = evaluate(FixedSurrogate(data, predictions=np.array([[[5]]])), data)
    assert report.by_group["rmse"].item() == 3
    assert report.by_group["input:rmse"].item() == 0


def test_group_with_varying_initial_conditions_has_no_replicate_mean_metric():
    data = bank([[0], [1]], [[[1]], [[3]]], group_keys=[("scenario",), ("scenario",)])
    report = evaluate(FixedSurrogate(data, predictions=np.array([[[1]], [[1]]])), data)
    assert report.by_group["replicate_mean_rmse"].item() is None
    assert report.by_group["input:parameter"].item() is None
    assert "replicate_mean_rmse" not in report.metrics


def test_stochastic_crps_matches_pairwise_definition_and_interval_quantiles():
    data = bank([[0], [1]], [[[1], [4]], [[7], [2]]])
    draws = np.array(
        [
            [[[0], [2]], [[2], [6]], [[4], [8]]],
            [[[5], [0]], [[7], [2]], [[9], [4]]],
        ],
        dtype=float,
    )
    model = FixedSurrogate(data, draws=draws)
    report = evaluate(model, data, n_samples=3, batch_size=1)
    truth = data.outputs
    pairwise = np.abs(draws[:, :, None] - draws[:, None, :]).mean(axis=(1, 2))
    expected_crps = np.abs(draws - truth[:, None]).mean(axis=1) - 0.5 * pairwise
    lo, hi = np.quantile(draws, [0.05, 0.95], axis=1)
    assert report.metrics["crps"] == pytest.approx(expected_crps.mean())
    assert report.by_output["crps"].item() == pytest.approx(expected_crps.mean())
    np.testing.assert_allclose(report.lower, lo)
    np.testing.assert_allclose(report.upper, hi)
    assert report.metrics["predictive_coverage_90"] == np.mean(
        (truth >= lo) & (truth <= hi)
    )
    assert report.metrics["predictive_interval_width_90"] == pytest.approx(
        np.mean(hi - lo)
    )
    np.testing.assert_allclose(report.predictions, draws.mean(axis=1))
    assert model.calls[0][1]["seed"] == 42
    assert model.calls[1][1]["seed"] != 42
    assert "Pointwise predictive" in report.metadata["uncertainty"]
    assert "no surrogate-error confidence intervals" in report.metadata["uncertainty"]
    assert report.metadata["batch_size"] == 1


def test_draw_and_prediction_shapes_checked_before_metrics():
    data = bank([[0]], [[[1], [2]]])
    model = FixedSurrogate(data, draws=np.zeros((1, 3, 1, 2)))
    with pytest.raises(ValueError, match="shape"):
        evaluate(model, data, n_samples=3)
    model = FixedSurrogate(data, predictions=np.array([[[np.nan], [1]]]))
    with pytest.raises(ValueError, match="non-finite"):
        evaluate(model, data)


def test_empirical_crps_is_invariant_to_large_common_offset():
    data = bank([[0]], [[[1e9]]])
    draws = np.array([[[[1e9 - 2]], [[1e9]], [[1e9 + 2]]]])
    report = evaluate(FixedSurrogate(data, draws=draws), data, n_samples=3)
    assert report.metrics["crps"] == pytest.approx(4 / 9)


def test_overlap_warning_and_metadata_distinguish_training_and_tuning():
    data = bank([[0], [1]], [[[1]], [[2]]])
    model = FixedSurrogate(data, predictions=data.outputs)
    model.training_metadata["train_group_keys"] = [[0]]
    model.training_metadata["validation_group_keys"] = [[1]]
    with pytest.warns(UserWarning, match="not independent test results"):
        report = evaluate(model, data)
    assert report.metadata["training_group_overlap"] == 1
    assert report.metadata["tuning_group_overlap"] == 1
    assert "absence of overlap does not establish" in report.metadata["overlap_check"]


def test_explicit_constraints_check_all_draws_without_clipping():
    data = bank([[0], [1]], np.zeros((2, 2, 2)), output_names=("a", "b"))
    draws = np.array(
        [
            [[[-1, 4], [2, 1]], [[1, 2], [0, 3]]],
            [[[0, 3], [1, 2]], [[1, 2], [2, 2]]],
        ],
        dtype=float,
    )
    original = draws.copy()
    constraints = {
        "nonnegative": ["a"],
        "nondecreasing": ["a"],
        "conserved_total": {"outputs": ["a", "b"], "value": 3, "atol": 0},
    }
    model = FixedSurrogate(data, draws=draws)
    report = evaluate(model, data, n_samples=2, constraints=constraints, batch_size=1)
    checked = report.metadata["constraints"]
    assert checked["nonnegative:a:violation_fraction"] == pytest.approx(1 / 8)
    assert checked["nondecreasing:a:violation_fraction"] == pytest.approx(1 / 4)
    assert checked["conserved_total:violation_fraction"] == pytest.approx(1 / 8)
    np.testing.assert_array_equal(draws, original)
    assert report.metadata["constraint_definitions"] == constraints
    without_checks = evaluate(model, data, n_samples=2)
    assert without_checks.metadata["constraints"] == {}


@pytest.mark.parametrize(
    "constraints",
    [
        {"nonnegative": "count"},
        {"nonnegative": ["missing"]},
        {"nonnegative": ["count", "count"]},
        {"unknown": []},
        {"conserved_total": {"outputs": ["count"], "value": 1}},
        {"conserved_total": {"outputs": ["count"], "value": 1, "atol": -1}},
        {"conserved_total": {"outputs": ["count"], "value": float("nan"), "atol": 0}},
        {"conserved_total": {"outputs": [], "value": 0, "atol": 0}},
    ],
)
def test_invalid_constraints_fail_before_prediction(constraints):
    data = bank([[0]], [[[1]]])
    model = FixedSurrogate(data, predictions=data.outputs)
    with pytest.raises(ValueError):
        evaluate(model, data, constraints=constraints)
    assert model.calls == []


@pytest.mark.parametrize(
    "kwargs", [{"batch_size": 0}, {"batch_size": True}, {"n_samples": 1}, {"seed": -1}]
)
def test_invalid_sampling_arguments(kwargs):
    data = bank([[0]], [[[1]]])
    model = FixedSurrogate(data, draws=np.ones((1, 2, 1, 1)))
    with pytest.raises(ValueError):
        evaluate(model, data, **kwargs)


def test_single_output_single_run_plot_and_json(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    data = bank([[0]], [[[1]]])
    report = evaluate(
        FixedSurrogate(data, draws=np.array([[[[0]], [[2]]]])), data, n_samples=2
    )
    figure = report.plot()
    assert len(figure.axes) == 1
    assert "count" == figure.axes[0].get_ylabel()
    assert "predictive interval" in figure.axes[0].get_legend_handles_labels()[1][-1]
    plt.close(figure)
    path = tmp_path / "diagnostics.json"
    report.save(path)
    saved = json.loads(path.read_text())
    assert saved["metrics"] == report.metrics
    assert saved["metadata"]["model"] == "itarflow"


def test_public_surrogate_evaluate_in_original_units():
    from metis.surrogate import Surrogate, TrainingConfig

    data = bank(np.arange(5)[:, None], (100 + 10 * np.arange(5))[:, None, None])
    train, test = data.subset([0, 2, 4]), data.subset([1, 3])
    model = Surrogate("mlp", {"hidden_dim": 8, "num_blocks": 1}).fit(
        train,
        config=TrainingConfig(steps=2, batch_size=3, device="cpu"),
    )
    prediction = model.predict(test.inputs)
    report = model.evaluate(test)
    assert report.metrics["rmse"] == pytest.approx(
        np.sqrt(np.mean((prediction.astype(float) - test.outputs) ** 2))
    )
    assert report.metadata["training_fingerprint"] == train.fingerprint
    assert report.metadata["training_group_overlap"] == 0
    np.testing.assert_array_equal(report.predictions, prediction)


def _fitted_provenance_model(*, width=8, seed=17):
    from metis.surrogate import Surrogate, TrainingConfig

    data = bank(np.arange(5)[:, None], (100 + 10 * np.arange(5))[:, None, None])
    train, validation, test = data.subset([0, 2, 4]), data.subset([1]), data.subset([3])
    model = Surrogate("mlp", {"hidden_dim": width, "num_blocks": 1}).fit(
        train,
        validation,
        config=TrainingConfig(
            steps=3, batch_size=3, eval_every=1, seed=seed, device="cpu"
        ),
    )
    return model, test


def test_exact_model_provenance_survives_save_load_and_records_actual_fit(tmp_path):
    from metis import Surrogate

    model, test = _fitted_provenance_model()
    report = model.evaluate(test)
    metadata = report.metadata
    assert metadata["resolved_model_options"] == model.resolved_model_options
    assert metadata["model_config"] == model.model_config
    assert metadata["parameter_count"] == model.parameter_count
    assert metadata["training_config"] == model.training_metadata["config"]
    assert metadata["training_config"]["seed"] == 17
    summary = metadata["training_summary"]
    assert summary["best_step"] == model.training_metadata["best_step"]
    assert summary["selected_step"] == summary["best_step"]
    assert summary["selected_record"] == next(
        record for record in model.history if record["step"] == summary["selected_step"]
    )
    assert summary["final_record"] == model.history[-1]
    assert summary["completed_steps"] == len(model.history)
    assert metadata["model_fingerprint_status"].startswith("complete:")
    assert len(metadata["model_fingerprint"]) == 64
    loaded = Surrogate.load(model.save(tmp_path / "model.pt"))
    loaded_report = loaded.evaluate(test, batch_size=1)
    assert loaded_report.metadata["model_fingerprint"] == metadata["model_fingerprint"]
    assert loaded_report.metadata["training_config"] == metadata["training_config"]
    report.save(tmp_path / "evaluation.json")
    assert (
        json.loads((tmp_path / "evaluation.json").read_text())["metadata"] == metadata
    )
    # Later metadata edits must not mutate an earlier report's provenance.
    model.training_metadata["config"]["seed"] = 999
    model.history[-1]["train_loss"] = -1
    assert metadata["training_config"]["seed"] == 17
    assert summary["final_record"]["train_loss"] != -1


def test_fingerprint_distinguishes_weights_and_scalers_for_same_training_bank():
    import torch
    from metis import Standardizer

    model, test = _fitted_provenance_model()
    initial = model.evaluate(test).metadata["model_fingerprint"]
    with torch.no_grad():
        next(model.model.parameters()).add_(0.01)
    different_weights = model.evaluate(test).metadata["model_fingerprint"]
    assert different_weights != initial
    scaler = model.output_scaler
    model.output_scaler = Standardizer(
        scaler.mean + 1, scaler.scale, scaler.constant_mask
    )
    different_scaler = model.evaluate(test).metadata["model_fingerprint"]
    assert different_scaler != different_weights


def test_provenance_distinguishes_widths_and_training_seeds():
    baseline, test = _fitted_provenance_model(width=8, seed=17)
    different_width, _ = _fitted_provenance_model(width=16, seed=17)
    different_seed, _ = _fitted_provenance_model(width=8, seed=18)
    reports = [
        model.evaluate(test) for model in (baseline, different_width, different_seed)
    ]
    assert len({report.metadata["training_fingerprint"] for report in reports}) == 1
    assert len({report.metadata["model_fingerprint"] for report in reports}) == 3
    assert [
        report.metadata["resolved_model_options"]["hidden_dim"] for report in reports
    ] == [8, 16, 8]
    assert [report.metadata["training_config"]["seed"] for report in reports] == [
        17,
        17,
        18,
    ]
