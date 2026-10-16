"""Reuse must save generation without changing which observations are scored."""

from types import SimpleNamespace

import numpy as np
import pytest

from metis.data import SimulationDataset
from metis.evaluation import evaluate


class RecordingGenerator:
    def __init__(self, data, *, conditioned=False, random=False):
        self.input_names = data.input_names
        self.training_output_names = data.output_names
        self.observed_outputs = ("observed",) if conditioned else ()
        self.output_names = tuple(
            name for name in data.output_names if name not in self.observed_outputs
        )
        self.times = data.times
        self.model_name = "flow_matching" if conditioned else "itarflow"
        self.training_metadata = {}
        self.calls = []
        self.random = random

    def ensembles(self, inputs, n_samples, seed, conditioning=None):
        offset = inputs[:, :1, None, None]
        if conditioning is not None:
            offset = offset + conditioning[:, None].sum(axis=(2, 3), keepdims=True)
        shape = (len(inputs), n_samples, len(self.times), len(self.output_names))
        if self.random:
            residual = np.random.default_rng(seed).normal(size=shape)
        else:
            residual = np.linspace(-2, 4, n_samples)[None, :, None, None]
        return np.broadcast_to(offset + residual, shape).copy()

    def sample(self, inputs, *, n_samples, seed, conditioning=None, **kwargs):
        self.calls.append(
            {
                "inputs": inputs.copy(),
                "seed": seed,
                "conditioning": None if conditioning is None else conditioning.copy(),
            }
        )
        return self.ensembles(inputs, n_samples, seed, conditioning)


def make_bank(inputs, outputs, *, conditioned=False, group_keys=None):
    return SimulationDataset.from_arrays(
        np.asarray(inputs),
        np.asarray(outputs),
        input_names=("parameter",),
        output_names=("count", "observed") if conditioned else ("count",),
        group_keys=group_keys,
    )


def test_reuse_scores_every_unequal_nonadjacent_replicate_and_timepoint():
    # Seven runs at input 0, two at 1, and two at 2. No row subsampling or
    # equal-context weighting may replace the original eleven-run weighting.
    inputs = [[0], [1], [0], [2], [0], [1], [0], [2], [0], [0], [0]]
    truth = np.arange(22, dtype=float).reshape(11, 2, 1) / 3 - 2
    data = make_bank(inputs, truth)
    model = RecordingGenerator(data)
    report = evaluate(
        model,
        data,
        n_samples=3,
        batch_size=2,
        reuse_samples=True,
        constraints={"nonnegative": ["count"], "nondecreasing": ["count"]},
    )
    # Independent, deliberately small all-row calculation of the scores.
    draws = model.ensembles(data.inputs, 3, 42)
    mean = draws.mean(axis=1)
    low, high = np.quantile(draws, [0.05, 0.95], axis=1)
    pairwise = np.abs(draws[:, :, None] - draws[:, None, :]).mean(axis=(1, 2))
    crps = np.abs(draws - data.outputs[:, None]).mean(axis=1) - pairwise / 2
    np.testing.assert_allclose(report.predictions, mean)
    np.testing.assert_allclose(report.lower, low)
    np.testing.assert_allclose(report.upper, high)
    assert report.metrics["rmse"] == pytest.approx(
        np.sqrt(np.mean((mean - truth) ** 2))
    )
    assert report.metrics["mae"] == pytest.approx(np.abs(mean - truth).mean())
    assert report.metrics["crps"] == pytest.approx(crps.mean())
    assert report.by_output["crps"].item() == pytest.approx(crps.mean())
    assert report.metrics["predictive_coverage_90"] == pytest.approx(
        ((data.outputs >= low) & (data.outputs <= high)).mean()
    )
    assert report.metrics["predictive_interval_width_90"] == pytest.approx(
        (high - low).mean()
    )
    assert report.metadata["constraints"][
        "nonnegative:count:violation_fraction"
    ] == pytest.approx((draws < 0).mean())
    assert report.metadata["constraints"]["nondecreasing:count:violation_fraction"] == 0
    assert report.by_group["n_replicates"].to_list() == [7, 2, 2]
    expected_group_mse = [
        np.mean(
            (
                mean[data.inputs[:, 0] == x].mean(axis=0)
                - data.outputs[data.inputs[:, 0] == x].mean(axis=0)
            )
            ** 2
        )
        for x in (0, 1, 2)
    ]
    assert report.metrics["replicate_mean_rmse"] == pytest.approx(
        np.sqrt(np.mean(expected_group_mse))
    )
    assert [len(call["inputs"]) for call in model.calls] == [2, 1]
    assert report.metadata["n_unique_contexts"] == 3
    assert report.metadata["n_generated_ensembles"] == 3
    assert report.metadata["n_generated_trajectories"] == 9
    assert report.metadata["n_runs"] == 11


def test_reuse_requires_identical_complete_conditioning_not_group_identity():
    data = make_bank(
        [[0], [0], [0], [1], [0]],
        [
            [[0, 1], [1, 2]],
            [[8, 1], [9, 3]],
            [[2, 1], [3, 2]],
            [[4, 1], [5, 2]],
            [[6, 1], [7, 2]],
        ],
        conditioned=True,
        group_keys=[("same",)] * 5,
    )
    model = RecordingGenerator(data, conditioned=True)
    report = evaluate(model, data, n_samples=3, reuse_samples=True, batch_size=1)
    assert len(model.calls) == 3
    assert [call["conditioning"][0, :, 0].tolist() for call in model.calls] == [
        [1, 2],
        [1, 3],
        [1, 2],
    ]
    expected = model.ensembles(data.inputs, 3, 42, data.outputs[:, :, 1:])
    np.testing.assert_array_equal(report.predictions, expected.mean(axis=1))
    assert report.metadata["n_unique_contexts"] == 3
    # Shared group identity alone does not warrant a replicate-mean diagnostic.
    assert report.by_group["replicate_mean_rmse"].item() is None


def test_hash_collision_cannot_merge_different_conditioning_paths(monkeypatch):
    class CollidingHash:
        def digest(self):
            return b"collision"

    monkeypatch.setattr(
        "metis.evaluation.hashlib",
        SimpleNamespace(sha256=lambda *args: CollidingHash()),
    )
    data = make_bank(
        [[0], [0], [0]],
        [[[0, 1], [1, 2]], [[8, 1], [9, 3]], [[2, 1], [3, 2]]],
        conditioned=True,
    )
    model = RecordingGenerator(data, conditioned=True)
    report = evaluate(model, data, n_samples=3, reuse_samples=True)
    assert report.metadata["n_unique_contexts"] == 2
    np.testing.assert_array_equal(report.predictions[:, 0, 0], [4, 5, 4])


def test_exact_matching_includes_signed_zero_but_not_nearby_inputs():
    data = make_bank(
        [[0], [-0.0], [0.00001]],
        [[[0, -0.0]], [[2, 0]], [[5, 0]]],
        conditioned=True,
    )
    report = evaluate(
        RecordingGenerator(data, conditioned=True),
        data,
        n_samples=3,
        reuse_samples=True,
    )
    assert report.metadata["n_unique_contexts"] == 2


def test_default_preserves_run_batches_and_seeds_and_reuse_is_repeatable():
    data = make_bank([[0], [1], [0], [2], [1]], np.zeros((5, 2, 1)))
    default_model = RecordingGenerator(data, random=True)
    default = evaluate(default_model, data, n_samples=3, batch_size=2, seed=25)
    disabled = evaluate(
        RecordingGenerator(data, random=True),
        data,
        n_samples=3,
        batch_size=2,
        seed=25,
        reuse_samples=False,
    )
    seeds = [25] + [
        int(np.random.SeedSequence([25, start]).generate_state(1, dtype=np.uint64)[0])
        % 2**63
        for start in (2, 4)
    ]
    assert [call["seed"] for call in default_model.calls] == seeds
    assert [len(call["inputs"]) for call in default_model.calls] == [2, 2, 1]
    assert default.metadata["n_generated_ensembles"] == 5
    assert default.metadata["n_unique_contexts"] is None
    np.testing.assert_array_equal(default.predictions, disabled.predictions)
    expected = np.concatenate(
        [
            default_model.ensembles(data.inputs[start : start + 2], 3, seed).mean(
                axis=1
            )
            for start, seed in zip((0, 2, 4), seeds)
        ]
    )
    np.testing.assert_array_equal(default.predictions, expected)
    reused = [
        evaluate(
            RecordingGenerator(data, random=True),
            data,
            n_samples=3,
            batch_size=2,
            seed=25,
            reuse_samples=True,
        )
        for _ in range(2)
    ]
    np.testing.assert_array_equal(reused[0].predictions, reused[1].predictions)
    np.testing.assert_array_equal(reused[0].predictions[0], reused[0].predictions[2])
    assert not np.array_equal(default.predictions[0], default.predictions[2])
    assert reused[0].metadata["n_generated_ensembles"] == 3
    assert "first-appearance" in reused[0].metadata["sampling_seed_scheme"]


def test_many_replicates_do_not_expand_the_sample_tensor(monkeypatch):
    # Capture each subtraction immediately before its absolute-error reduction.
    # Even a context with 1001 replicates must never create an all-replicate
    # (runs, samples, time, output) error tensor.
    data = make_bank([[0]] * 1001, np.zeros((1001, 2, 1)))
    original_abs = np.abs
    ensemble_shapes = []

    def record_abs(values, *args, **kwargs):
        if np.ndim(values) == 4:
            ensemble_shapes.append(np.shape(values))
        return original_abs(values, *args, **kwargs)

    monkeypatch.setattr("metis.evaluation.np.abs", record_abs)
    model = RecordingGenerator(data)
    report = evaluate(model, data, n_samples=3, batch_size=7, reuse_samples=True)
    assert len(model.calls) == 1 and len(model.calls[0]["inputs"]) == 1
    assert max(shape[0] for shape in ensemble_shapes) <= 7
    assert sum(shape[0] for shape in ensemble_shapes) == 1001
    assert report.metadata["n_generated_trajectories"] == 3


@pytest.mark.parametrize("value", [1, 0, None, "true", np.bool_(True)])
def test_reuse_option_requires_a_boolean(value):
    data = make_bank([[0]], [[[0]]])
    model = RecordingGenerator(data)
    with pytest.raises(ValueError, match="reuse_samples must be a boolean"):
        evaluate(model, data, reuse_samples=value)
    assert model.calls == []


def test_deterministic_predictions_explicitly_ignore_reuse():
    data = make_bank([[0], [0], [1]], [[[0]], [[2]], [[4]]])
    model = RecordingGenerator(data)
    model.model_name = "mlp"
    model.predict = lambda inputs: inputs[:, :1, None]
    default = evaluate(model, data, batch_size=2)
    reused = evaluate(model, data, batch_size=2, reuse_samples=True)
    assert default.metrics == reused.metrics
    np.testing.assert_array_equal(default.predictions, reused.predictions)
    assert "ignored" in reused.metadata["sample_reuse_policy"]
    assert reused.metadata["n_generated_ensembles"] is None
