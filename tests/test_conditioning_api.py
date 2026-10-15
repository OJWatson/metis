"""Observed-output conditioning uses saved units, exact channels and timing."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from metis import SimulationDataset, Surrogate, TrainingConfig
from metis.data import Standardizer


@pytest.fixture
def conditional_bank():
    rng = np.random.default_rng(14)
    x = np.arange(12, dtype=np.float32)[:, None]
    observed = rng.normal(100, 3, (12, 4))
    targets = 2 * observed + x
    return SimulationDataset.from_arrays(
        x,
        np.stack([targets, observed], -1),
        input_names=["rate"],
        output_names=["target", "observed"],
        times=[0, 1, 3, 7],
    ).split()


def fit_conditional(bank, family):
    options = (
        {"n_layer": 1, "n_embd": 16}
        if family == "transformer"
        else {"n_blocks": 1, "embd_dim": 16}
    )
    return Surrogate(family, options, observed_outputs=["observed"]).fit(
        bank.train, bank.validation, TrainingConfig(steps=2, batch_size=4, eval_every=1)
    )


@pytest.mark.parametrize("family", ["transformer", "flow_matching"])
def test_conditioning_roundtrip_units_evaluation(conditional_bank, family, tmp_path):
    bank = conditional_bank
    model = fit_conditional(bank, family)
    x = bank.test.inputs
    conditions = bank.test.outputs[:, :, 1:]
    kwargs = {"n_samples": 3, "num_steps": 2, "seed": 14} if model.is_stochastic else {}
    predicted = model.predict(x, conditioning=conditions, **kwargs)
    assert predicted.shape == (len(x), 4, 1)
    restored = Surrogate.load(model.save(tmp_path / "conditional.pt"))
    np.testing.assert_array_equal(
        predicted,
        restored.predict(x, conditioning={"observed": conditions[:, :, 0]}, **kwargs),
    )
    report = model.evaluate(bank.test, n_samples=3, num_steps=2)
    assert report.by_output["output"].to_list() == ["target"]
    np.testing.assert_array_equal(report.observed, bank.test.outputs[:, :, :1])
    assert report.metadata["observed_outputs"] == ["observed"]
    assert (
        report.metadata["model_fingerprint"]
        == restored.evaluate(bank.test, n_samples=3, num_steps=2).metadata[
            "model_fingerprint"
        ]
    )
    with pytest.raises(ValueError, match="conditioning is required"):
        model.predict(x, **kwargs)
    with pytest.raises(ValueError, match="shape"):
        model.predict(x, conditioning=conditions[:, :-1], **kwargs)
    with pytest.raises(ValueError, match="exactly"):
        model.predict(x, conditioning={"typo": conditions[:, :, 0]}, **kwargs)
    # The scaler is fitted on training observations, not held-out trajectories.
    np.testing.assert_allclose(
        model.conditioning_scaler.mean, bank.train.outputs[:, :, 1:].mean(0), rtol=1e-6
    )


def test_transformer_conditioning_has_no_future_leakage(conditional_bank):
    model = fit_conditional(conditional_bank, "transformer")
    x = conditional_bank.train.inputs[:2]
    conditioning = conditional_bank.train.outputs[:2, :, 1:].copy()
    before = model.predict(x, conditioning=conditioning)
    conditioning[:, 2:] += 300
    after = model.predict(x, conditioning=conditioning)
    np.testing.assert_array_equal(before[:, :3], after[:, :3])


@pytest.mark.parametrize("family", ["mlp", "itarflow"])
def test_unsupported_conditioning_rejected(family):
    with pytest.raises(ValueError, match="conditioning requires"):
        Surrogate(family, observed_outputs=["obs"])


def test_old_static_artifact_still_loads(conditional_bank, tmp_path):
    model = Surrogate("mlp", {"hidden_dim": 8}).fit(
        conditional_bank.train, config=TrainingConfig(steps=1)
    )
    path = model.save(tmp_path / "old.pt")
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 1
    for key in ("observed_outputs", "training_output_names", "conditioning_scaler"):
        payload.pop(key)
    for key in ("input_scaler", "output_scaler"):
        payload[key].pop("arithmetic_dtype")
    payload["training_config"] = {
        k: payload["training_config"][k]
        for k in (
            "steps",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "seed",
            "device",
            "eval_every",
            "patience",
        )
    }
    torch.save(payload, path)
    restored = Surrogate.load(path)
    np.testing.assert_array_equal(
        model.predict(conditional_bank.train.inputs),
        restored.predict(conditional_bank.train.inputs),
    )


@pytest.mark.parametrize("family", ["transformer", "flow_matching"])
def test_multiple_reordered_observations_preserve_scaling_and_evaluation(
    family, tmp_path
):
    x = np.arange(12, dtype=np.float32)[:, None]
    grid = np.arange(4, dtype=np.float32)[None]
    outputs = np.stack(
        [100 + x + grid, 10 + 2 * x + grid, 1000 + 3 * x - grid, -50 + x - 2 * grid],
        axis=-1,
    )
    bank = SimulationDataset.from_arrays(
        x,
        outputs,
        input_names=["rate"],
        output_names=[
            "first_observed",
            "first_predicted",
            "last_observed",
            "last_predicted",
        ],
    ).split()
    options = (
        {"n_layer": 1, "n_embd": 16}
        if family == "transformer"
        else {"n_blocks": 1, "embd_dim": 16}
    )
    # Declare observed channels in the reverse of their physical bank order.
    model = Surrogate(
        family, options, observed_outputs=["last_observed", "first_observed"]
    ).fit(
        bank.train, bank.validation, TrainingConfig(steps=2, batch_size=4, eval_every=1)
    )
    assert model.output_names == ("first_predicted", "last_predicted")
    conditions = bank.test.outputs[:, :, [2, 0]]
    full_scaler = Standardizer.fit(bank.train.outputs)
    np.testing.assert_array_equal(
        model.conditioning_scaler.transform(conditions),
        full_scaler.transform(bank.test.outputs)[:, :, [2, 0]],
    )
    np.testing.assert_array_equal(model.output_scaler.mean, full_scaler.mean[:, [1, 3]])
    sampling = (
        {"n_samples": 3, "seed": 44, "num_steps": 2} if model.is_stochastic else {}
    )
    expected = model.predict(bank.test.inputs, conditioning=conditions, **sampling)
    # Mapping insertion order is irrelevant; declared observed channel order wins.
    mapping = {
        "first_observed": conditions[:, :, 1],
        "last_observed": conditions[:, :, 0],
    }
    np.testing.assert_array_equal(
        expected, model.predict(bank.test.inputs, conditioning=mapping, **sampling)
    )
    restored = Surrogate.load(model.save(tmp_path / "multiple.pt"))
    np.testing.assert_array_equal(
        expected, restored.predict(bank.test.inputs, conditioning=mapping, **sampling)
    )
    report = model.evaluate(bank.test, n_samples=3, seed=44, num_steps=2)
    np.testing.assert_allclose(report.predictions, expected, rtol=1e-6)
    np.testing.assert_array_equal(report.observed, bank.test.outputs[:, :, [1, 3]])
    assert report.by_output["output"].to_list() == list(model.output_names)
    assert report.metadata["observed_outputs"] == ["last_observed", "first_observed"]


@pytest.mark.parametrize("family", ["transformer", "flow_matching"])
def test_public_conditioned_resume_matches_uninterrupted(
    conditional_bank, family, tmp_path
):
    bank = conditional_bank
    options = (
        {"n_layer": 1, "n_embd": 16}
        if family == "transformer"
        else {"n_blocks": 1, "embd_dim": 16}
    )

    def model():
        return Surrogate(family, options, observed_outputs=["observed"])

    config = TrainingConfig(
        steps=4, batch_size=4, eval_every=1, checkpoint_every=1, checkpoint_dir=tmp_path
    )
    uninterrupted = model().fit(bank.train, bank.validation, config)

    def interrupt(record):
        if record["step"] == 2:
            raise RuntimeError("controlled interruption")

    with pytest.raises(RuntimeError, match="controlled interruption"):
        model().fit(bank.train, bank.validation, config, on_progress=interrupt)
    resumed = model().fit(
        bank.train, bank.validation, replace(config, resume_from=tmp_path / "last.pt")
    )
    assert resumed.history == uninterrupted.history
    for name, value in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(
            resumed.model.state_dict()[name], value, rtol=0, atol=0
        )


def test_resume_rejects_changed_raw_bank_even_when_standardized_tensors_match(tmp_path):
    x = np.arange(8, dtype=np.float32)[:, None]
    grid = np.arange(3, dtype=np.float32)[None]
    train = SimulationDataset.from_arrays(
        x,
        np.stack([2 * x + grid, 4 * x - grid], axis=-1),
        input_names=["rate"],
        output_names=["target", "observed"],
    )
    changed = replace(
        train, outputs=train.outputs + np.array([0, 100], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        Standardizer.fit(train.outputs).transform(train.outputs),
        Standardizer.fit(changed.outputs).transform(changed.outputs),
    )
    config = TrainingConfig(
        steps=2, batch_size=4, checkpoint_every=1, checkpoint_dir=tmp_path
    )
    model = Surrogate(
        "transformer", {"n_layer": 1, "n_embd": 16}, observed_outputs=["observed"]
    )
    model.fit(train, config=config)
    with pytest.raises(ValueError, match="does not match"):
        model.fit(changed, config=replace(config, resume_from=tmp_path / "last.pt"))
