"""Acceptance tests of the portable static-input API, including original units."""

import math

import numpy as np
import polars as pl
import pytest
import torch

from metis.data import SimulationDataset
from metis.surrogate import Surrogate, TrainingConfig


OPTIONS = {
    "mlp": {"hidden_dim": 16, "num_blocks": 1, "num_layers_per_block": 2},
    "transformer": {"n_embd": 16, "n_layer": 1, "n_head": 2, "n_kv_head": 2},
    "flow_matching": {"embd_dim": 16, "n_blocks": 1, "n_head": 2, "n_kv_head": 2},
    "itarflow": {
        "channels": 16,
        "num_layers": 1,
        "num_blocks": 2,
        "num_heads": 2,
        "num_static_tokens": 2,
    },
}


@pytest.fixture
def bank():
    # Three replicates per scenario: deliberate stochastic variation, original
    # units far from zero, and a constant output at the initial timepoint.
    rng = np.random.default_rng(7)
    inputs = np.repeat(
        np.array([[i, i**2 + 1] for i in range(12)], dtype=np.float32), 3, axis=0
    )
    times = np.array([0.0, 0.5, 2.0, 5.0])
    outputs = np.empty((len(inputs), 4, 2), dtype=np.float32)
    outputs[:, :, 0] = 100 + inputs[:, :1] * times[None]
    outputs[:, :, 1] = 50 + inputs[:, 1:] * 0.1 + rng.normal(0, 0.2, (len(inputs), 4))
    return SimulationDataset.from_arrays(
        inputs, outputs, input_names=["a", "b"], output_names=["y", "z"], times=times
    )


def make_fit(bank, family="mlp", steps=3, validation=False):
    train, val = bank.subset(np.arange(27)), bank.subset(np.arange(27, 36))
    model = Surrogate(family, OPTIONS[family])
    model.fit(
        train,
        val if validation else None,
        TrainingConfig(steps=steps, batch_size=8, eval_every=1),
    )
    return model, train, val


@pytest.mark.parametrize("family", Surrogate.FAMILIES)
def test_train_infer_safe_artifact_roundtrip_all_families(
    bank, family, tmp_path, monkeypatch
):
    surrogate, train, val = make_fit(bank, family, validation=True)
    np.testing.assert_allclose(
        surrogate.input_scaler.mean, train.inputs.mean(axis=0), rtol=1e-6
    )
    assert surrogate.output_scaler.constant_mask[0, 0]
    assert surrogate.training_metadata["best_step"] is not None
    assert surrogate.training_metadata["training_fingerprint"] == train.fingerprint
    assert len(surrogate.history) == 3
    inputs = train.inputs[:2]
    if surrogate.is_stochastic:
        with pytest.raises(ValueError, match="stochastic"):
            surrogate.predict(inputs)
        output = surrogate.sample(inputs, n_samples=3, seed=19, num_steps=2)
        assert output.shape == (2, 3, 4, 2)
        np.testing.assert_array_equal(
            output, surrogate.sample(inputs, n_samples=3, seed=19, num_steps=2)
        )
        np.testing.assert_allclose(
            surrogate.predict(inputs, n_samples=3, seed=19, num_steps=2),
            output.mean(axis=1),
        )
    else:
        output = surrogate.predict(inputs)
        assert output.shape == (2, 4, 2)
        with pytest.raises(ValueError, match="deterministic"):
            surrogate.sample(inputs)
    assert np.isfinite(output).all()
    assert output.mean() > 25  # In original units, not normalized values.
    path = surrogate.save(tmp_path / f"{family}.pt")
    payload = torch.load(path, weights_only=True)
    assert payload["format_version"] == Surrogate.ARTIFACT_VERSION
    assert "model_config" in payload and "input_scaler" in payload
    monkeypatch.chdir(tmp_path)  # No cwd/repository/data-file dependency.
    restored = Surrogate.load(path)
    actual = (
        restored.sample(inputs, n_samples=3, seed=19, num_steps=2)
        if restored.is_stochastic
        else restored.predict(inputs)
    )
    np.testing.assert_array_equal(output, actual)
    assert restored.input_names == ("a", "b")
    assert restored.output_names == ("y", "z")
    assert restored.history == surrogate.history
    np.testing.assert_array_equal(restored.times, bank.times)
    assert list(tmp_path.glob("*.tmp")) == []


def test_input_schema_forms_domain_and_invalid_values(bank):
    surrogate, train, _ = make_fit(bank)
    rows = [{"b": float(row[1]), "a": float(row[0])} for row in train.inputs[:2]]
    expected = surrogate.predict(train.inputs[:2])
    np.testing.assert_allclose(surrogate.predict(rows), expected)
    np.testing.assert_allclose(surrogate.predict(pl.DataFrame(rows)), expected)
    np.testing.assert_allclose(
        surrogate.predict({"b": train.inputs[:2, 1], "a": train.inputs[:2, 0]}),
        expected,
    )
    np.testing.assert_allclose(surrogate.predict(rows[0]), expected[:1])
    for bad in ([], [[1]], [[float("nan"), 1]], {"a": 1}, {"a": 1, "b": 2, "extra": 3}):
        with pytest.raises(ValueError):
            surrogate.predict(bad)
    with pytest.warns(UserWarning, match="outside the training ranges"):
        assert surrogate.predict({"a": 100, "b": 1}).shape == (1, 4, 2)


def test_pandas_input_names_are_validated_and_reordered(bank):
    pd = pytest.importorskip("pandas")
    surrogate, train, _ = make_fit(bank)
    frame = pd.DataFrame({"b": train.inputs[:2, 1], "a": train.inputs[:2, 0]})
    expected = surrogate.predict(train.inputs[:2])
    np.testing.assert_array_equal(surrogate.predict(frame), expected)
    with pytest.raises(ValueError, match="Input columns must be exactly"):
        surrogate.predict(frame.rename(columns={"b": "wrong_name"}))
    with pytest.raises(ValueError, match="Input columns must be exactly"):
        surrogate.predict(pd.DataFrame([[1.0, 2.0]], columns=["a", "a"]))


def test_validation_groups_and_schema_must_match(bank):
    model = Surrogate("mlp", OPTIONS["mlp"])
    with pytest.raises(ValueError, match="overlap"):
        model.fit(bank, bank, TrainingConfig(steps=1))
    with pytest.raises(ValueError, match="patience"):
        model.fit(bank, config=TrainingConfig(steps=1, patience=1))


def test_fit_load_and_sample_preserve_callers_torch_rng(bank, tmp_path):
    torch.manual_seed(135)
    before = torch.random.get_rng_state().clone()
    surrogate, train, _ = make_fit(bank, "flow_matching")
    assert torch.equal(before, torch.random.get_rng_state())
    surrogate.sample(train.inputs[:2], n_samples=2, num_steps=2)
    assert torch.equal(before, torch.random.get_rng_state())
    restored = Surrogate.load(surrogate.save(tmp_path / "rng.pt"))
    assert torch.equal(before, torch.random.get_rng_state())
    assert restored.model_name == "flow_matching"


def test_fit_reproducibility(bank):
    first, train, _ = make_fit(bank, "flow_matching", validation=True)
    second, _, _ = make_fit(bank, "flow_matching", validation=True)
    assert first.history == second.history
    np.testing.assert_array_equal(
        first.sample(train.inputs[:1], n_samples=2, num_steps=2),
        second.sample(train.inputs[:1], n_samples=2, num_steps=2),
    )


def test_best_validation_state_restored_and_patience(bank, monkeypatch):
    train, val = bank.subset(np.arange(27)), bank.subset(np.arange(27, 36))
    surrogate = Surrogate("mlp", OPTIONS["mlp"])
    states = []

    def controlled_validation(tensors, config):
        states.append(
            {k: v.detach().clone() for k, v in surrogate.model.state_dict().items()}
        )
        return [1.0, 2.0, 3.0][len(states) - 1]

    monkeypatch.setattr(surrogate, "_validation_loss", controlled_validation)
    surrogate.fit(
        train, val, TrainingConfig(steps=10, batch_size=8, eval_every=1, patience=2)
    )
    assert surrogate.training_metadata["completed_steps"] == 3
    assert surrogate.training_metadata["best_step"] == 1
    assert surrogate.training_metadata["stopped_early"]
    for key, value in surrogate.model.state_dict().items():
        torch.testing.assert_close(value, states[0][key])


def test_progress_callback_step_order_validation_and_defensive_copy(bank):
    train, validation = bank.subset(np.arange(27)), bank.subset(np.arange(27, 36))
    events = []

    def record(event):
        events.append(dict(event))
        event.clear()
        event["train_loss"] = "mutated by callback"

    surrogate = Surrogate("mlp", OPTIONS["mlp"]).fit(
        train,
        validation,
        TrainingConfig(steps=3, batch_size=8, eval_every=2),
        on_progress=record,
    )
    assert [event["step"] for event in events] == [1, 2, 3]
    assert "validation_loss" not in events[0]
    assert "validation_loss" in events[1] and "validation_loss" in events[2]
    assert surrogate.history == events


def test_progress_callback_errors_propagate(bank):
    def broken(event):
        raise RuntimeError("logging service failed")

    with pytest.raises(RuntimeError, match="logging service failed"):
        Surrogate("mlp", OPTIONS["mlp"]).fit(
            bank,
            config=TrainingConfig(steps=1),
            on_progress=broken,
        )


def test_nonfinite_loss_fails_actionably(bank, monkeypatch):
    from metis.models.mlp import ResidualMLP

    def broken_loss(self, *args, **kwargs):
        return next(self.parameters()).sum() * float("nan")

    monkeypatch.setattr(ResidualMLP, "compute_loss", broken_loss)
    with pytest.raises(ValueError, match="Training loss at step 1 is nonfinite"):
        make_fit(bank)


def test_itarflow_density_full_constants_jacobian_and_gradients(bank):
    surrogate, train, _ = make_fit(bank, "itarflow")
    inputs = torch.tensor(train.inputs[:2].copy(), requires_grad=True)
    outputs = torch.tensor(train.outputs[:2].copy(), requires_grad=True)
    actual = surrogate.log_prob_tensor(inputs, outputs)
    normalized_x = torch.tensor(surrogate.input_scaler.transform(train.inputs[:2]))
    normalized_y = torch.tensor(surrogate.output_scaler.transform(train.outputs[:2]))
    z, logdet = surrogate.model(normalized_y, normalized_x)
    d = outputs.shape[1] * outputs.shape[2]
    expected = (
        -0.5 * z.double().square().flatten(1).sum(1)
        - 0.5 * d * math.log(2 * math.pi)
        + d * logdet.double()
    )
    expected = expected - np.log(surrogate.output_scaler.scale).sum()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        actual.detach().numpy(),
        surrogate.log_prob(train.inputs[:2], train.outputs[:2]),
        rtol=1e-6,
    )
    actual.sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert outputs.grad is not None and torch.isfinite(outputs.grad).all()
    assert torch.count_nonzero(outputs.grad) > 0


def test_itarflow_density_normalizes_large_offsets_with_exact_saved_statistics():
    values = np.float32(1e8) + 8 * np.arange(8, dtype=np.float32)
    bank = SimulationDataset.from_arrays(
        values[:, None],
        np.repeat(values[:, None, None], 2, axis=1),
        input_names=["parameter"],
        output_names=["trajectory"],
    )
    surrogate = Surrogate("itarflow", OPTIONS["itarflow"]).fit(
        bank,
        config=TrainingConfig(steps=1, batch_size=8),
    )
    assert surrogate.output_scaler.mean[0, 0] == 100000028
    inputs = torch.tensor(bank.inputs.copy(), requires_grad=True)
    outputs = torch.tensor(bank.outputs.copy(), requires_grad=True)
    actual = surrogate.log_prob_tensor(inputs, outputs)
    x = torch.tensor(surrogate.input_scaler.transform(bank.inputs))
    y = torch.tensor(surrogate.output_scaler.transform(bank.outputs))
    z, mean_logdet = surrogate.model(y, x)
    dimensions = bank.outputs.shape[1] * bank.outputs.shape[2]
    expected = (
        -0.5 * (z.double().square() + math.log(2 * math.pi)).flatten(1).sum(1)
        + dimensions * mean_logdet.double()
        - np.log(surrogate.output_scaler.scale).sum()
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    actual.sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert outputs.grad is not None and torch.isfinite(outputs.grad).all()
    assert torch.count_nonzero(outputs.grad) > 0


def test_itarflow_identity_density_scales_with_original_units(bank):
    surrogate, train, _ = make_fit(bank, "itarflow")
    # Zeroing final projections gives an identity normalizing flow; test an
    # independently known Gaussian density, not only the implementation formula.
    with torch.no_grad():
        for block in surrogate.model.blocks:
            block.final_layer.proj_out.weight.zero_()
            block.final_layer.proj_out.bias.zero_()
    y = surrogate.output_scaler.mean[None] + 0.5 * surrogate.output_scaler.scale[None]
    logp = surrogate.log_prob(train.inputs[:1], y)
    d = y.shape[1] * y.shape[2]
    expected = (
        -0.5 * d * (0.25 + math.log(2 * math.pi))
        - np.log(surrogate.output_scaler.scale).sum()
    )
    np.testing.assert_allclose(logp, expected, atol=1e-4)


@pytest.mark.parametrize(
    "model, options",
    [
        ("invalid", {}),
        ("mlp", {"input_dim": 3}),
        ("transformer", {"cross_attn": {}}),
        ("mlp", {"_target_": "os.system"}),
        ("flow_matching", {"unknown": 2}),
    ],
)
def test_invalid_model_options(model, options):
    with pytest.raises(ValueError):
        Surrogate(model, options)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"batch_size": True},
        {"learning_rate": float("nan")},
        {"learning_rate": 0},
        {"weight_decay": -1},
        {"seed": -1},
        {"patience": 0},
    ],
)
def test_invalid_training_configuration(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_unfitted_and_corrupt_artifact_errors(bank, tmp_path):
    with pytest.raises(RuntimeError, match="Fit"):
        Surrogate().predict([[1, 2]])
    surrogate, _, _ = make_fit(bank)
    path = surrogate.save(tmp_path / "model.pt")
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 999
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Unsupported"):
        Surrogate.load(path)
    payload["format_version"] = 1
    payload["model_name"] = "os.system"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Unknown model"):
        Surrogate.load(path)


def test_artifact_versions_are_recorded_without_restricting_portable_load(
    bank, tmp_path
):
    surrogate, train, _ = make_fit(bank)
    path = surrogate.save(tmp_path / "portable.pt")
    payload = torch.load(path, weights_only=True)
    assert set(payload["producer"]) == {"metis", "python", "pytorch", "numpy"}
    assert payload["training_metadata"]["environment"]["fit_dtype"] == "float32"
    assert payload["training_metadata"]["environment"]["fit_device"] == "cpu"
    # Version information is provenance, not a strict relocation requirement.
    payload["producer"]["python"] = "different compatible interpreter"
    torch.save(payload, path)
    np.testing.assert_array_equal(
        Surrogate.load(path).predict(train.inputs[:2]),
        surrogate.predict(train.inputs[:2]),
    )
    # Earlier format-1 bundles did not record these optional metadata fields.
    payload.pop("producer")
    payload["training_metadata"].pop("environment")
    torch.save(payload, path)
    np.testing.assert_array_equal(
        Surrogate.load(path).predict(train.inputs[:2]),
        surrogate.predict(train.inputs[:2]),
    )
