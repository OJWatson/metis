"""Acceptance checks for the single installed training engine."""

from copy import deepcopy
from dataclasses import replace
import os

import pytest
import torch

from metis.models.mlp import ResidualMLP, ResidualMLPConfig
from metis.models.sit import InterpolantTransformer, LinearInterpolant, SiTConfig
from metis.training import TrainingConfig, run_training


def _model(family="mlp"):
    with torch.random.fork_rng():
        torch.manual_seed(31)
        if family == "flow_matching":
            return InterpolantTransformer(
                SiTConfig(
                    sequence_len=3,
                    data_dim=1,
                    cond_dim=2,
                    n_blocks=1,
                    n_head=2,
                    n_kv_head=2,
                    embd_dim=8,
                ),
                LinearInterpolant(),
            )
        return ResidualMLP(
            ResidualMLPConfig(
                input_dim=2,
                output_dim=3,
                num_blocks=1,
                hidden_dim=8,
                num_layers_per_block=2,
            )
        )


@pytest.fixture
def tensors():
    generator = torch.Generator().manual_seed(44)
    inputs = torch.randn(13, 2, generator=generator)
    outputs = inputs[:, :1, None] + torch.tensor([0.0, 0.5, 1.0])[None, :, None]
    return inputs, outputs


class Interrupted(RuntimeError):
    pass


@pytest.mark.parametrize("family", ["mlp", "flow_matching"])
def test_resume_matches_uninterrupted_including_stochastic_rng(
    family, tensors, tmp_path
):
    model = _model(family)
    reference, interrupted, resumed = deepcopy(model), deepcopy(model), deepcopy(model)
    config = TrainingConfig(
        steps=6,
        batch_size=4,
        eval_every=2,
        schedule="warmup_cosine",
        checkpoint_every=1,
        checkpoint_dir=str(tmp_path),
        deterministic=True,
    )
    metadata = {"scalers": {"mean": [0.0, 0.0]}, "family": family}
    validation = (tensors[0][:5], tensors[1][:5])
    expected = run_training(reference, *tensors, validation, config, metadata=metadata)

    def interrupt(record):
        if record["step"] == 3:
            raise Interrupted("simulated interruption")

    with pytest.raises(Interrupted):
        run_training(
            interrupted,
            *tensors,
            validation,
            config,
            metadata=metadata,
            on_progress=interrupt,
        )
    checkpoint = torch.load(tmp_path / "last.pt", weights_only=True)
    assert checkpoint["completed_steps"] == 3
    assert checkpoint["optimizer_state"]["state"]
    actual = run_training(
        resumed,
        *tensors,
        validation,
        replace(config, resume_from=tmp_path / "last.pt"),
        metadata=metadata,
    )
    assert actual == expected
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["best.pt", "last.pt"]


@pytest.mark.parametrize(
    "changed", ["data", "metadata", "steps", "precision", "model", "topology"]
)
def test_resume_rejects_scientific_or_runtime_changes(changed, tensors, tmp_path):
    config = TrainingConfig(
        steps=2, batch_size=4, checkpoint_every=1, checkpoint_dir=tmp_path
    )
    run_training(_model(), *tensors, None, config, metadata={"scaler": 1})
    config = replace(config, resume_from=tmp_path / "last.pt")
    model, inputs, metadata = _model(), tensors[0], {"scaler": 1}
    if changed == "data":
        inputs = inputs + 0.1
    elif changed == "metadata":
        metadata = {"scaler": 2}
    elif changed == "steps":
        config = replace(config, steps=3)
    elif changed == "precision":
        config = replace(config, precision="bf16")
    elif changed == "model":
        model = ResidualMLP(
            ResidualMLPConfig(
                input_dim=2,
                output_dim=3,
                num_blocks=2,
                hidden_dim=8,
                num_layers_per_block=2,
            )
        )
    else:
        config = replace(config, num_devices=2)
    with pytest.raises(ValueError, match="does not match"):
        run_training(model, inputs, tensors[1], None, config, metadata=metadata)


def test_resume_rejects_raw_legacy_and_incomplete_checkpoints(tensors, tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"model": _model().state_dict(), "step": 2}, path)
    with pytest.raises(ValueError, match="raw legacy"):
        run_training(
            _model(),
            *tensors,
            None,
            TrainingConfig(steps=2, resume_from=path),
            metadata={},
        )
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="safe Metis"):
        run_training(
            _model(),
            *tensors,
            None,
            TrainingConfig(steps=2, resume_from=path),
            metadata={},
        )


def test_best_selection_and_patience_restores_weights(tensors, monkeypatch):
    import metis.training as training

    captured, losses = [], iter([1.0, 2.0, 3.0])

    def evaluation(loss_module, *args):
        captured.append(deepcopy(loss_module.model.state_dict()))
        return next(losses)

    monkeypatch.setattr(training, "_evaluate", evaluation)
    model = _model()
    result = run_training(
        model,
        *tensors,
        tensors,
        TrainingConfig(steps=10, batch_size=4, eval_every=1, patience=2),
        metadata={},
    )
    assert (
        result.best_step,
        result.best_loss,
        result.completed_steps,
        result.stopped_early,
    ) == (1, 1.0, 3, True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, captured[0][name], rtol=0, atol=0)
    assert not model.training


def test_schedule_optimizer_configuration_and_rng_isolation(tensors, tmp_path):
    before = torch.get_rng_state().clone()
    flags = (
        torch.get_float32_matmul_precision(),
        torch.are_deterministic_algorithms_enabled(),
    )
    config = TrainingConfig(
        steps=6,
        batch_size=4,
        schedule="warmup_cosine",
        warmup_fraction=0.5,
        learning_rate=0.03,
        min_learning_rate=0.001,
        optimizer_betas=(0.8, 0.95),
        deterministic=True,
        checkpoint_dir=tmp_path,
        checkpoint_every=6,
    )
    result = run_training(_model(), *tensors, None, config, metadata={})
    assert [entry["learning_rate"] for entry in result.history] == pytest.approx(
        [0.01, 0.02, 0.03, 0.03, 0.0155, 0.001]
    )
    assert torch.equal(before, torch.get_rng_state())
    assert flags == (
        torch.get_float32_matmul_precision(),
        torch.are_deterministic_algorithms_enabled(),
    )
    state = torch.load(tmp_path / "last.pt", weights_only=True)
    assert state["optimizer_state"]["param_groups"][0]["betas"] == (0.8, 0.95)


def test_bf16_cpu_updates_float32_model_and_callback_is_defensive(tensors):
    model = _model()
    original = deepcopy(model.state_dict())

    def callback(record):
        record["step"] = -1

    result = run_training(
        model,
        *tensors,
        tensors,
        TrainingConfig(steps=2, batch_size=4, precision="bf16"),
        metadata={},
        on_progress=callback,
    )
    assert result.history[0]["step"] == 1
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert any(
        not torch.equal(value, original[name])
        for name, value in model.state_dict().items()
    )


def test_precision_and_rng_restore_even_when_callback_interrupts(tensors):
    previous = torch.get_float32_matmul_precision()
    random_state = torch.get_rng_state().clone()
    try:
        # 'medium' exposed a PyTorch 2.10 conflict when the precision API and
        # legacy allow_tf32 setter were mixed during restoration.
        torch.set_float32_matmul_precision("medium")

        def interrupt(_record):
            raise Interrupted()

        with pytest.raises(Interrupted):
            run_training(
                _model(),
                *tensors,
                None,
                TrainingConfig(steps=2, batch_size=4),
                metadata={},
                on_progress=interrupt,
            )
        assert torch.get_float32_matmul_precision() == "medium"
        assert torch.equal(random_state, torch.get_rng_state())
    finally:
        torch.set_float32_matmul_precision(previous)


def test_compile_executes_training_graph(tensors, monkeypatch):
    original_compile = torch.compile
    compiled_graphs = []

    def backend(graph, _inputs):
        compiled_graphs.append(graph)
        return graph.forward

    # Exercise Dynamo graph capture cheaply; production uses the default backend.
    monkeypatch.setattr(
        torch, "compile", lambda model: original_compile(model, backend=backend)
    )
    result = run_training(
        _model(),
        *tensors,
        None,
        TrainingConfig(steps=2, batch_size=4, compile_model=True),
        metadata={},
    )
    assert result.completed_steps == 2
    assert compiled_graphs


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"num_devices": 3, "batch_size": 4}, "divisible"),
        ({"compile_model": True, "deterministic": True}, "deterministic"),
        ({"checkpoint_every": 1}, "supplied together"),
        ({"precision": "float16"}, "precision"),
        ({"warmup_fraction": 1.0}, "warmup_fraction"),
        ({"optimizer_betas": (0.9, 1.0)}, "optimizer_betas"),
    ],
)
def test_invalid_config_has_actionable_error(settings, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**settings)


@pytest.mark.slow
def test_spawned_cpu_global_batch_and_uneven_validation_match_single(tensors):
    single, distributed = _model(), _model()
    config = TrainingConfig(steps=3, batch_size=4, eval_every=1, deterministic=True)
    # One validation row also exercises an empty shard on rank 1.
    validation = (tensors[0][:1], tensors[1][:1])
    expected = run_training(single, *tensors, validation, config, metadata={})
    parent = os.getpid()
    callbacks = []

    def callback(record):
        assert os.getpid() == parent
        callbacks.append(record)

    actual = run_training(
        distributed,
        *tensors,
        validation,
        replace(config, num_devices=2),
        metadata={},
        on_progress=callback,
    )
    assert actual.completed_steps == expected.completed_steps
    assert actual.best_loss == pytest.approx(expected.best_loss, rel=2e-6)
    assert callbacks == actual.history
    for name, value in single.state_dict().items():
        torch.testing.assert_close(
            distributed.state_dict()[name], value, rtol=2e-5, atol=1e-6
        )


@pytest.mark.slow
def test_spawned_stochastic_resume_preserves_each_rank_rng(tensors, tmp_path):
    config = TrainingConfig(
        steps=4,
        batch_size=4,
        num_devices=2,
        eval_every=2,
        checkpoint_every=1,
        checkpoint_dir=tmp_path,
        deterministic=True,
    )
    expected_model, resumed_model = _model("flow_matching"), _model("flow_matching")
    expected = run_training(expected_model, *tensors, tensors, config, metadata={})

    def interrupt(record):
        if record["step"] == 2:
            raise Interrupted()

    with pytest.raises(Interrupted):
        run_training(
            _model("flow_matching"),
            *tensors,
            tensors,
            config,
            metadata={},
            on_progress=interrupt,
        )
    actual = run_training(
        resumed_model,
        *tensors,
        tensors,
        replace(config, resume_from=tmp_path / "last.pt"),
        metadata={},
    )
    assert actual == expected
    for name, value in expected_model.state_dict().items():
        torch.testing.assert_close(
            resumed_model.state_dict()[name], value, rtol=0, atol=0
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices and an NCCL PyTorch build",
)
def test_spawned_cuda_nccl_training(tensors):
    result = run_training(
        _model(),
        *tensors,
        tensors,
        TrainingConfig(steps=2, batch_size=4, device="cuda:0", num_devices=2),
        metadata={},
    )
    assert result.completed_steps == 2
    assert result.best_loss is not None
