"""The installed Hydra frontend composes and runs the public package API."""

import json
import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_module
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import polars as pl
import pytest


def configured(*overrides):
    with initialize_config_module(version_base="1.3", config_module="metis.conf"):
        return compose(config_name="config", overrides=list(overrides))


def command(tmp_path, *arguments, module=False):
    environment = dict(os.environ)
    # Installed-package resolution must work without a checkout on PYTHONPATH.
    environment.pop("PYTHONPATH", None)
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLBACKEND="Agg")
    executable = (
        [sys.executable, "-m", "metis.hydra_app"]
        if module
        else [
            str(
                Path(sys.executable).with_name(
                    "metis-train.exe" if os.name == "nt" else "metis-train"
                )
            )
        ]
    )
    result = subprocess.run(
        [*executable, *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return result


@pytest.mark.parametrize("family", ["mlp", "transformer", "flow_matching", "itarflow"])
def test_packaged_model_groups_instantiate_the_public_api(family):
    from metis import Surrogate, TrainingConfig

    config = configured(f"model={family}")
    assert config.dataset._target_ == "metis.load_example"
    assert config.split.validation_fraction == config.split.test_fraction == 0.2
    assert config.evaluation.partition == "test"
    assert not config.wandb.enabled and not config.plot.enabled
    model = instantiate(config.model, _convert_="all")
    assert isinstance(model, Surrogate)
    assert model.model_name == family
    assert isinstance(instantiate(config.training), TrainingConfig)
    # Dimensions derive from data, not from user-maintained YAML values.
    assert "sequence_len" not in config.model.model_options
    assert "input_dim" not in config.model.model_options


def test_parquet_template_requires_schema_and_path():
    config = configured("dataset=parquet")
    assert OmegaConf.is_missing(config.dataset, "path")
    assert OmegaConf.is_missing(config.dataset, "inputs")
    assert OmegaConf.is_missing(config.dataset, "outputs")
    assert config.dataset.groups is None


def test_installed_config_can_be_printed_outside_checkout(tmp_path):
    result = command(tmp_path, "--cfg", "job", module=True)
    config = OmegaConf.create(result.stdout)
    assert config.dataset._target_ == "metis.load_example"
    assert config.training.device == "cpu"


def test_installed_hydra_training_produces_reloadable_model(tmp_path):
    from metis import Surrogate, load_example

    output = tmp_path / "experiment"
    command(
        tmp_path,
        "training.steps=2",
        "training.batch_size=4",
        "training.eval_every=1",
        "model.model_options.hidden_dim=8",
        f"hydra.run.dir={output}",
    )
    for filename in (
        "model.pt",
        "metrics.json",
        "history.json",
        "config.yaml",
        ".hydra/config.yaml",
    ):
        assert (output / filename).exists()
    surrogate = Surrogate.load(output / "model.pt")
    data = load_example()
    prediction = surrogate.predict(data.inputs[:2])
    assert prediction.shape == data.outputs[:2].shape
    assert np.isfinite(prediction).all()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["metadata"]["evaluation_partition"] == "test"
    assert metrics["metadata"]["training_group_overlap"] == 0
    assert metrics["metadata"]["tuning_group_overlap"] == 0
    history = json.loads((output / "history.json").read_text())
    assert [row["step"] for row in history] == [1, 2]


def test_hydra_multirun_saves_independent_experiment_artifacts(tmp_path):
    sweep = tmp_path / "sweep"
    result = command(
        tmp_path,
        "--multirun",
        "training.steps=1",
        "training.batch_size=4",
        "training.learning_rate=0.001,0.0003",
        "model.model_options.hidden_dim=8",
        f"hydra.sweep.dir={sweep}",
    )
    assert "sweep evaluates the test partition" in result.stderr
    for index, rate in enumerate((0.001, 0.0003)):
        folder = sweep / str(index)
        assert (folder / "model.pt").exists()
        assert (folder / "metrics.json").exists()
        config = OmegaConf.load(folder / "config.yaml")
        assert config.training.learning_rate == rate


def test_parquet_paths_resolve_from_launch_directory_when_hydra_changes_cwd(tmp_path):
    table = pl.DataFrame(
        [
            {"theta": float(i), "rep": 0, "t": t, "y": i + t * 0.1}
            for i in range(5)
            for t in range(3)
        ]
    )
    table.write_parquet(tmp_path / "bank.parquet")
    output = tmp_path / "parquet_run"
    command(
        tmp_path,
        "dataset=parquet",
        "dataset.path=bank.parquet",
        "dataset.inputs=[theta]",
        "dataset.outputs=[y]",
        "training.steps=1",
        "training.batch_size=2",
        "model.model_options.hidden_dim=8",
        "hydra.job.chdir=true",
        f"hydra.run.dir={output}",
    )
    report = json.loads((output / "metrics.json").read_text())
    assert report["metadata"]["n_runs"] == 1
    assert report["by_output"][0]["output"] == "y"


def test_absent_evaluation_partition_is_an_error_not_a_training_fallback(tmp_path):
    from metis.hydra_app import run

    config = configured("split.test_fraction=0", "training.steps=1")
    with pytest.raises(ValueError, match="No test partition exists"):
        run(config, output_dir=tmp_path)
    assert not (tmp_path / "model.pt").exists()


def test_real_offline_wandb_training_needs_no_credentials(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    output = tmp_path / "offline_tracking"
    command(
        tmp_path,
        "training.steps=2",
        "training.eval_every=1",
        "training.batch_size=4",
        "model.model_options.hidden_dim=8",
        "wandb.enabled=true",
        "wandb.mode=offline",
        "wandb.project=metis-offline-test",
        f"hydra.run.dir={output}",
    )
    assert (output / "model.pt").exists()
    assert (output / "metrics.json").exists()
    assert list((output / "wandb").glob("offline-run-*/run-*.wandb"))


def test_conditioned_hydra_checkpoints_resume_after_directory_change(tmp_path):
    from metis import Surrogate

    frame = pl.DataFrame(
        [
            {"theta": float(i), "rep": 0, "t": t, "target": i + t, "obs": i - t * 0.1}
            for i in range(8)
            for t in range(3)
        ]
    )
    frame.write_parquet(tmp_path / "bank.parquet")
    options = (
        "dataset=parquet",
        "dataset.path=bank.parquet",
        "dataset.inputs=[theta]",
        "dataset.outputs=[target,obs]",
        "model=transformer",
        "model.observed_outputs=[obs]",
        "model.model_options.n_embd=16",
        "model.model_options.n_layer=1",
        "training.steps=2",
        "training.batch_size=4",
        "training.eval_every=1",
        "training.checkpoint_dir=checkpoints",
        "training.checkpoint_every=1",
        "training.schedule=warmup_cosine",
        "hydra.job.chdir=true",
    )
    command(tmp_path, *options, "hydra.run.dir=first")
    assert (tmp_path / "first/checkpoints/last.pt").exists()
    command(
        tmp_path,
        *options,
        "training.resume_from=first/checkpoints/last.pt",
        "hydra.run.dir=resumed",
    )
    first = Surrogate.load(tmp_path / "first/model.pt")
    resumed = Surrogate.load(tmp_path / "resumed/model.pt")
    assert first.history == resumed.history
    assert first.observed_outputs == ("obs",)
    conditions = np.ones((1, 3, 1))
    np.testing.assert_array_equal(
        first.predict([[3.0]], conditioning=conditions),
        resumed.predict([[3.0]], conditioning=conditions),
    )
