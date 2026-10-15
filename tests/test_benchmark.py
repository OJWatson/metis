import csv
import json

import numpy as np
import pytest

from benchmarks.compare_models import MeanTrajectoryBaseline, main, run_benchmark
from metis import SimulationDataset, Surrogate, evaluate


def tiny_bank():
    parameters = np.repeat(np.arange(10, dtype=np.float32), 2)[:, None]
    outputs = (
        parameters[:, None] * np.arange(1, 5, dtype=np.float32)[None, :, None] + 10
    )
    return SimulationDataset.from_arrays(
        parameters, outputs, input_names=["p"], output_names=["y"]
    )


def test_benchmark_public_api_all_families_common_test_and_artifacts(tmp_path):
    options = {
        "mlp": {"hidden_dim": 8, "num_blocks": 1, "num_layers_per_block": 2},
        "transformer": {"n_embd": 8, "n_layer": 1, "n_head": 2, "n_kv_head": 2},
        "flow_matching": {"embd_dim": 8, "n_blocks": 1, "n_head": 2, "n_kv_head": 2},
        "itarflow": {
            "channels": 8,
            "num_layers": 1,
            "num_blocks": 2,
            "num_heads": 2,
            "num_static_tokens": 2,
        },
    }
    destination = tmp_path / "pilot"
    result = run_benchmark(
        destination,
        data=tiny_bank(),
        steps=1,
        n_samples=2,
        num_steps=2,
        batch_size=4,
        evaluation_batch_size=2,
        model_options=options,
    )
    assert result["manifest"]["failed_models"] == []
    assert len(result["summary"]) == 5
    fingerprint = result["manifest"]["split"]["test"]["fingerprint"]
    assert {row["test_fingerprint"] for row in result["summary"]} == {fingerprint}
    with (destination / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert "validation_loss" not in rows[0]
    for name in options:
        run = json.loads((destination / name / "run.json").read_text())
        metrics = json.loads((destination / name / "metrics.json").read_text())
        assert run["parameter_count"] > 0
        assert metrics["metadata"]["training_group_overlap"] == 0
        assert metrics["metadata"]["tuning_group_overlap"] == 0
        assert (
            run["training_metadata"]["training_fingerprint"]
            == result["manifest"]["split"]["train"]["fingerprint"]
        )
        assert "test_trajectories.npz" in {
            p.name for p in (destination / name).iterdir()
        }
        restored = Surrogate.load(destination / name / "model.pt")
        assert restored.model_name == name
        assert run["model_options_resolved"] == restored.resolved_model_options
    assert (destination / "train_mean_baseline" / "model.json").exists()
    with pytest.raises(FileExistsError):
        run_benchmark(destination, data=tiny_bank(), steps=1, models=["mlp"])


def test_mean_baseline_uses_training_only():
    data = tiny_bank()
    split = data.split(seed=3)
    baseline = MeanTrajectoryBaseline(split.train)
    expected = split.train.outputs.mean(axis=0, dtype=np.float64)
    np.testing.assert_array_equal(baseline.predict(split.test.inputs)[0], expected)
    report = evaluate(baseline, split.test)
    assert report.metrics["rmse"] >= 0
    assert "predictive_coverage_90" not in report.metrics


def test_benchmark_records_family_failure_without_hiding_it(tmp_path):
    result = run_benchmark(
        tmp_path,
        data=tiny_bank(),
        steps=1,
        models=["mlp"],
        model_options={"mlp": {"hidden_dim": 0}},
    )
    assert result["manifest"]["failed_models"] == ["mlp"]
    assert result["summary"][0]["status"] == "ok"
    assert result["summary"][1]["status"] == "failed"
    assert (tmp_path / "mlp" / "failure.json").exists()


def test_benchmark_cli_help():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
