"""Optional tracking streams progress and always closes runs cleanly."""

import builtins
import sys
from types import SimpleNamespace

import pytest

from metis.tracking import WandbTracker


class RecordingRun:
    def __init__(self):
        self.logs = []
        self.exit_codes = []

    def log(self, record, **kwargs):
        self.logs.append((record, kwargs))

    def finish(self, *, exit_code):
        self.exit_codes.append(exit_code)


@pytest.fixture
def backend(monkeypatch):
    run = RecordingRun()
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    return run, calls


def test_disabled_tracking_never_imports_wandb(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "wandb" or name.startswith("wandb."):
            raise AssertionError("Disabled tracking imported W&B")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with WandbTracker(enabled=False) as tracker:
        tracker.log_training({"step": 1, "train_loss": 0.2})
        tracker.log_evaluation(SimpleNamespace(metrics={"rmse": 0.3}))


def test_training_streams_immediately_without_mutating_history(backend, tmp_path):
    run, calls = backend
    record = {"step": 1, "train_loss": 2.0, "validation_loss": 3.0}
    with WandbTracker(
        project="study",
        entity="team",
        name="mlp",
        mode="offline",
        dir=tmp_path,
        config={"seed": 42},
        tags=["test"],
    ) as tracker:
        tracker.log_training(record)
        assert run.logs == [(record, {"step": 1})]  # Present before context closes.
        assert run.logs[0][0] is not record
        record["train_loss"] = 99
        assert run.logs[0][0]["train_loss"] == 2
        tracker.log_evaluation(
            SimpleNamespace(metrics={"rmse": 4.0}), prefix="validation"
        )
        assert run.logs[-1] == ({"validation/rmse": 4.0}, {})
        assert not run.exit_codes
    assert calls[0]["project"] == "study"
    assert calls[0]["config"] == {"seed": 42}
    assert calls[0]["dir"] == str(tmp_path)
    assert run.exit_codes == [0]
    tracker.finish()
    assert run.exit_codes == [0]


def test_training_failure_finishes_run_and_preserves_original_error(backend):
    run, _ = backend
    with pytest.raises(ValueError, match="training failed"):
        with WandbTracker(mode="offline"):
            raise ValueError("training failed")
    assert run.exit_codes == [1]


def test_cleanup_failure_does_not_hide_training_failure(backend, monkeypatch):
    run, _ = backend

    def finish(**kwargs):
        raise RuntimeError("backend cleanup failed")

    monkeypatch.setattr(run, "finish", finish)
    with pytest.warns(RuntimeWarning, match="original exception is preserved"):
        with pytest.raises(ValueError, match="training failed"):
            with WandbTracker(mode="offline"):
                raise ValueError("training failed")


def test_missing_optional_dependency_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match=r"metis-surrogate\[wandb\]"):
        with WandbTracker(mode="offline"):
            pass


def test_tracker_requires_context_and_is_not_reused(backend):
    tracker = WandbTracker(mode="offline")
    with pytest.raises(RuntimeError, match="with.*block"):
        tracker.log_training({"step": 1})
    with tracker:
        pass
    with pytest.raises(RuntimeError, match="new WandbTracker"):
        with tracker:
            pass


def test_real_fit_emits_live_metrics_through_tracker(backend):
    import numpy as np
    from metis import SimulationDataset, Surrogate, TrainingConfig

    run, _ = backend
    bank = SimulationDataset.from_arrays(
        np.arange(5)[:, None],
        np.arange(10).reshape(5, 2, 1),
        input_names=["theta"],
        output_names=["output"],
    )
    split = bank.split()
    model = Surrogate("mlp", {"hidden_dim": 8, "num_blocks": 1})
    with WandbTracker(mode="offline") as tracker:

        def callback(record):
            assert not model._fitted
            tracker.log_training(record)
            assert run.logs[-1][0]["step"] == record["step"]

        model.fit(
            split.train,
            split.validation,
            TrainingConfig(steps=2, batch_size=2, eval_every=1),
            on_progress=callback,
        )
        assert [record[0]["step"] for record in run.logs] == [1, 2]
        assert all("validation_loss" in record[0] for record in run.logs)
    assert run.exit_codes == [0]
