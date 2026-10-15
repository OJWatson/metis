"""Synthetic upstream-format checkpoints and independently calculated parity."""

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from metis.data import SimulationDataset, Standardizer
from metis.migration import LegacyPreprocessing, import_legacy_checkpoint
from metis.surrogate import Surrogate


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
def statistics():
    # Frozen independent fixture, in legacy channel-first statistics layout.
    # Sample SD sqrt(2) follows from two values [1, 3]; scale 1 alone must not
    # be interpreted as evidence a column was constant in the original fit.
    return LegacyPreprocessing.from_statistics(
        input_names=["a", "b"],
        output_names=["y", "z"],
        times=[0, 1, 2, 3],
        input_mean=[[2] * 4, [10] * 4],
        input_std=[[math.sqrt(2)] * 4, [1] * 4],
        output_mean=[[10, 20, 30, 40], [100, 200, 300, 400]],
        output_std=[[2, 3, 4, 5], [10, 20, 30, 40]],
        domain_min=[0, 8],
        domain_max=[4, 12],
        provenance="Frozen synthetic upstream-compatible statistics; no user checkpoint",
    )


def checkpoint(tmp_path, family, statistics):
    source = Surrogate(family, OPTIONS[family])
    source.input_names, source.output_names = (
        statistics.input_names,
        statistics.output_names,
    )
    source.times = statistics.times
    with torch.random.fork_rng():
        torch.manual_seed(11)
        raw = source._create_model().eval()
    model_config = copy.deepcopy(source._resolved_model_config)
    if family == "flow_matching":
        model_config = {
            "_target_": "models.sit.create_interpolant_transformer",
            "interpolant": "linear",
            "num_steps": 10,
            "solver": "heun",
            "multi_mode": None,
            "sit": model_config,
        }
    else:
        model_config["_target_"] = model_config["_target_"].replace(
            "metis.models.", "models."
        )
    payload = {
        "config": {
            "model": model_config,
            "dataset": {
                "surrogate_input_variables": list(statistics.input_names),
                "output_variables": list(statistics.output_names),
                "num_timesteps": 4,
                "observed_output_variables": [],
            },
            "training": {"warmup_steps": 10, "num_steps": 100},
        },
        "model": raw.state_dict(),
        "step": 40,
        "train_losses": [2.0, 1.0],
        "val_losses": [3.0],
    }
    path = tmp_path / f"{family}-old.pt"
    torch.save(payload, path)
    return path, raw, payload


def torch_normalize(array, mean, scale):
    return (
        torch.as_tensor(np.asarray(array).copy()).float()
        - torch.tensor(mean, dtype=torch.float32)
    ) / torch.tensor(scale, dtype=torch.float32)


def torch_denormalize(array, statistics):
    return (
        array * torch.tensor(statistics.output_scaler.scale, dtype=torch.float32)
        + torch.tensor(statistics.output_scaler.mean, dtype=torch.float32)
    ).numpy()


@pytest.mark.parametrize("family", Surrogate.FAMILIES)
def test_import_four_families_preserves_raw_outputs_and_safe_artifact(
    tmp_path, statistics, family
):
    path, raw, payload = checkpoint(tmp_path, family, statistics)
    rng = torch.get_rng_state().clone()
    imported = import_legacy_checkpoint(
        path, preprocessing=statistics, output_path=tmp_path / "native.pt"
    )
    assert torch.equal(rng, torch.get_rng_state())
    restored = Surrogate.load(tmp_path / "native.pt")
    assert restored.input_scaler.arithmetic_dtype == "float32"
    assert restored.output_scaler.arithmetic_dtype == "float32"
    assert imported.history == restored.history == []
    assert imported.training_metadata["best_step"] is None
    assert imported.training_metadata["config"] is None
    from metis.evaluation import _model_provenance

    provenance = _model_provenance(restored)
    assert provenance["legacy_import"] == restored.training_metadata["legacy_import"]
    assert provenance["training_summary"]["completed_steps"] is None
    assert provenance["training_config"] is None
    origin = restored.training_metadata["legacy_import"]
    assert origin["saved_step"] == 40
    assert origin["saved_val_steps"] is None  # Do not invent old step chronology.
    assert origin["config"] == payload["config"]
    assert not origin["preprocessing"]["constant_masks_known"]
    assert len(origin["source_sha256"]) == 64
    for name, value in raw.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])
    inputs = np.array([[1.1, 9.5], [3.2, 11.1]], dtype=np.float32)
    x = torch_normalize(
        inputs, statistics.input_scaler.mean, statistics.input_scaler.scale
    )
    with torch.no_grad():
        if family == "mlp":
            expected = torch_denormalize(raw(x).reshape(2, 4, 2), statistics)
            np.testing.assert_array_equal(restored.predict(inputs), expected)
        elif family == "transformer":
            expected = torch_denormalize(
                torch.cat(
                    list(raw.generate(x, max_output_steps=4, use_kv_cache=False)), dim=1
                ),
                statistics,
            )
            np.testing.assert_array_equal(restored.predict(inputs), expected)
        elif family == "flow_matching":
            latent = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) / 16
            clock = torch.tensor([0.2, 0.6])
            torch.testing.assert_close(
                restored.model.v_t(latent, clock, x),
                raw.v_t(latent, clock, x),
                rtol=0,
                atol=0,
            )
            expected, _ = raw.sample(latent, x, num_steps=3, method="euler")
            actual, _ = restored.model.sample(latent, x, num_steps=3, method="euler")
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        else:
            original_y = np.arange(16, dtype=np.float32).reshape(2, 4, 2) + 20
            normalized_y = torch_normalize(
                original_y,
                statistics.output_scaler.mean,
                statistics.output_scaler.scale,
            )
            z, mean_logdet = raw(normalized_y, x)
            expected = -0.5 * (z.double().square() + math.log(2 * math.pi)).flatten(
                1
            ).sum(1)
            expected += 8 * mean_logdet.double()
            expected -= (
                torch.tensor(statistics.output_scaler.scale, dtype=torch.float32)
                .log()
                .sum()
            )
            np.testing.assert_allclose(
                restored.log_prob(inputs, original_y), expected.numpy(), rtol=0, atol=0
            )
            latent = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) / 16
            expected_y, _ = raw.reverse(latent, x)
            actual_y, _ = restored.model.reverse(latent, x)
            torch.testing.assert_close(expected_y, actual_y, rtol=0, atol=0)
        if restored.is_stochastic:
            np.testing.assert_array_equal(
                imported.sample(inputs, 3, seed=9, num_steps=2),
                restored.sample(inputs, 3, seed=9, num_steps=2),
            )


def cohort():
    # Group-first storage, replicate-first original loader order: 0,2,1,3.
    return SimulationDataset.from_arrays(
        [[1], [3], [5], [7]],
        np.array([[[1], [10]], [[3], [30]], [[5], [50]], [[7], [70]]]),
        input_names=["x"],
        output_names=["y"],
        group_keys=[("a",), ("a",), ("b",), ("b",)],
        replicate_ids=[0, 1, 0, 1],
        times=[0, 2],
    )


def test_explicit_full_bank_and_corrected_training_only_sample_sd():
    bank = cohort()
    train = bank.subset([0, 1])
    old = LegacyPreprocessing.from_dataset(
        bank,
        policy="upstream-full-bank",
        training_data=train,
        provenance="Known original bank and training membership",
    )
    corrected = LegacyPreprocessing.from_dataset(
        train,
        policy="corrected-training-only",
        provenance="Known corrected trainer cohort",
    )
    assert old.input_scaler.mean.item() == 4
    assert old.input_scaler.scale.item() == np.float32(math.sqrt(20 / 3))
    assert corrected.input_scaler.mean.item() == 2
    assert corrected.input_scaler.scale.item() == np.float32(math.sqrt(2))
    assert (
        corrected.input_scaler.scale.item()
        != Standardizer.fit(train.inputs).scale.item()
    )
    np.testing.assert_array_equal(old.domain_min, [1])
    np.testing.assert_array_equal(old.domain_max, [3])
    assert old.normalization_fingerprint == bank.fingerprint
    assert old.training_fingerprint == train.fingerprint
    expected = torch_normalize(
        bank.inputs, old.input_scaler.mean, old.input_scaler.scale
    ).numpy()
    np.testing.assert_array_equal(old.input_scaler.transform(bank.inputs), expected)


def test_missing_cohort_and_wrong_membership_are_rejected():
    bank = cohort()
    with pytest.raises(ValueError, match="explicit training_data"):
        LegacyPreprocessing.from_dataset(
            bank, policy="upstream-full-bank", provenance="Known"
        )
    changed = SimulationDataset.from_arrays(
        bank.inputs,
        bank.outputs + 1,
        input_names=bank.input_names,
        output_names=bank.output_names,
        group_keys=bank.group_keys,
        replicate_ids=bank.replicate_ids,
        times=bank.times,
    )
    with pytest.raises(ValueError, match="unchanged"):
        LegacyPreprocessing.from_dataset(
            bank, policy="upstream-full-bank", training_data=changed, provenance="Known"
        )
    with pytest.raises(ValueError, match="exact training"):
        LegacyPreprocessing.from_dataset(
            bank,
            policy="corrected-training-only",
            training_data=bank.subset([0]),
            provenance="Known",
        )


def test_upstream_zero_scales_rejected_corrected_singleton_uses_unit_scale():
    single = cohort().subset([0])
    with pytest.raises(ValueError, match="undefined"):
        LegacyPreprocessing.from_dataset(
            single,
            policy="upstream-full-bank",
            training_data=single,
            provenance="Known",
        )
    corrected = LegacyPreprocessing.from_dataset(
        single, policy="corrected-training-only", provenance="Known"
    )
    np.testing.assert_array_equal(corrected.input_scaler.scale, 1)
    np.testing.assert_array_equal(corrected.output_scaler.scale, 1)
    assert corrected.input_scaler.constant_mask.all()
    np.testing.assert_array_equal(corrected.output_scaler.transform(single.outputs), 0)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda p: p["config"]["dataset"].update(observed_output_variables=["z"]),
            "conditioning",
        ),
        (lambda p: p["config"]["model"].update(_target_="os.system"), "Unsupported"),
        (lambda p: p["config"]["model"].pop("hidden_dim"), "omits options"),
        (lambda p: p["config"]["model"].update(input_dim=99), "does not match"),
        (lambda p: p["model"].pop(next(iter(p["model"]))), "weights do not match"),
    ],
)
def test_unverifiable_checkpoint_is_rejected(tmp_path, statistics, mutation, message):
    path, _, payload = checkpoint(tmp_path, "mlp", statistics)
    mutation(payload)
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        import_legacy_checkpoint(path, preprocessing=statistics)


class UnsafeObject:
    def __reduce__(self):
        return (Path.touch, (Path("should-never-exist"),))


def test_unsafe_pickle_has_no_automatic_fallback(tmp_path, statistics, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "unsafe.pt"
    torch.save({"config": UnsafeObject()}, path)
    with pytest.raises(ValueError, match="No unsafe-pickle fallback"):
        import_legacy_checkpoint(path, preprocessing=statistics)
    assert not (tmp_path / "should-never-exist").exists()


def test_original_checkpoint_never_overwritten(tmp_path, statistics):
    path, _, _ = checkpoint(tmp_path, "mlp", statistics)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="preserve the original"):
        import_legacy_checkpoint(path, preprocessing=statistics, output_path=path)
    assert path.read_bytes() == original


def test_exact_statistics_require_provenance_and_valid_scales():
    kwargs = dict(
        input_names=["x"],
        output_names=["y"],
        times=[0, 1],
        input_mean=[0],
        input_std=[1],
        output_mean=[[0, 1]],
        output_std=[[1, 1]],
        domain_min=[0],
        domain_max=[1],
        provenance="Known",
    )
    with pytest.raises(ValueError, match="provenance"):
        LegacyPreprocessing.from_statistics(**(kwargs | {"provenance": ""}))
    with pytest.raises(ValueError, match="positive scales"):
        LegacyPreprocessing.from_statistics(**(kwargs | {"input_std": [0]}))
    with pytest.raises(ValueError, match="vary over time"):
        LegacyPreprocessing.from_statistics(
            **(kwargs | {"input_mean": [[0, 1]], "input_std": [[1, 1]]})
        )


def test_cohort_statistics_match_historical_polars_export_reduction_layout():
    import polars as pl

    # Larger offsets expose reduction-order differences hidden by small integers.
    rng = np.random.default_rng(423)
    n, p, t, c = 100, 9, 20, 6
    inputs = rng.normal(1e6, 100, (n, p)).astype(np.float32)
    outputs = rng.normal(1e5, 100, (n, t, c)).astype(np.float32)
    bank = SimulationDataset.from_arrays(
        inputs,
        outputs,
        input_names=[f"x{i}" for i in range(p)],
        output_names=[f"y{i}" for i in range(c)],
        group_keys=[(i,) for i in range(n)],
        replicate_ids=[0] * n,
    )
    reconstructed = LegacyPreprocessing.from_dataset(
        bank,
        policy="upstream-full-bank",
        training_data=bank.subset(range(10)),
        provenance="Synthetic historical exporter regression",
    )
    # Independently replay the original loader's Polars->NumPy->tensor path.
    x_rows = pl.DataFrame(
        np.repeat(inputs[:, None], t, axis=1).reshape(n * t, p)
    ).to_numpy()
    y_rows = pl.DataFrame(outputs.reshape(n * t, c)).to_numpy()
    x = torch.from_numpy(x_rows).view(n, t, p).permute(0, 2, 1)
    y = torch.from_numpy(y_rows).view(n, t, c).permute(0, 2, 1)
    np.testing.assert_array_equal(
        reconstructed.input_scaler.mean, x.mean(0).numpy()[:, 0]
    )
    np.testing.assert_array_equal(reconstructed.output_scaler.mean, y.mean(0).numpy().T)
    np.testing.assert_array_equal(
        reconstructed.input_scaler.scale, x.std(0).numpy()[:, 0]
    )
    np.testing.assert_array_equal(reconstructed.output_scaler.scale, y.std(0).numpy().T)


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_legacy_import_rejects_silent_weight_dtype_casts(tmp_path, statistics, dtype):
    path, _, payload = checkpoint(tmp_path, "mlp", statistics)
    key = next(iter(payload["model"]))
    payload["model"][key] = payload["model"][key].to(dtype)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="dtype"):
        import_legacy_checkpoint(path, preprocessing=statistics)
