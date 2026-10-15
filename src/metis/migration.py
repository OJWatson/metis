"""Explicit, inference-only import of historical METIS checkpoints.

The source trainer did not save its fitted normalization. A checkpoint alone
cannot be converted faithfully. Supply exact statistics or explicitly identify
the complete cohort used to calculate them; METIS never guesses a split, repairs
old zero scales, unpickles arbitrary objects, or invents training history.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from metis.data import SimulationDataset, Standardizer
from metis.surrogate import Surrogate, TrainingConfig


def _array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    with np.errstate(over="ignore"):
        array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} exceeds float32 range")
    return array


def _plain(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("Legacy configuration keys must be strings")
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(
        "Legacy metadata must contain only finite plain values and tensors"
    )


@dataclass(frozen=True)
class LegacyPreprocessing:
    """Explicit schema, exact float32 scaling and known training ranges.

    ``provenance`` should identify how the original statistics/cohorts were
    established. It is a required human-readable assertion, not a verification
    that METIS can recover from an incomplete historical checkpoint.
    """

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    times: np.ndarray
    input_scaler: Standardizer
    output_scaler: Standardizer
    domain_min: np.ndarray
    domain_max: np.ndarray
    policy: str
    provenance: str
    normalization_fingerprint: str | None = None
    training_fingerprint: str | None = None
    training_group_keys: tuple[tuple, ...] = ()
    normalization_runs: int | None = None
    constant_masks_known: bool = False

    def __post_init__(self):
        if self.policy not in (
            "upstream-full-bank",
            "corrected-training-only",
            "supplied-exact-statistics",
        ):
            raise ValueError("Unknown legacy preprocessing policy")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("Explicit preprocessing provenance is required")
        names = (*self.input_names, *self.output_names)
        if (
            not self.input_names
            or not self.output_names
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Legacy input/output names must be nonempty and distinct")
        times = np.asarray(self.times).copy()
        if (
            times.ndim != 1
            or not len(times)
            or times.dtype.kind not in "biuf"
            or not np.isfinite(times).all()
            or not np.all(times[1:] > times[:-1])
        ):
            raise ValueError("Legacy times must be a finite, strictly increasing grid")
        p, t, c = len(self.input_names), len(times), len(self.output_names)
        if self.input_scaler.mean.shape != (p,) or self.output_scaler.mean.shape != (
            t,
            c,
        ):
            raise ValueError("Legacy statistics do not match schema/time dimensions")
        if (
            self.input_scaler.arithmetic_dtype != "float32"
            or self.output_scaler.arithmetic_dtype != "float32"
        ):
            raise ValueError("Historical METIS preprocessing used float32 arithmetic")
        lower, upper = (
            _array(self.domain_min, "domain_min"),
            _array(self.domain_max, "domain_max"),
        )
        if lower.shape != (p,) or upper.shape != (p,) or np.any(lower > upper):
            raise ValueError("Supply known training bounds matching the input columns")
        for name, value in (
            ("times", times),
            ("domain_min", lower.copy()),
            ("domain_max", upper.copy()),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "input_names", tuple(self.input_names))
        object.__setattr__(self, "output_names", tuple(self.output_names))

    @classmethod
    def from_statistics(
        cls,
        *,
        input_names: Sequence[str],
        output_names: Sequence[str],
        times: Any,
        input_mean: Any,
        input_std: Any,
        output_mean: Any,
        output_std: Any,
        domain_min: Any,
        domain_max: Any,
        provenance: str,
        layout: str = "channel_time",
        input_constant_mask: Any = None,
        output_constant_mask: Any = None,
    ) -> LegacyPreprocessing:
        """Use exact saved means/sample SDs, without recalculating them.

        Default ``layout='channel_time'`` accepts historical input statistics
        ``(P,T)`` and output statistics ``(C,T)``. Static input statistics must be
        identical at every time; a ``(P,)`` input vector is also accepted.
        ``layout='time_channel'`` accepts output statistics ``(T,C)`` instead.
        Masks are optional: absent masks are recorded as unknown, never inferred
        merely because a scale equals one. Zero/nonfinite scales are rejected.
        """
        x_mean, x_std = _array(input_mean, "input_mean"), _array(input_std, "input_std")
        y_mean, y_std = (
            _array(output_mean, "output_mean"),
            _array(output_std, "output_std"),
        )
        if x_mean.ndim == 2:
            if x_mean.shape != x_std.shape or x_mean.shape[1] != len(times):
                raise ValueError(
                    "Historical input statistics must match (inputs, times)"
                )
            if not np.all(x_mean == x_mean[:, :1]) or not np.all(x_std == x_std[:, :1]):
                raise ValueError(
                    "Historical input statistics vary over time; static import is unsupported"
                )
            x_mean, x_std = x_mean[:, 0], x_std[:, 0]
        if layout == "channel_time":
            y_mean, y_std = y_mean.T, y_std.T
            if output_constant_mask is not None:
                output_constant_mask = np.asarray(output_constant_mask).T
        elif layout != "time_channel":
            raise ValueError("layout must be channel_time or time_channel")
        masks_known = (
            input_constant_mask is not None and output_constant_mask is not None
        )
        x_mask = (
            np.zeros_like(x_mean, dtype=bool)
            if input_constant_mask is None
            else np.asarray(input_constant_mask, dtype=bool)
        )
        if x_mask.ndim == 2:
            if not np.all(x_mask == x_mask[:, :1]):
                raise ValueError("Historical input constant mask varies over time")
            x_mask = x_mask[:, 0]
        y_mask = (
            np.zeros_like(y_mean, dtype=bool)
            if output_constant_mask is None
            else np.asarray(output_constant_mask, dtype=bool)
        )
        return cls(
            tuple(input_names),
            tuple(output_names),
            np.asarray(times),
            Standardizer(x_mean, x_std, x_mask, arithmetic_dtype="float32"),
            Standardizer(y_mean, y_std, y_mask, arithmetic_dtype="float32"),
            _array(domain_min, "domain_min"),
            _array(domain_max, "domain_max"),
            "supplied-exact-statistics",
            provenance,
            constant_masks_known=masks_known,
        )

    @classmethod
    def from_dataset(
        cls,
        normalization_data: SimulationDataset,
        *,
        policy: str,
        provenance: str,
        training_data: SimulationDataset | None = None,
    ) -> LegacyPreprocessing:
        """Reproduce historical float32/sample-SD normalization explicitly.

        For ``upstream-full-bank``, pass the exact full normalization bank and
        the exact training cohort separately. For ``corrected-training-only``,
        normalization_data itself is the exact training cohort. Group splits
        are never inferred. Rows are put in the source loader's replicate-first,
        group-second order. Reject zero upstream scales instead of silently
        changing a broken historical preprocessing rule.
        """
        if not isinstance(normalization_data, SimulationDataset):
            raise TypeError("normalization_data must be a validated SimulationDataset")
        if policy == "corrected-training-only":
            if (
                training_data is not None
                and training_data.fingerprint != normalization_data.fingerprint
            ):
                raise ValueError(
                    "corrected-training-only requires the exact training normalization cohort"
                )
            training_data = normalization_data
        elif policy == "upstream-full-bank":
            if not isinstance(training_data, SimulationDataset):
                raise ValueError(
                    "upstream-full-bank requires explicit training_data; its split cannot be recovered from the checkpoint"
                )
        else:
            raise ValueError(
                "Choose upstream-full-bank or corrected-training-only explicitly"
            )
        assert training_data is not None
        if (
            training_data.input_names != normalization_data.input_names
            or training_data.output_names != normalization_data.output_names
            or not np.array_equal(training_data.times, normalization_data.times)
        ):
            raise ValueError(
                "Training and normalization cohorts must have the same schema and grid"
            )
        # Establish true membership including arrays, not merely a coincidentally
        # repeated group label in an unrelated bank.
        bank_rows = {
            (key, rep): i
            for i, (key, rep) in enumerate(
                zip(normalization_data.group_keys, normalization_data.replicate_ids)
            )
        }
        for i, identity in enumerate(
            zip(training_data.group_keys, training_data.replicate_ids)
        ):
            j = bank_rows.get(identity)
            if (
                j is None
                or not np.array_equal(
                    training_data.inputs[i], normalization_data.inputs[j]
                )
                or not np.array_equal(
                    training_data.outputs[i], normalization_data.outputs[j]
                )
            ):
                raise ValueError(
                    "Every training realization must occur unchanged in the normalization bank"
                )
        try:
            order = sorted(
                range(len(normalization_data)),
                key=lambda i: (
                    normalization_data.replicate_ids[i],
                    normalization_data.group_keys[i],
                ),
            )
        except TypeError as exc:
            raise ValueError(
                "Legacy identities cannot be sorted consistently; supply exact statistics instead"
            ) from exc
        n, t = len(normalization_data), len(normalization_data.times)
        p, c = len(normalization_data.input_names), len(normalization_data.output_names)
        # Polars' historical to_numpy() exported column-major (N*T, channels)
        # arrays. Reproduce those strides before the original view/permute:
        # changing reduction layout can change float32 means at large offsets.
        x_rows = np.asfortranarray(
            np.repeat(normalization_data.inputs[order, None, :], t, axis=1).reshape(
                n * t, p
            )
        )
        y_rows = np.asfortranarray(normalization_data.outputs[order].reshape(n * t, c))
        x = torch.from_numpy(x_rows).view(n, t, p).permute(0, 2, 1)
        y = torch.from_numpy(y_rows).view(n, t, c).permute(0, 2, 1)
        if policy == "upstream-full-bank" and len(x) < 2:
            raise ValueError(
                "Upstream sample SD is undefined for one realization; supply valid exact statistics"
            )
        correction = 1 if len(x) > 1 else 0
        x_mean, y_mean = x.mean(0), y.mean(0)
        x_std, y_std = x.std(0, correction=correction), y.std(0, correction=correction)
        x_mask, y_mask = x_std == 0, y_std == 0
        if policy == "upstream-full-bank" and (
            bool(x_mask.any()) or bool(y_mask.any())
        ):
            raise ValueError(
                "Upstream normalization has zero scales; import cannot silently replace them with unit scales"
            )
        if policy == "corrected-training-only":
            x_std = torch.where(x_mask, torch.ones_like(x_std), x_std)
            y_std = torch.where(y_mask, torch.ones_like(y_std), y_std)
        return cls(
            normalization_data.input_names,
            normalization_data.output_names,
            normalization_data.times,
            Standardizer(
                x_mean[:, 0].numpy(),
                x_std[:, 0].numpy(),
                x_mask[:, 0].numpy(),
                arithmetic_dtype="float32",
            ),
            Standardizer(
                y_mean.T.numpy(),
                y_std.T.numpy(),
                y_mask.T.numpy(),
                arithmetic_dtype="float32",
            ),
            training_data.inputs.min(axis=0),
            training_data.inputs.max(axis=0),
            policy,
            provenance,
            normalization_data.fingerprint,
            training_data.fingerprint,
            training_data.group_keys,
            len(normalization_data),
            True,
        )


_TARGETS = {
    f"{prefix}.{module}.{factory}": family
    for prefix in ("models", "metis.models")
    for module, factory, family in (
        ("mlp", "create_residual_mlp", "mlp"),
        ("transformer", "create_autoregressive_transformer", "transformer"),
        ("sit", "create_interpolant_transformer", "flow_matching"),
        ("itarflow", "create_itarflow", "itarflow"),
    )
}


def _architecture(config: dict, preprocessing: LegacyPreprocessing) -> tuple[str, dict]:
    from metis.models.mlp import ResidualMLPConfig
    from metis.models.transformer import TransformerConfig
    from metis.models.sit import SiTConfig
    from metis.models.itarflow import iTarFlowConfig

    dataset = config.get("dataset")
    raw = config.get("model")
    if not isinstance(dataset, dict) or not isinstance(raw, dict):
        raise ValueError(
            "Legacy import requires resolved model and dataset configuration"
        )
    family = _TARGETS.get(raw.get("_target_"))
    if family is None:
        raise ValueError(
            "Unsupported legacy model target; only allowlisted static METIS families can be imported"
        )
    if dataset.get("observed_output_variables"):
        raise ValueError(
            "Observed-sequence conditioning import is unsupported; it requires separate verified migration"
        )
    if (
        tuple(dataset.get("surrogate_input_variables", ())) != preprocessing.input_names
        or tuple(dataset.get("output_variables", ())) != preprocessing.output_names
        or dataset.get("num_timesteps") != len(preprocessing.times)
    ):
        raise ValueError(
            "Explicit preprocessing schema does not match the saved resolved dataset configuration"
        )
    architecture = raw.get("sit") if family == "flow_matching" else raw
    if not isinstance(architecture, dict):
        raise ValueError(
            "Legacy flow-matching config must contain a resolved sit mapping"
        )
    if architecture.get("cross_attn") is not None:
        raise ValueError(
            "Cross-attention checkpoint import is unsupported; conditioning cannot be inferred"
        )
    p, t, c = (
        len(preprocessing.input_names),
        len(preprocessing.times),
        len(preprocessing.output_names),
    )
    dimensions = {
        "mlp": {"input_dim": p, "output_dim": t * c},
        "transformer": {"static_input_dim": p, "x_input_dim": c, "sequence_len": t + 1},
        "flow_matching": {"cond_dim": p, "data_dim": c, "sequence_len": t},
        "itarflow": {"static_in": p, "out_channels": c, "num_timesteps": t},
    }[family]
    for name, expected in dimensions.items():
        if architecture.get(name) != expected:
            raise ValueError(
                f"Saved architecture {name} does not match the explicit schema ({expected})"
            )
    config_type = {
        "mlp": ResidualMLPConfig,
        "transformer": TransformerConfig,
        "flow_matching": SiTConfig,
        "itarflow": iTarFlowConfig,
    }[family]
    required = (
        {field.name for field in fields(config_type)}
        - {"_target_", "cross_attn"}
        - dimensions.keys()
    )
    if required - architecture.keys():
        raise ValueError(
            f"Saved architecture omits options {sorted(required - architecture.keys())}; import refuses to guess historical defaults"
        )
    allowed = required | dimensions.keys() | {"_target_", "cross_attn"}
    if architecture.keys() - allowed:
        raise ValueError(
            f"Unrecognized legacy architecture options: {sorted(architecture.keys() - allowed)}"
        )
    options = {name: architecture[name] for name in required}
    if family == "flow_matching":
        if raw.keys() - {
            "_target_",
            "interpolant",
            "sit",
            "num_steps",
            "solver",
            "multi_mode",
        }:
            raise ValueError("Unrecognized legacy flow-matching wrapper options")
        if raw.get("interpolant") not in ("flow-matching", "linear", "trig"):
            raise ValueError(
                "Saved interpolant must explicitly name a supported historical interpolant"
            )
        options["interpolant"] = raw["interpolant"]
    return family, options


def import_legacy_checkpoint(
    path: str | Path,
    *,
    preprocessing: LegacyPreprocessing,
    output_path: str | Path | None = None,
    device: str = "cpu",
    inference_batch_size: int = 64,
) -> Surrogate:
    """Import an allowlisted historical checkpoint for inference only.

    Unsafe pickled configuration objects are rejected by the restricted loader;
    there is no ``weights_only=False`` fallback. Source weights are float32,
    architecture dimensions are verified, and state keys load strictly. The
    original file is never overwritten. This does not resume an optimizer or
    infer missing preprocessing, validation identities, loss steps, or best step.

    The returned object's TrainingConfig is labelled as an inference-only batch
    adapter, not a reconstruction of the old warmup/cosine training schedule.
    Old sampling defaults remain in legacy_import.config; specify solver/steps
    explicitly when reproducing an old flow-matching generation recipe. The
    canonical Heun solver corrects every step, including the final step; old
    trainer samples using its uncorrected final step are not bitwise reproduced.
    """
    if not isinstance(preprocessing, LegacyPreprocessing):
        raise TypeError(
            "Supply LegacyPreprocessing with explicit schema, statistics and provenance"
        )
    source = Path(path)
    if output_path is not None and Path(output_path).resolve() == source.resolve():
        raise ValueError(
            "Choose a new output path; preserve the original legacy checkpoint"
        )
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            "Legacy checkpoint could not be read with the restricted weights-only loader. "
            "No unsafe-pickle fallback is attempted; export tensor/plain metadata from a trusted original environment."
        ) from exc
    if not isinstance(checkpoint, dict) or "format" in checkpoint:
        raise ValueError(
            "Expected a historical checkpoint; use Surrogate.load for native artifacts"
        )
    if not isinstance(checkpoint.get("config"), dict) or not isinstance(
        checkpoint.get("model"), Mapping
    ):
        raise ValueError(
            "Historical checkpoint must contain plain resolved config and model state"
        )
    config = _plain(checkpoint["config"])
    family, options = _architecture(config, preprocessing)
    state = checkpoint["model"]
    if not state or any(
        not isinstance(k, str) or not isinstance(v, torch.Tensor)
        for k, v in state.items()
    ):
        raise ValueError("Legacy model state must be named tensors")
    for name, tensor in state.items():
        if not bool(torch.isfinite(tensor).all()) or (
            tensor.is_floating_point() and tensor.dtype != torch.float32
        ):
            raise ValueError(
                f"Legacy state {name!r} must be finite float32; other numerical modes need a verified converter"
            )
    # Reuse the package's allowlisted device validation and artifact construction;
    # no dynamic target from legacy metadata is imported or instantiated.
    from metis.surrogate import _device, _validate_weights

    target_device = _device(device)
    model = Surrogate(family, options)
    model.input_names = preprocessing.input_names
    model.output_names = preprocessing.output_names
    model.times = np.array(preprocessing.times, copy=True)
    model.input_scaler = preprocessing.input_scaler
    model.output_scaler = preprocessing.output_scaler
    model.domain_min = preprocessing.domain_min.copy()
    model.domain_max = preprocessing.domain_max.copy()
    model.training_bounds = {"min": model.domain_min, "max": model.domain_max}
    model.training_config = TrainingConfig(
        steps=1, batch_size=inference_batch_size, device=str(target_device)
    )
    model.device = target_device
    devices = [target_device.index] if target_device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        model.model = model._create_model().to(target_device)
        try:
            _validate_weights(model.model, state)
            model.model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "Legacy weights do not match the saved architecture; import cannot reinterpret or drop keys"
            ) from exc
    model.model.eval()
    model.history = []
    model.training_metadata = {
        "config": None,
        "training_fingerprint": preprocessing.training_fingerprint,
        "validation_fingerprint": None,
        "train_group_keys": _plain(preprocessing.training_group_keys),
        "validation_group_keys": [],
        "completed_steps": None,
        "best_step": None,
        "best_validation_loss": None,
        "normalization": "Explicit legacy float32 arithmetic and sample-SD statistics; see legacy_import.preprocessing",
        "legacy_import": {
            "migration_version": 1,
            "scope": "inference only; optimizer resume and historical loss chronology not reconstructed",
            "source_checkpoint": str(source.resolve()),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_format": "legacy train.py plain config/model checkpoint; no embedded scalers",
            "config": config,
            "saved_step": _plain(checkpoint.get("step")),
            "saved_train_losses": _plain(checkpoint.get("train_losses")),
            "saved_val_losses": _plain(checkpoint.get("val_losses")),
            "saved_val_steps": _plain(checkpoint.get("val_steps")),
            "runtime_config_role": "TrainingConfig fields are an inference-only adapter; not historical fit settings",
            "sampling_numerics": "Canonical solver: Heun corrects every step, including the final step. Historical uncorrected-final-step Heun samples are not reproduced bitwise; architecture, weights and preprocessing are preserved.",
            "preprocessing": {
                "policy": preprocessing.policy,
                "provenance": preprocessing.provenance,
                "arithmetic_dtype": "float32",
                "standard_deviation": "sample SD (correction=1), except corrected single-run unit scales; exact supplied values otherwise",
                "normalization_fingerprint": preprocessing.normalization_fingerprint,
                "normalization_runs": preprocessing.normalization_runs,
                "constant_masks_known": preprocessing.constant_masks_known,
                "training_bounds": "explicitly supplied or calculated from explicit training_data; never inferred from mean/SD",
            },
        },
    }
    model._fitted = True
    if output_path is not None:
        model.save(output_path)
    return model
