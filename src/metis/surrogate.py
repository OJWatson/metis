"""Train, persist and use neural surrogates without managing PyTorch tensors.

The public API models fixed-length trajectories conditional on static inputs.
MLPs and autoregressive transformers predict one trajectory; flow matching and
iTARFlow represent a conditional distribution of trajectories. Neural-network
weights are point estimates, not Bayesian parameter posterior samples.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any
import warnings

import numpy as np
import torch

from metis.data import DatasetSplit, SimulationDataset, Standardizer


@dataclass(frozen=True)
class TrainingConfig:
    """Small, CPU-compatible defaults; ``steps`` counts optimizer updates.

    ``patience`` counts validation evaluations without improvement. It requires
    a validation dataset. Validation never updates normalization statistics.
    """

    steps: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.001
    seed: int = 42
    device: str = "cpu"
    eval_every: int = 50
    patience: int | None = None

    def __post_init__(self):
        for name in ("steps", "batch_size", "eval_every"):
            _positive_int(getattr(self, name), name)
        if self.patience is not None:
            _positive_int(self.patience, "patience")
        _valid_seed(self.seed)
        if not isinstance(self.device, str):
            raise ValueError("device must be a string such as 'cpu' or 'cuda:0'")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _valid_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer between 0 and 2**63 - 1")


def _device(value):
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid device {value!r}; use 'cpu' or 'cuda:0'") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This release supports device='cpu' or 'cuda[:index]'")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable. Use device='cpu' or install a compatible CUDA PyTorch build."
            )
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is unavailable")
        device = torch.device("cuda", index)
    return device


def _rng_devices(device):
    return [device.index] if device.type == "cuda" else []


def _seed_rng(seed, device):
    # Unlike torch.manual_seed, this does not change generators on other GPUs.
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.default_generators[device.index].manual_seed(seed)


def _plain(value):
    """Restrict artifact metadata to weights_only-compatible plain values."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


def _environment_versions():
    from metis import __version__

    return {
        "metis": __version__,
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "numpy": str(np.__version__),
    }


class Surrogate:
    """A static-input surrogate for a bank of fixed-length simulations.

    ``model_options`` contains flat architecture options for the selected model.
    Dimensions come from the training dataset and cannot be overridden. Sequence
    conditioning remains available through the advanced ``metis.models`` API.
    """

    FAMILIES = ("mlp", "transformer", "flow_matching", "itarflow")
    ARTIFACT_VERSION = 1

    def __init__(self, model: str = "mlp", model_options: Mapping | None = None):
        if model not in self.FAMILIES:
            raise ValueError(f"Unknown model {model!r}; choose one of {self.FAMILIES}")
        if model_options is not None and not isinstance(model_options, Mapping):
            raise ValueError(
                "model_options must be a dictionary of architecture options"
            )
        self.model_name = model
        self.model_options = _plain(dict(model_options or {}))
        self._validate_options()
        self.model: Any = None
        self.history: list[dict[str, Any]] = []
        self.training_metadata: dict[str, Any] = {}
        self._fitted = False

    @property
    def is_stochastic(self):
        return self.model_name in ("flow_matching", "itarflow")

    @property
    def resolved_model_options(self):
        """Architecture options including defaults, suitable for reproduction."""
        self._require_fit()
        return _plain(self._resolved_model_options)

    @property
    def model_config(self):
        """Resolved architecture including dataset-derived dimensions."""
        self._require_fit()
        return _plain(self._resolved_model_config)

    @property
    def parameter_count(self):
        """Number of trainable neural-network parameters."""
        self._require_fit()
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

    def _validate_options(self):
        from metis.models.mlp import ResidualMLPConfig
        from metis.models.transformer import TransformerConfig
        from metis.models.sit import SiTConfig
        from metis.models.itarflow import iTarFlowConfig

        config_type = {
            "mlp": ResidualMLPConfig,
            "transformer": TransformerConfig,
            "flow_matching": SiTConfig,
            "itarflow": iTarFlowConfig,
        }[self.model_name]
        inferred = {
            "input_dim",
            "output_dim",
            "sequence_len",
            "x_input_dim",
            "static_input_dim",
            "data_dim",
            "cond_dim",
            "num_timesteps",
            "out_channels",
            "static_in",
        }
        advanced = {
            "cross_attn",
            "observed_output_variables",
            "observed_variables",
            "conditioning_sequence",
        }
        if advanced.intersection(self.model_options):
            raise ValueError(
                "Surrogate supports static inputs only. For observed sequence conditioning, use the advanced metis.models API."
            )
        if inferred.intersection(self.model_options):
            raise ValueError(
                "Model input, output and time dimensions are inferred from SimulationDataset; remove dimension overrides from model_options."
            )
        allowed = (
            {f.name for f in fields(config_type)}
            - inferred
            - {"_target_", "cross_attn"}
        )
        if self.model_name == "flow_matching":
            allowed.add("interpolant")
        unknown = set(self.model_options) - allowed
        if unknown:
            raise ValueError(
                f"Unknown {self.model_name} model options: {sorted(unknown)}. Allowed: {sorted(allowed)}"
            )

    def _create_model(self):
        from metis.models.mlp import ResidualMLP, ResidualMLPConfig
        from metis.models.transformer import Transformer, TransformerConfig
        from metis.models.sit import (
            InterpolantTransformer,
            SiTConfig,
            LinearInterpolant,
            TrigonometricInterpolant,
        )
        from metis.models.itarflow import ImprovedTARFlow, iTarFlowConfig

        p, t, c = len(self.input_names), len(self.times), len(self.output_names)
        options = dict(self.model_options)
        conf: dict[str, Any]
        if self.model_name == "mlp":
            conf = dict(num_blocks=2, hidden_dim=64, num_layers_per_block=2)
            conf.update(options)
            for key in conf:
                _positive_int(conf[key], key)
            if conf["num_layers_per_block"] < 2:
                raise ValueError("num_layers_per_block must be at least 2")
            architecture = ResidualMLPConfig(input_dim=p, output_dim=t * c, **conf)
            model = ResidualMLP(architecture)
        elif self.model_name == "transformer":
            conf = dict(n_layer=2, n_head=4, n_kv_head=4, n_embd=64)
            conf.update(options)
            self._check_attention(conf, "n_embd", "n_head", "n_kv_head")
            _positive_int(conf["n_layer"], "n_layer")
            pattern = conf.get("window_pattern", "L")
            if not isinstance(pattern, str) or not pattern or set(pattern) - {"L", "S"}:
                raise ValueError(
                    "window_pattern must be a nonempty string containing only L and S"
                )
            architecture = TransformerConfig(
                sequence_len=t + 1, x_input_dim=c, static_input_dim=p, **conf
            )
            model = Transformer(architecture)
            model.init_weights()
        elif self.model_name == "flow_matching":
            name = options.pop("interpolant", "linear")
            if name not in ("linear", "flow-matching", "trig"):
                raise ValueError("interpolant must be 'linear' or 'trig'")
            conf = dict(
                n_blocks=2, n_head=4, n_kv_head=4, embd_dim=64, cond_dropout_prob=0.0
            )
            conf.update(options)
            self._check_attention(conf, "embd_dim", "n_head", "n_kv_head")
            _positive_int(conf["n_blocks"], "n_blocks")
            if not 0 <= conf["cond_dropout_prob"] < 1:
                raise ValueError("cond_dropout_prob must be in [0, 1)")
            interpolant = (
                TrigonometricInterpolant() if name == "trig" else LinearInterpolant()
            )
            architecture = SiTConfig(sequence_len=t, data_dim=c, cond_dim=p, **conf)
            model = InterpolantTransformer(architecture, interpolant)
        else:
            conf = dict(
                channels=64,
                num_layers=2,
                num_blocks=2,
                num_heads=4,
                num_static_tokens=4,
                patch_size=1,
            )
            conf.update(options)
            self._check_attention(conf, "channels", "num_heads")
            for key in ("num_layers", "num_blocks", "num_static_tokens", "patch_size"):
                _positive_int(conf[key], key)
            if t % conf["patch_size"]:
                raise ValueError(
                    "patch_size must divide the number of dataset timepoints"
                )
            architecture = iTarFlowConfig(
                num_timesteps=t, out_channels=c, static_in=p, **conf
            )
            model = ImprovedTARFlow(architecture)
        self._resolved_model_config = asdict(architecture)
        self._resolved_model_options = {
            key: value
            for key, value in self._resolved_model_config.items()
            if key
            not in {
                "_target_",
                "cross_attn",
                "input_dim",
                "output_dim",
                "sequence_len",
                "x_input_dim",
                "static_input_dim",
                "data_dim",
                "cond_dim",
                "num_timesteps",
                "out_channels",
                "static_in",
            }
        }
        if self.model_name == "flow_matching":
            self._resolved_model_options["interpolant"] = name
        return model

    @staticmethod
    def _check_attention(conf, width, heads, kv=None):
        for key in (width, heads) + ((kv,) if kv else ()):
            _positive_int(conf[key], key)
        if conf[width] % conf[heads] or (conf[width] // conf[heads]) % 2:
            raise ValueError(
                f"{width} must be divisible by {heads}, with an even dimension per head for rotary embeddings"
            )
        if kv and (conf[heads] % conf[kv]):
            raise ValueError(f"{heads} must be divisible by {kv}")

    def fit(
        self,
        train: SimulationDataset,
        validation: SimulationDataset | None = None,
        config: TrainingConfig | None = None,
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ):
        """Fit on training simulations and restore the best validation weights.

        Validation groups must be disjoint from training groups. For stochastic
        models validation uses the training objective, not trajectory MSE.
        Calls isolate their PyTorch random state from the caller's random state.
        ``on_progress`` receives a copy of each step's history record after any
        validation and best-state selection. Callback exceptions propagate.
        """
        if not isinstance(train, SimulationDataset):
            raise TypeError("train must be a SimulationDataset")
        cfg = config if config is not None else TrainingConfig()
        if not isinstance(cfg, TrainingConfig):
            raise TypeError("config must be a TrainingConfig")
        if on_progress is not None and not callable(on_progress):
            raise TypeError("on_progress must be callable or None")
        if validation is not None:
            DatasetSplit.from_datasets(train, validation)
        elif cfg.patience is not None:
            raise ValueError("patience requires a validation dataset")
        device = _device(cfg.device)
        self._fitted = False
        self.input_names = tuple(train.input_names)
        self.output_names = tuple(train.output_names)
        self.times = np.array(train.times, copy=True)
        self.input_scaler = Standardizer.fit(train.inputs, axis=0)
        self.output_scaler = Standardizer.fit(train.outputs, axis=0)
        self.domain_min = np.min(train.inputs, axis=0).copy()
        self.domain_max = np.max(train.inputs, axis=0).copy()
        self.training_bounds = {"min": self.domain_min, "max": self.domain_max}
        self.training_config = cfg
        self.device = device
        self.history = []
        inputs = torch.from_numpy(
            np.asarray(
                self.input_scaler.transform(train.inputs), dtype=np.float32
            ).copy()
        )
        outputs = torch.from_numpy(
            np.asarray(
                self.output_scaler.transform(train.outputs), dtype=np.float32
            ).copy()
        )
        val_tensors = None
        if validation is not None:
            val_tensors = (
                torch.from_numpy(
                    np.asarray(
                        self.input_scaler.transform(validation.inputs), dtype=np.float32
                    ).copy()
                ),
                torch.from_numpy(
                    np.asarray(
                        self.output_scaler.transform(validation.outputs),
                        dtype=np.float32,
                    ).copy()
                ),
            )
        best_state, best_loss, best_step, stale = None, float("inf"), None, 0
        stopped = False
        with torch.random.fork_rng(devices=_rng_devices(device)):
            _seed_rng(cfg.seed, device)
            self.model = self._create_model().to(device)
            # AdamW without fused/compiled paths also supports ordinary CPUs.
            decay = [
                p for p in self.model.parameters() if p.requires_grad and p.ndim >= 2
            ]
            no_decay = [
                p for p in self.model.parameters() if p.requires_grad and p.ndim < 2
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": cfg.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=cfg.learning_rate,
            )
            generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
            for step in range(1, cfg.steps + 1):
                self.model.train()
                idx = torch.randint(len(inputs), (cfg.batch_size,), generator=generator)
                x, y = inputs[idx].to(device), outputs[idx].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = self.model.compute_loss(x, y[:, :-1], y)
                self._finite(loss, f"Training loss at step {step}")
                loss.backward()
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        self._finite(parameter.grad, f"Gradient at step {step}")
                optimizer.step()
                record = {"step": step, "train_loss": float(loss.detach())}
                if val_tensors is not None and (
                    step % cfg.eval_every == 0 or step == cfg.steps
                ):
                    val_loss = self._validation_loss(val_tensors, cfg)
                    record["validation_loss"] = val_loss
                    if val_loss < best_loss:
                        best_loss, best_step, stale = val_loss, step, 0
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in self.model.state_dict().items()
                        }
                    else:
                        stale += 1
                    if cfg.patience is not None and stale >= cfg.patience:
                        stopped = True
                self.history.append(record)
                if on_progress is not None:
                    on_progress(dict(record))
                if stopped:
                    break
            if best_state is not None:
                self.model.load_state_dict(best_state)
            self.model.eval()
        self.training_metadata = {
            "config": asdict(cfg),
            "completed_steps": len(self.history),
            "best_step": best_step,
            "best_validation_loss": best_loss if best_step is not None else None,
            "stopped_early": stopped,
            "training_fingerprint": train.fingerprint,
            "validation_fingerprint": validation.fingerprint
            if validation is not None
            else None,
            "train_group_keys": _plain(train.group_keys),
            "validation_group_keys": _plain(validation.group_keys)
            if validation is not None
            else [],
            "normalization": "training data only; mean and population standard deviation per input and per output/timepoint",
            "parameter_count": sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            "environment": {
                **_environment_versions(),
                "fit_dtype": "float32",
                "fit_device": str(device),
            },
        }
        self._fitted = True
        return self

    def _validation_loss(self, tensors, cfg):
        x, y = tensors
        total = 0.0
        self.model.eval()
        # Equal Monte Carlo draws each evaluation improve comparability without
        # consuming the training stream's random numbers.
        with torch.random.fork_rng(devices=_rng_devices(self.device)), torch.no_grad():
            _seed_rng((cfg.seed + 1) % (2**63), self.device)
            for start in range(0, len(x), cfg.batch_size):
                xb, yb = (
                    x[start : start + cfg.batch_size].to(self.device),
                    y[start : start + cfg.batch_size].to(self.device),
                )
                loss = self.model.compute_loss(xb, yb[:, :-1], yb)
                self._finite(loss, "Validation loss")
                total += float(loss) * len(xb)
        return total / len(x)

    @staticmethod
    def _finite(value, name):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(
                f"{name} is nonfinite. Check data units and finite values, lower learning_rate, and inspect model options. No clipping was applied."
            )

    def _require_fit(self):
        if not self._fitted:
            raise RuntimeError(
                "Fit this Surrogate or load a saved artifact before inference"
            )

    def _check_keys(self, keys):
        if set(keys) != set(self.input_names) or len(keys) != len(self.input_names):
            raise ValueError(
                f"Input columns must be exactly {list(self.input_names)}; got {list(keys)}"
            )

    def _inputs(self, inputs, *, check_domain=True):
        self._require_fit()
        if isinstance(inputs, Mapping):
            self._check_keys(inputs.keys())
            columns = [np.asarray(inputs[name]) for name in self.input_names]
            if all(column.ndim == 0 for column in columns):
                values = np.array([[inputs[name] for name in self.input_names]])
            elif (
                all(column.ndim == 1 for column in columns)
                and len({len(column) for column in columns}) == 1
            ):
                values = np.column_stack(columns)
            else:
                raise ValueError(
                    "Input mapping values must all be scalars or equally sized one-dimensional columns"
                )
        elif (
            isinstance(inputs, (list, tuple))
            and inputs
            and isinstance(inputs[0], Mapping)
        ):
            for row in inputs:
                if not isinstance(row, Mapping):
                    raise ValueError("All input rows must be dictionaries")
                self._check_keys(row.keys())
            values = [[row[name] for name in self.input_names] for row in inputs]
        elif (
            hasattr(inputs, "columns")
            and hasattr(inputs, "select")
            and hasattr(inputs, "to_numpy")
        ):
            self._check_keys(inputs.columns)
            values = inputs.select(list(self.input_names)).to_numpy()
        elif (
            hasattr(inputs, "columns")
            and hasattr(inputs, "loc")
            and hasattr(inputs, "to_numpy")
        ):
            # Labelled pandas frames must not fall through to positional array
            # coercion. Duck typing keeps pandas an optional dependency.
            self._check_keys(inputs.columns)
            values = inputs.loc[:, list(self.input_names)].to_numpy()
        else:
            values = inputs
        try:
            array = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Inputs must contain numeric values") from exc
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != len(self.input_names) or not len(array):
            raise ValueError(
                f"Inputs must have shape (batch, {len(self.input_names)}) with a nonempty batch"
            )
        if not np.isfinite(array).all():
            raise ValueError("Inputs must contain only finite values")
        if check_domain:
            self._warn_domain(array)
        return array

    def _warn_domain(self, array):
        outside = np.any((array < self.domain_min) | (array > self.domain_max), axis=0)
        if outside.any():
            names = [name for name, flag in zip(self.input_names, outside) if flag]
            warnings.warn(
                f"Inputs outside the training ranges for {names}; predictions extrapolate and are not clipped.",
                UserWarning,
                stacklevel=3,
            )

    def predict(
        self,
        inputs,
        *,
        n_samples: int | None = None,
        seed: int = 0,
        num_steps: int = 50,
        solver: str = "heun",
    ) -> np.ndarray:
        """Return trajectories with shape ``(batch, time, output)`` in data units.

        Stochastic models require explicit ``n_samples`` and return their sample
        mean. Use :meth:`sample` to retain variability rather than average it.
        """
        self._require_fit()
        if self.is_stochastic:
            if n_samples is None:
                raise ValueError(
                    "This model is stochastic. Use sample(inputs, n_samples=100), or predict(inputs, n_samples=100) for a Monte Carlo mean."
                )
            return self.sample(
                inputs,
                n_samples=n_samples,
                seed=seed,
                num_steps=num_steps,
                solver=solver,
            ).mean(axis=1)
        if n_samples is not None:
            raise ValueError(
                "n_samples applies only to stochastic models; call predict(inputs) for this deterministic model"
            )
        array = self._inputs(inputs)
        values = np.asarray(self.input_scaler.transform(array), dtype=np.float32)
        batches = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(values), self.training_config.batch_size):
                x = torch.from_numpy(
                    values[start : start + self.training_config.batch_size].copy()
                ).to(self.device)
                if self.model_name == "mlp":
                    output = self.model(x).reshape(
                        len(x), len(self.times), len(self.output_names)
                    )
                else:
                    output = torch.cat(
                        list(
                            self.model.generate(
                                x, max_output_steps=len(self.times), use_kv_cache=False
                            )
                        ),
                        dim=1,
                    )
                self._finite(output, "Prediction")
                batches.append(output.cpu().numpy())
        return np.asarray(
            self.output_scaler.inverse_transform(np.concatenate(batches)),
            dtype=np.float32,
        )

    def sample(
        self,
        inputs,
        n_samples: int = 100,
        seed: int = 0,
        num_steps: int = 50,
        solver: str = "heun",
    ) -> np.ndarray:
        """Draw ``(batch, sample, time, output)`` trajectories in original units.

        Samples express fitted simulator variability, not uncertainty about
        network weights. ``seed`` controls an isolated random stream.
        """
        self._require_fit()
        if not self.is_stochastic:
            raise ValueError(
                "This model is deterministic; use predict(inputs). Choose flow_matching or itarflow to model simulation variability."
            )
        _positive_int(n_samples, "n_samples")
        _positive_int(num_steps, "num_steps")
        _valid_seed(seed)
        if solver not in ("euler", "heun"):
            raise ValueError("solver must be 'euler' or 'heun'")
        array = self._inputs(inputs)
        x_values = np.asarray(self.input_scaler.transform(array), dtype=np.float32)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        count = len(array) * n_samples
        batches = []
        self.model.eval()
        with torch.random.fork_rng(devices=_rng_devices(self.device)), torch.no_grad():
            _seed_rng(seed, self.device)
            for start in range(0, count, self.training_config.batch_size):
                stop = min(count, start + self.training_config.batch_size)
                ids = np.arange(start, stop) // n_samples
                x = torch.from_numpy(x_values[ids].copy()).to(self.device)
                z = torch.randn(
                    stop - start,
                    len(self.times),
                    len(self.output_names),
                    generator=generator,
                ).to(self.device)
                if self.model_name == "flow_matching":
                    output, _ = self.model.sample(
                        z, x, num_steps=num_steps, method=solver
                    )
                else:
                    output, _ = self.model.reverse(z, x)
                self._finite(output, "Generated samples")
                batches.append(output.cpu().numpy())
        samples = np.concatenate(batches).reshape(
            len(array), n_samples, len(self.times), len(self.output_names)
        )
        return np.asarray(
            self.output_scaler.inverse_transform(samples), dtype=np.float32
        )

    def log_prob_tensor(
        self, inputs: torch.Tensor, outputs: torch.Tensor
    ) -> torch.Tensor:
        """iTARFlow log density per trajectory, including constants and units.

        Inputs and outputs are in original units. Gradients through both tensors
        are preserved, permitting parameter fitting by a caller. This evaluates
        a continuous density; it is not a discrete count probability or a
        Bayesian posterior. Shape is ``(batch,)``.
        """
        self._require_fit()
        if self.model_name != "itarflow":
            raise ValueError("Exact log_prob is available only for itarflow")
        if not isinstance(inputs, torch.Tensor) or not isinstance(
            outputs, torch.Tensor
        ):
            raise TypeError(
                "log_prob_tensor requires PyTorch inputs and outputs tensors"
            )
        if (
            inputs.ndim != 2
            or inputs.shape[1] != len(self.input_names)
            or not len(inputs)
        ):
            raise ValueError(f"inputs must have shape (batch, {len(self.input_names)})")
        expected = (len(inputs), len(self.times), len(self.output_names))
        if tuple(outputs.shape) != expected:
            raise ValueError(f"outputs must have shape {expected}")
        self._finite(inputs, "Inputs")
        self._finite(outputs, "Outputs")
        self._warn_domain(inputs.detach().cpu().numpy())
        x, y = (
            inputs.to(device=self.device, dtype=torch.float64),
            outputs.to(device=self.device, dtype=torch.float64),
        )
        xm = torch.tensor(self.input_scaler.mean, dtype=x.dtype, device=self.device)
        xs = torch.tensor(self.input_scaler.scale, dtype=x.dtype, device=self.device)
        ym = torch.tensor(self.output_scaler.mean, dtype=y.dtype, device=self.device)
        ys = torch.tensor(self.output_scaler.scale, dtype=y.dtype, device=self.device)
        self.model.eval()
        # Match Standardizer exactly: center/scale with saved float64 statistics,
        # then cast standardized values for the float32 neural network. Casting
        # means before centering loses small variations around a large offset.
        z, mean_logdet = self.model(
            ((y - ym) / ys).to(torch.float32),
            ((x - xm) / xs).to(torch.float32),
        )
        dimensions = len(self.times) * len(self.output_names)
        result = -0.5 * (z.double().square() + math.log(2 * math.pi)).flatten(1).sum(1)
        result = result + dimensions * mean_logdet.double() - torch.log(ys).sum()
        self._finite(result, "Log probability")
        return result

    def log_prob(self, inputs, outputs) -> np.ndarray:
        """Return iTARFlow trajectory log densities in original output units."""
        array = self._inputs(inputs, check_domain=False)
        try:
            y = np.asarray(outputs, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise ValueError("Outputs must contain numeric values") from exc
        if y.ndim == 2 and len(array) == 1:
            y = y[None]
        with torch.no_grad():
            return (
                self.log_prob_tensor(
                    torch.from_numpy(array.copy()), torch.from_numpy(y.copy())
                )
                .cpu()
                .numpy()
            )

    def evaluate(self, dataset, **kwargs):
        """Evaluate held-out simulations using :mod:`metis.evaluation`."""
        from metis.evaluation import evaluate

        return evaluate(self, dataset, **kwargs)

    def save(self, path):
        """Atomically write a portable inference bundle; no training bank needed."""
        self._require_fit()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "format": "metis-surrogate",
            "format_version": self.ARTIFACT_VERSION,
            "producer": _environment_versions(),
            "model_name": self.model_name,
            "model_options": _plain(self._resolved_model_options),
            "model_config": _plain(self._resolved_model_config),
            "weights": {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            },
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "times": _plain(self.times),
            "input_scaler": _plain(self.input_scaler.to_dict()),
            "output_scaler": _plain(self.output_scaler.to_dict()),
            "domain_min": _plain(self.domain_min),
            "domain_max": _plain(self.domain_max),
            "training_config": _plain(asdict(self.training_config)),
            "history": _plain(self.history),
            "training_metadata": _plain(self.training_metadata),
        }
        fd, temporary = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
        )
        os.close(fd)
        try:
            torch.save(bundle, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    @classmethod
    def load(cls, path, device="cpu"):
        """Load a safe, allowlisted inference bundle onto CPU or CUDA.

        This uses ``torch.load(weights_only=True)`` and never imports arbitrary
        model targets from artifact metadata. Optimizer resume is not supported.
        """
        target_device = _device(device)
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(bundle, dict) or bundle.get("format") != "metis-surrogate":
            raise ValueError("Not a METIS Surrogate artifact")
        if bundle.get("format_version") != cls.ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported METIS artifact version {bundle.get('format_version')!r}"
            )
        required = {
            "model_name",
            "model_options",
            "weights",
            "input_names",
            "output_names",
            "times",
            "input_scaler",
            "output_scaler",
            "domain_min",
            "domain_max",
            "training_config",
            "history",
            "training_metadata",
        }
        if required - bundle.keys():
            raise ValueError(
                f"Incomplete METIS artifact; missing {sorted(required - bundle.keys())}"
            )
        surrogate = cls(bundle["model_name"], bundle["model_options"])
        surrogate.input_names = tuple(bundle["input_names"])
        surrogate.output_names = tuple(bundle["output_names"])
        surrogate.times = np.asarray(bundle["times"])
        for name, values in (
            ("input_names", surrogate.input_names),
            ("output_names", surrogate.output_names),
        ):
            if (
                not values
                or any(not isinstance(v, str) or not v for v in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"Invalid artifact {name}")
        if (
            surrogate.times.ndim != 1
            or not len(surrogate.times)
            or not np.isfinite(surrogate.times).all()
            or np.any(np.diff(surrogate.times) <= 0)
        ):
            raise ValueError("Invalid artifact time grid")
        surrogate.input_scaler = Standardizer.from_dict(bundle["input_scaler"])
        surrogate.output_scaler = Standardizer.from_dict(bundle["output_scaler"])
        p, t, c = (
            len(surrogate.input_names),
            len(surrogate.times),
            len(surrogate.output_names),
        )
        if np.shape(surrogate.input_scaler.mean) != (p,) or np.shape(
            surrogate.output_scaler.mean
        ) != (t, c):
            raise ValueError(
                "Artifact normalization dimensions do not match its schema"
            )
        surrogate.domain_min = np.asarray(bundle["domain_min"], dtype=np.float32)
        surrogate.domain_max = np.asarray(bundle["domain_max"], dtype=np.float32)
        if (
            surrogate.domain_min.shape != (p,)
            or surrogate.domain_max.shape != (p,)
            or not np.isfinite(surrogate.domain_min).all()
            or not np.isfinite(surrogate.domain_max).all()
            or np.any(surrogate.domain_min > surrogate.domain_max)
        ):
            raise ValueError("Invalid artifact training bounds")
        surrogate.training_bounds = {
            "min": surrogate.domain_min,
            "max": surrogate.domain_max,
        }
        training_config = dict(bundle["training_config"])
        training_config["device"] = str(target_device)
        surrogate.training_config = TrainingConfig(**training_config)
        surrogate.history = bundle["history"]
        surrogate.training_metadata = bundle["training_metadata"]
        surrogate.device = target_device
        with torch.random.fork_rng(devices=_rng_devices(target_device)):
            surrogate.model = surrogate._create_model().to(target_device)
            surrogate.model.load_state_dict(bundle["weights"], strict=True)
        surrogate.model.eval()
        surrogate._fitted = True
        return surrogate
