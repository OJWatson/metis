"""One training engine for the Python and Hydra interfaces.

Training checkpoints resume optimizer updates; they are deliberately separate
from portable :class:`metis.Surrogate` inference artifacts. A checkpoint is only
resumable with the same data, model, preprocessing, total step budget and runtime
topology. Its tensors are loaded with PyTorch's restricted ``weights_only`` loader.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import tempfile
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class TrainingConfig:
    """Training controls; ``steps`` is the total number of optimizer updates.

    ``batch_size`` is global across ``num_devices`` processes, including CPU
    processes. ``patience`` counts unsuccessful validation evaluations. Resume
    retains the original total step budget, rather than adding another budget.
    ``checkpoint_every`` and ``checkpoint_dir`` must be supplied together.
    """

    steps: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.001
    seed: int = 42
    device: str = "cpu"
    eval_every: int = 50
    patience: int | None = None
    schedule: str = "constant"
    warmup_fraction: float = 0.2
    min_learning_rate: float = 1e-6
    precision: str = "fp32"
    compile_model: bool = False
    deterministic: bool = False
    num_devices: int = 1
    checkpoint_every: int | None = None
    checkpoint_dir: str | None = None
    resume_from: str | None = None
    optimizer_betas: tuple[float, float] | None = None

    def __post_init__(self):
        for name in ("steps", "batch_size", "eval_every", "num_devices"):
            _positive_int(getattr(self, name), name)
        for name in ("patience", "checkpoint_every"):
            if getattr(self, name) is not None:
                _positive_int(getattr(self, name), name)
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("seed must be an integer between 0 and 2**63 - 1")
        for name in (
            "learning_rate",
            "weight_decay",
            "warmup_fraction",
            "min_learning_rate",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError(
                "warmup_fraction must be between 0 (inclusive) and 1 (exclusive)"
            )
        if self.schedule not in ("constant", "warmup_cosine"):
            raise ValueError("schedule must be 'constant' or 'warmup_cosine'")
        if (
            self.schedule == "warmup_cosine"
            and self.min_learning_rate > self.learning_rate
        ):
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if self.precision not in ("fp32", "tf32", "bf16"):
            raise ValueError("precision must be 'fp32', 'tf32', or 'bf16'")
        if not isinstance(self.device, str):
            raise ValueError("device must be a string such as 'cpu' or 'cuda:0'")
        for name in ("compile_model", "deterministic"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.compile_model and self.deterministic:
            raise ValueError(
                "compile_model=True is not supported with deterministic=True; disable compilation for deterministic training"
            )
        if self.batch_size % self.num_devices:
            raise ValueError(
                "batch_size is global and must be divisible by num_devices"
            )
        if (self.checkpoint_every is None) != (self.checkpoint_dir is None):
            raise ValueError(
                "checkpoint_every and checkpoint_dir must be supplied together"
            )
        for name in ("checkpoint_dir", "resume_from"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (str, os.PathLike)) or not str(value):
                    raise ValueError(f"{name} must be a nonempty filesystem path")
                object.__setattr__(self, name, str(value))
        if self.optimizer_betas is not None:
            if (
                not isinstance(self.optimizer_betas, (list, tuple))
                or len(self.optimizer_betas) != 2
            ):
                raise ValueError("optimizer_betas must contain two numbers in [0, 1)")
            for value in self.optimizer_betas:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value < 1
                ):
                    raise ValueError(
                        "optimizer_betas must contain two numbers in [0, 1)"
                    )
            object.__setattr__(self, "optimizer_betas", tuple(self.optimizer_betas))


@dataclass
class TrainingResult:
    history: list[dict[str, Any]]
    best_step: int | None
    best_loss: float | None
    stopped_early: bool
    completed_steps: int


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _device(config, rank=0):
    try:
        device = torch.device(config.device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("device must be 'cpu' or 'cuda[:index]'") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Training supports device='cpu' or 'cuda[:index]'")
    if device.type == "cpu":
        if config.precision == "tf32":
            raise ValueError(
                "precision='tf32' requires CUDA; use 'fp32' or 'bf16' on CPU"
            )
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable. Use device='cpu' or install a compatible CUDA PyTorch build."
        )
    base = device.index if device.index is not None else torch.cuda.current_device()
    if base + config.num_devices > torch.cuda.device_count():
        raise ValueError(
            "num_devices exceeds the available contiguous CUDA devices starting at device"
        )
    device = torch.device("cuda", base + rank)
    if config.precision == "bf16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError(f"precision='bf16' is not supported on {device}")
    return device


def _seed(seed, device):
    torch.random.default_generator.manual_seed(seed % (2**63))
    if device.type == "cuda":
        torch.cuda.default_generators[device.index].manual_seed(seed % (2**63))


@contextmanager
def _runtime_options(config, device):
    """Keep precision, determinism and random streams local to this fit."""
    previous = (
        torch.get_float32_matmul_precision(),
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            # This API also controls matmul TF32. Mixing its setter with
            # allow_tf32 loses a caller's 'medium' setting in PyTorch 2.10.
            torch.set_float32_matmul_precision(
                "high" if config.precision == "tf32" else "highest"
            )
            torch.backends.cudnn.allow_tf32 = config.precision == "tf32"
            torch.use_deterministic_algorithms(config.deterministic)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = config.deterministic
            if config.deterministic:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            yield
        finally:
            torch.set_float32_matmul_precision(previous[0])
            torch.backends.cudnn.allow_tf32 = previous[1]
            torch.use_deterministic_algorithms(previous[2], warn_only=previous[3])
            torch.backends.cudnn.benchmark = previous[4]
            torch.backends.cudnn.deterministic = previous[5]
            torch.backends.cuda.enable_flash_sdp(previous[6])
            torch.backends.cuda.enable_mem_efficient_sdp(previous[7])
            torch.backends.cuda.enable_math_sdp(previous[8])
            if previous[9] is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous[9]


def _autocast(config, device):
    if config.precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


class _TrainingLoss(nn.Module):
    def __init__(self, model, pred_and_obs_idx):
        super().__init__()
        self.model = model
        self.pred_and_obs_idx = pred_and_obs_idx

    def forward(self, inputs, targets):
        if self.pred_and_obs_idx is None:
            return self.model.compute_loss(inputs, targets[:, :-1], targets)
        return self.model.compute_loss(
            inputs,
            targets[:, :-1],
            targets,
            pred_and_obs_idx=self.pred_and_obs_idx,
        )


def _learning_rate(config, step):
    if config.schedule == "constant":
        return config.learning_rate
    warmup = int(config.steps * config.warmup_fraction)
    if step <= warmup:
        return config.learning_rate * step / warmup
    # The first post-warmup update uses the peak and the last uses the floor.
    decay_steps = config.steps - warmup
    progress = (step - warmup - 1) / max(1, decay_steps - 1)
    return (
        config.min_learning_rate
        + (config.learning_rate - config.min_learning_rate)
        * (1 + math.cos(math.pi * progress))
        / 2
    )


def _cpu_state(model):
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _fingerprint(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _json_plain(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint metadata must contain only finite JSON-compatible values"
        ) from exc


def _contract(model, inputs, outputs, validation, config, metadata, pred_and_obs_idx):
    settings = asdict(config)
    for name in ("checkpoint_every", "checkpoint_dir", "resume_from"):
        settings.pop(name)
    return _json_plain(
        {
            "config": settings,
            "metadata": metadata,
            "tensors": [_fingerprint(inputs), _fingerprint(outputs)]
            + (
                [_fingerprint(value) for value in validation]
                if validation is not None
                else []
            ),
            "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
            "state_schema": {
                name: [list(value.shape), str(value.dtype)]
                for name, value in model.state_dict().items()
            },
            "pred_and_obs_idx": pred_and_obs_idx,
            "torch_version": str(torch.__version__),
        }
    )


def _load_checkpoint(path, contract):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Cannot read a safe Metis training checkpoint from {path!s}. Raw legacy checkpoints are unsupported; use a checkpoint produced by the installed training engine."
        ) from exc
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != "metis-training"
        or checkpoint.get("version") != 1
    ):
        raise ValueError(
            "Unsupported training checkpoint. Expected metis-training version 1; inference artifacts and raw legacy checkpoints cannot resume training."
        )
    if checkpoint.get("contract") != contract:
        raise ValueError(
            "Training checkpoint does not match the data, model, preprocessing metadata, training settings, PyTorch version or device topology. Resume with the original settings and total steps."
        )
    required = {
        "model_state",
        "optimizer_state",
        "rank_rng",
        "history",
        "completed_steps",
        "best_state",
        "best_step",
        "best_loss",
        "stale",
        "stopped_early",
    }
    if (
        not required <= checkpoint.keys()
        or len(checkpoint["rank_rng"]) != contract["config"]["num_devices"]
    ):
        raise ValueError("Training checkpoint is incomplete")
    if (
        not 0 <= checkpoint["completed_steps"] <= contract["config"]["steps"]
        or len(checkpoint["history"]) != checkpoint["completed_steps"]
    ):
        raise ValueError("Training checkpoint has inconsistent step history")
    return checkpoint


def _atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _assert_finite(value, name, device, world_size):
    invalid = torch.tensor(int(not bool(torch.isfinite(value).all())), device=device)
    if world_size > 1:
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
    if invalid.item():
        raise ValueError(
            f"{name} is nonfinite. Check data units and finite values, lower learning_rate, and inspect model options. No clipping was applied."
        )


def _evaluate(loss_module, validation, config, device, rank, world_size):
    inputs, outputs = validation
    total = torch.zeros(2, dtype=torch.float64, device=device)
    loss_module.eval()
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        _seed(config.seed + 1 + rank, device)
        local_indices = torch.arange(rank, len(inputs), world_size)
        local_batch = config.batch_size // world_size
        for indices in local_indices.split(local_batch):
            if not len(indices):
                continue
            with _autocast(config, device):
                loss = loss_module(
                    inputs[indices].to(device), outputs[indices].to(device)
                )
            # Every rank may have a different number of validation batches, so
            # defer collective error detection until all local batches finish.
            total[0] += loss.detach().double() * len(indices)
            total[1] += len(indices)
    if world_size > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    _assert_finite(total, "Validation loss", device, world_size)
    return float(total[0] / total[1])


def _train_loop(
    model,
    inputs,
    outputs,
    validation,
    config,
    contract,
    checkpoint,
    on_progress,
    pred_and_obs_idx,
    rank=0,
):
    world_size = config.num_devices
    device = _device(config, rank)
    model.to(device)
    loss_module = _TrainingLoss(model, pred_and_obs_idx)
    forward = torch.compile(loss_module) if config.compile_model else loss_module
    if world_size > 1:
        forward = DistributedDataParallel(
            forward, device_ids=[device.index] if device.type == "cuda" else None
        )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for p in model.parameters() if p.requires_grad and p.ndim >= 2
                ],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [
                    p for p in model.parameters() if p.requires_grad and p.ndim < 2
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=config.learning_rate,
        betas=config.optimizer_betas or (0.9, 0.999),
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    history, best_state, best_step, best_loss, stale, completed, stopped = (
        [],
        None,
        None,
        None,
        0,
        0,
        False,
    )
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = deepcopy(checkpoint["history"])
        best_state, best_step, best_loss = (
            checkpoint["best_state"],
            checkpoint["best_step"],
            checkpoint["best_loss"],
        )
        stale, completed, stopped = (
            checkpoint["stale"],
            checkpoint["completed_steps"],
            checkpoint["stopped_early"],
        )
        rng = checkpoint["rank_rng"][rank]
        generator.set_state(rng["batch"])
        torch.set_rng_state(rng["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], device)
    local_batch = config.batch_size // world_size
    for step in range(completed + 1, config.steps + 1):
        if stopped:
            break
        loss_module.train()
        global_indices = torch.randint(
            len(inputs), (config.batch_size,), generator=generator
        )
        indices = global_indices[rank * local_batch : (rank + 1) * local_batch]
        learning_rate = _learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with _autocast(config, device):
            loss = forward(inputs[indices].to(device), outputs[indices].to(device))
        _assert_finite(loss, f"Training loss at step {step}", device, world_size)
        loss.backward()
        finite_gradients = torch.tensor(
            all(
                bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
                if p.grad is not None
            ),
            device=device,
        )
        _assert_finite(
            torch.where(finite_gradients, 0.0, float("nan")),
            f"Gradient at step {step}",
            device,
            world_size,
        )
        optimizer.step()
        train_loss = loss.detach().float()
        if world_size > 1:
            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
            train_loss /= world_size
        record = {
            "step": step,
            "train_loss": float(train_loss),
            "learning_rate": learning_rate,
        }
        improved = False
        if validation is not None and (
            step % config.eval_every == 0 or step == config.steps
        ):
            validation_loss = _evaluate(
                loss_module, validation, config, device, rank, world_size
            )
            record["validation_loss"] = validation_loss
            if best_loss is None or validation_loss < best_loss:
                best_loss, best_step, stale, improved = validation_loss, step, 0, True
                best_state = _cpu_state(model)
            else:
                stale += 1
            stopped = config.patience is not None and stale >= config.patience
        history.append(record)
        completed = step
        if config.checkpoint_dir is not None and (
            improved
            or step % config.checkpoint_every == 0
            or step == config.steps
            or stopped
        ):
            rng = {
                "batch": generator.get_state(),
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else None,
            }
            rank_rng = [None] * world_size
            if world_size > 1:
                dist.all_gather_object(rank_rng, rng)
            else:
                rank_rng[0] = rng
            if rank == 0:
                state = {
                    "format": "metis-training",
                    "version": 1,
                    "contract": contract,
                    "model_state": _cpu_state(model),
                    "optimizer_state": optimizer.state_dict(),
                    "rank_rng": rank_rng,
                    "history": history,
                    "completed_steps": completed,
                    "best_state": best_state,
                    "best_step": best_step,
                    "best_loss": best_loss,
                    "stale": stale,
                    "stopped_early": stopped,
                }
                _atomic_save(state, Path(config.checkpoint_dir) / "last.pt")
                if improved:
                    _atomic_save(state, Path(config.checkpoint_dir) / "best.pt")
        # Save first: interruption by a progress callback leaves a resumable step.
        if rank == 0 and on_progress is not None:
            on_progress(dict(record))
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(history, best_step, best_loss, bool(stopped), completed)


def _worker(
    rank,
    model,
    inputs,
    outputs,
    validation,
    config,
    contract,
    checkpoint,
    pred_and_obs_idx,
    rendezvous,
    result_path,
    progress,
    acknowledgements,
    cpu_rng,
    cuda_rng,
    threads,
):
    torch.set_num_threads(threads)
    # Multiprocessing shares CPU tensor storage while pickling arguments. Each
    # optimizer must own its parameter storage or CPU ranks update it twice.
    model = deepcopy(model)
    # AdamW's CPU moments and step counters also retain their loaded tensor
    # storage. A resumed worker must not share them with other workers.
    checkpoint = deepcopy(checkpoint)
    device = _device(config, rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    with _runtime_options(config, device):
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=config.num_devices,
        )
        if rank == 0:
            torch.set_rng_state(cpu_rng)
            if device.type == "cuda" and cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
        else:
            _seed(config.seed + rank, device)

        def report(record):
            progress.put(record)
            acknowledgements.get()

        result = _train_loop(
            model,
            inputs,
            outputs,
            validation,
            config,
            contract,
            checkpoint,
            report if progress is not None else None,
            pred_and_obs_idx,
            rank,
        )
        if rank == 0:
            _atomic_save(
                {"model_state": _cpu_state(model), "result": asdict(result)},
                result_path,
            )
        dist.destroy_process_group()


def run_training(
    model: nn.Module,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    validation: tuple[torch.Tensor, torch.Tensor] | None,
    config: TrainingConfig,
    *,
    metadata: Mapping[str, Any],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    pred_and_obs_idx: tuple[list[int], list[int]] | None = None,
) -> TrainingResult:
    """Fit a constructed model, restoring its best validation weights in place.

    Inputs and outputs are finite, normalized CPU tensors. The caller constructs
    the seeded model and supplies its resolved model/preprocessing metadata.
    The callback runs only in the parent process, including distributed fits.
    Multiprocess callers must use Python's ``if __name__ == '__main__':`` guard.
    """
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if on_progress is not None and not callable(on_progress):
        raise TypeError("on_progress must be callable or None")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    if validation is None and config.patience is not None:
        raise ValueError("patience requires a validation dataset")
    pairs = [(inputs, outputs)] + ([validation] if validation is not None else [])
    for x, y in pairs:
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("Training inputs and outputs must be CPU tensors")
        if x.device.type != "cpu" or y.device.type != "cpu":
            raise ValueError(
                "Training input and output tensors must be on CPU; device controls minibatch placement"
            )
        if x.ndim != 2 or y.ndim != 3 or len(x) == 0 or len(x) != len(y):
            raise ValueError(
                "Expected inputs [runs, parameters] and outputs [runs, times, variables] with matching nonzero run counts"
            )
        if (
            not x.is_floating_point()
            or not y.is_floating_point()
            or not bool(torch.isfinite(x).all())
            or not bool(torch.isfinite(y).all())
        ):
            raise ValueError("Training tensors must be finite floating-point values")
    device = _device(config)
    contract = _contract(
        model, inputs, outputs, validation, config, dict(metadata), pred_and_obs_idx
    )
    checkpoint = (
        _load_checkpoint(config.resume_from, contract)
        if config.resume_from is not None
        else None
    )
    if config.num_devices == 1:
        with _runtime_options(config, device):
            return _train_loop(
                model,
                inputs,
                outputs,
                validation,
                config,
                contract,
                checkpoint,
                on_progress,
                pred_and_obs_idx,
            )
    if not dist.is_available():
        raise ValueError("This PyTorch build does not support distributed training")
    # Spawned workers own their devices. Transferring through a temporary file
    # avoids queue tensor storage lifetimes after the producing process exits.
    context = mp.get_context("spawn")
    progress = context.Queue() if on_progress is not None else None
    acknowledgements = context.Queue() if on_progress is not None else None
    processes = None
    try:
        with tempfile.TemporaryDirectory(prefix="metis-training-") as directory:
            result_path = str(Path(directory) / "result.pt")
            processes = mp.spawn(
                _worker,
                args=(
                    deepcopy(model).cpu(),
                    inputs,
                    outputs,
                    validation,
                    config,
                    contract,
                    checkpoint,
                    pred_and_obs_idx,
                    Path(directory, "rendezvous").as_uri(),
                    result_path,
                    progress,
                    acknowledgements,
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    max(1, torch.get_num_threads() // config.num_devices),
                ),
                nprocs=config.num_devices,
                join=False,
            )
            while True:
                if progress is not None:
                    try:
                        record = progress.get(timeout=0.05)
                    except queue.Empty:
                        pass
                    else:
                        assert on_progress is not None and acknowledgements is not None
                        on_progress(dict(record))
                        acknowledgements.put(True)
                if processes.join(timeout=0.05):
                    break
            payload = torch.load(result_path, map_location="cpu", weights_only=True)
            model.load_state_dict(payload["model_state"])
            model.to(device).eval()
            return TrainingResult(**payload["result"])
    finally:
        if processes is not None:
            for process in processes.processes:
                if process.is_alive():
                    process.terminate()
            for process in processes.processes:
                process.join(timeout=5)
        for channel in (progress, acknowledgements):
            if channel is not None:
                channel.close()
                channel.cancel_join_thread()
