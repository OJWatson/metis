"""Installed Hydra frontend for the same public API used in Python notebooks.

Run ``metis-train --help`` for the packaged configuration groups. Hydra manages
overrides, experiment directories, and multirun sweeps; it does not introduce a
second data loader, training loop, or model artifact format.
"""

from __future__ import annotations

from dataclasses import replace
import json
import logging
from pathlib import Path
from typing import Any
import warnings

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from metis.data import DatasetSplit, SimulationDataset
from metis.surrogate import Surrogate, TrainingConfig
from metis.tracking import WandbTracker

log = logging.getLogger(__name__)


def _mapping(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(value, dict):
        raise TypeError("Expected a mapping in the METIS Hydra configuration")
    return {str(key): item for key, item in value.items()}


def _dataset(config: DictConfig) -> SimulationDataset | DatasetSplit:
    # Hydra can change the process working directory. Relative Parquet paths
    # always refer to the directory from which the user launched this command.
    config = DictConfig(_mapping(config))
    if str(config.get("_target_", "")).endswith("SimulationDataset.from_parquet"):
        config.path = to_absolute_path(str(config.path))
    result = instantiate(config, _convert_="all")
    if not isinstance(result, (SimulationDataset, DatasetSplit)):
        raise TypeError(
            "dataset._target_ must return a SimulationDataset or DatasetSplit. "
            "Use dataset=example or dataset=parquet, or supply a custom factory."
        )
    return result


def _wandb(config: DictConfig, *, output_dir: Path) -> WandbTracker:
    return WandbTracker(
        enabled=config.wandb.enabled,
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name,
        mode=config.wandb.mode,
        tags=list(config.wandb.tags),
        dir=str(output_dir),
        config=_mapping(config),
    )


def _warn_test_sweep(partition_name: str) -> None:
    if partition_name == "test" and HydraConfig.initialized():
        from hydra.types import RunMode

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            warnings.warn(
                "This sweep evaluates the test partition for every setting. "
                "Use evaluation.partition=validation when selecting models, then "
                "evaluate the chosen model on the test bank once.",
                UserWarning,
                stacklevel=2,
            )


def run(config: DictConfig, *, output_dir: str | Path | None = None) -> dict:
    """Execute a composed configuration and return local artifact locations.

    A dataset factory may return an explicit :class:`DatasetSplit`; in that
    case ``split`` fractions are unused. Otherwise complete scenario groups are
    split with the configured fractions and seed. Evaluation uses the explicitly
    chosen partition, with no automatic fallback to training data.

    W&B is optional and receives training metrics live via the same progress
    callback available in Python. Simulation data and model files are not
    uploaded. Local artifacts are produced in either case.
    """
    # Fail early on incomplete Parquet templates rather than beginning a run.
    _mapping(config)
    dataset = _dataset(config.dataset)
    partitions = (
        dataset
        if isinstance(dataset, DatasetSplit)
        else dataset.split(**_mapping(config.split))
    )
    model = instantiate(config.model, _convert_="all")
    training = instantiate(config.training, _convert_="all")
    if not isinstance(model, Surrogate):
        raise TypeError("model._target_ must construct a metis.Surrogate.")
    if not isinstance(training, TrainingConfig):
        raise TypeError("training._target_ must construct a metis.TrainingConfig.")
    partition_name = config.evaluation.partition
    if partition_name not in ("train", "validation", "test"):
        raise ValueError("evaluation.partition must be train, validation, or test.")
    _warn_test_sweep(partition_name)
    evaluation_data = getattr(partitions, partition_name)
    if evaluation_data is None:
        raise ValueError(
            f"No {partition_name} partition exists. Enable its split fraction or "
            "explicitly choose another evaluation.partition."
        )
    if output_dir is None:
        if not HydraConfig.initialized():
            raise ValueError(
                "Provide output_dir when calling run() outside a Hydra job."
            )
        output_dir = HydraConfig.get().runtime.output_dir
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Resume refers to the launch directory even with hydra.job.chdir=true.
    # Relative checkpoint destinations belong to this job's output directory.
    training = replace(
        training,
        resume_from=to_absolute_path(training.resume_from)
        if training.resume_from
        else None,
        checkpoint_dir=str(output_dir / training.checkpoint_dir)
        if training.checkpoint_dir
        else None,
    )
    OmegaConf.save(config, output_dir / "config.yaml", resolve=True)
    with _wandb(config, output_dir=output_dir) as tracking:
        log.info(
            "Training %s on %d realizations (%d scenario groups); evaluating %s",
            model.model_name,
            len(partitions.train),
            len(set(partitions.train.group_keys)),
            partition_name,
        )
        model.fit(
            partitions.train,
            partitions.validation,
            config=training,
            on_progress=tracking.log_training if config.wandb.enabled else None,
        )
        model_path = output_dir / "model.pt"
        model.save(model_path)
        history_path = output_dir / "history.json"
        history_path.write_text(json.dumps(model.history, indent=2, allow_nan=False))
        options = _mapping(config.evaluation)
        options.pop("partition")
        report = model.evaluate(evaluation_data, **options)
        report.metadata["evaluation_partition"] = partition_name
        report.metadata["frontend"] = "Hydra; shared metis.Surrogate API"
        metrics_path = output_dir / "metrics.json"
        report.save(metrics_path)
        if config.plot.enabled:
            figure = report.plot(
                output=config.plot.output, max_runs=config.plot.max_runs
            )
            figure.savefig(output_dir / "evaluation.png", dpi=150)
            # Import plotting only when explicitly requested.
            import matplotlib.pyplot as plt

            plt.close(figure)
        tracking.log_evaluation(report, prefix=partition_name)
        result = {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "history": str(history_path),
            "evaluation_partition": partition_name,
        }
        log.info("Saved model and evaluation to %s", output_dir)
        print(json.dumps(result, indent=2))
        return result


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """Hydra console entrypoint, also called by ``metis train``."""
    run(config)


if __name__ == "__main__":
    main()
