"""Reproducible CPU pilot comparing shared held-out simulation trajectories.

Run after installing METIS, for example::

    python benchmarks/compare_models.py --steps 100 --output results/pilot-100

The bundled bank is a small SIRDS teaching subset. This is an engineering pilot,
not evidence of universal architecture superiority or scientific adequacy.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from metis import SimulationDataset, Surrogate, TrainingConfig, evaluate, load_example


SIRDS_INPUTS = ["I_0", "beta", "gamma", "omega", "mu"]
SIRDS_OUTPUTS = ["S_t", "I_t", "R_t", "D_t"]
MODEL_FAMILIES = ("mlp", "transformer", "flow_matching", "itarflow")
LIMITATIONS = (
    "Bounded engineering pilot on a specified SIRDS simulation bank. One split and "
    "one training seed; no hyperparameter search or surrogate-error confidence "
    "intervals. Parameter counts, objectives and inference costs differ. Compare "
    "shared original-unit test metrics, never validation losses across families. "
    "The bundled short teaching subset does not establish universal architecture "
    "superiority, long-horizon fidelity, or adequacy for a scientific application."
)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _environment():
    versions = {}
    for name in ("metis-surrogate", "torch", "numpy", "polars", "hydra-core"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed as a distribution"
    result: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        result.update(git_commit=commit, git_dirty=bool(status))
    except (OSError, subprocess.CalledProcessError):
        result.update(git_commit=None, git_dirty=None)
    return result


def _partition_metadata(dataset):
    return {
        "fingerprint": dataset.fingerprint,
        "runs": len(dataset),
        "groups": len(set(dataset.group_keys)),
        "group_keys": [list(key) for key in dict.fromkeys(dataset.group_keys)],
    }


class MeanTrajectoryBaseline:
    """A training-bank mean trajectory, independent of the input parameters."""

    model_name = "train_mean_baseline"
    parameter_count = 0

    def __init__(self, train):
        self.input_names = train.input_names
        self.output_names = train.output_names
        self.times = train.times
        self.mean_trajectory = train.outputs.astype(np.float64).mean(axis=0)
        self.training_metadata = {
            "training_fingerprint": train.fingerprint,
            "validation_fingerprint": None,
            "train_group_keys": [list(key) for key in train.group_keys],
            "validation_group_keys": [],
        }

    def predict(self, inputs):
        return np.broadcast_to(
            self.mean_trajectory, (len(inputs), *self.mean_trajectory.shape)
        ).copy()

    def save(self, path):
        _write_json(
            path,
            {
                "format": "metis-training-mean-baseline",
                "format_version": 1,
                "input_names": list(self.input_names),
                "output_names": list(self.output_names),
                "times": self.times.tolist(),
                "mean_trajectory": self.mean_trajectory.tolist(),
                "training_metadata": self.training_metadata,
            },
        )


def _measure_inference(
    model, inputs, *, n_samples, seed, num_steps, solver, batch_size, device
):
    """Warm once, then measure complete trajectories, including CPU transfers.

    This reports deterministic predictions/s or stochastic draws/s separately.
    It does not compare against simulator execution time and is not ABM speedup.
    """
    stochastic = model.model_name in ("flow_matching", "itarflow")

    def synchronize():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(torch.device(device))

    def infer(batch, batch_seed, samples):
        if stochastic:
            return model.sample(
                batch,
                n_samples=samples,
                seed=batch_seed,
                num_steps=num_steps,
                solver=solver,
            )
        return model.predict(batch)

    infer(inputs[:1], seed, 1)
    synchronize()
    start = time.perf_counter()
    generated = 0
    for first in range(0, len(inputs), batch_size):
        batch = inputs[first : first + batch_size]
        batch_seed = (
            int(
                np.random.SeedSequence([seed, first]).generate_state(
                    1, dtype=np.uint64
                )[0]
            )
            % 2**63
        )
        values = infer(batch, batch_seed, n_samples)
        if not np.isfinite(values).all():
            raise ValueError(
                "Nonfinite generated trajectories during throughput measurement"
            )
        generated += len(batch) * (n_samples if stochastic else 1)
    synchronize()
    elapsed = time.perf_counter() - start
    return {
        "mode": "stochastic trajectory draws"
        if stochastic
        else "deterministic trajectory predictions",
        "elapsed_seconds": elapsed,
        "trajectories": generated,
        "trajectories_per_second": generated / elapsed,
        "seconds_per_trajectory": elapsed / generated,
        "input_batch_size": batch_size,
        "warmup": "one input, one trajectory",
        "includes": "input normalization, model inference, output denormalization and CPU result transfer",
        "excludes": "model loading, training and evaluation metric calculation",
        "interpretation": "No simulator runtime measured; this is not an ABM speedup estimate.",
    }


def run_benchmark(
    output,
    *,
    data=None,
    source="bundled SIRDS teaching subset",
    steps=100,
    seed=42,
    n_samples=32,
    num_steps=50,
    solver="heun",
    models=MODEL_FAMILIES,
    device="cpu",
    batch_size=64,
    evaluation_batch_size=16,
    validation_fraction=0.2,
    test_fraction=0.2,
    constraints=None,
    model_options=None,
):
    """Train all selected families on exactly the same scenario partitions.

    Writes portable models, full per-model metrics/provenance, held-out
    trajectories, a summary CSV and an experiment manifest. Failures are recorded
    explicitly and remaining families continue; the CLI exits nonzero if any fail.
    Existing benchmark manifests are never overwritten.
    """
    models = tuple(models)
    if (
        not models
        or len(set(models)) != len(models)
        or set(models) - set(MODEL_FAMILIES)
    ):
        raise ValueError(
            f"models must be a nonempty unique selection from {MODEL_FAMILIES}"
        )
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("n_samples must be at least 2 for distributional evaluation")
    if (
        isinstance(evaluation_batch_size, bool)
        or not isinstance(evaluation_batch_size, int)
        or evaluation_batch_size < 1
    ):
        raise ValueError("evaluation_batch_size must be a positive integer")
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    if solver not in ("euler", "heun"):
        raise ValueError("solver must be 'euler' or 'heun'")
    options = dict(model_options or {})
    if set(options) - set(models):
        raise ValueError("model_options contains an unselected model")
    config = TrainingConfig(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        device=device,
        eval_every=max(1, min(50, steps)),
    )
    data = load_example() if data is None else data
    split = data.split(
        validation_fraction=validation_fraction, test_fraction=test_fraction, seed=seed
    )
    if split.validation is None or split.test is None:
        raise ValueError("Benchmark needs nonempty validation and test partitions")
    destination = Path(output)
    if (destination / "manifest.json").exists():
        raise FileExistsError(
            f"Benchmark already exists at {destination}; choose a new output directory"
        )
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "purpose": LIMITATIONS,
        "source": source,
        "environment": _environment(),
        "models": list(models),
        "seed": seed,
        "dataset": _partition_metadata(data),
        "schema": {
            "inputs": list(data.input_names),
            "outputs": list(data.output_names),
            "times": data.times.tolist(),
        },
        "split": {
            "method": "scenario groups kept together, shared by every family and baseline",
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "train": _partition_metadata(split.train),
            "validation": _partition_metadata(split.validation),
            "test": _partition_metadata(split.test),
        },
        "training_config": asdict(config),
        "sampling": {
            "n_samples": n_samples,
            "seed": seed,
            "num_steps": num_steps,
            "solver": solver,
            "evaluation_batch_size": evaluation_batch_size,
        },
        "constraints": constraints,
        "constraint_interpretation": "Explicit diagnostics only; no outputs are clipped or projected. Nonnegative/nondecreasing use strict zero thresholds; conservation uses the declared atol.",
        "baseline": "Mean original-unit training trajectory, equal weight per training realization; independent of inputs.",
    }
    _write_json(destination / "manifest.json", manifest)
    summary = []
    for name in ("train_mean_baseline", *models):
        folder = destination / name
        folder.mkdir(exist_ok=True)
        row = {"model": name, "status": "failed"}
        start = time.perf_counter()
        try:
            if name == "train_mean_baseline":
                model = MeanTrajectoryBaseline(split.train)
                model_options_resolved, model_config = {}, {}
            else:
                model = Surrogate(name, options.get(name))
                model.fit(split.train, validation=split.validation, config=config)
                model_options_resolved, model_config = (
                    model.resolved_model_options,
                    model.model_config,
                )
            fit_seconds = time.perf_counter() - start
            artifact = folder / (
                "model.json" if name == "train_mean_baseline" else "model.pt"
            )
            model.save(artifact)
            report = evaluate(
                model,
                split.test,
                n_samples=n_samples,
                seed=seed,
                num_steps=num_steps,
                solver=solver,
                constraints=constraints,
                batch_size=evaluation_batch_size,
            )
            report.save(folder / "metrics.json")
            trajectories = {
                "predicted": report.predictions,
                "observed": report.observed,
                "times": report.times,
            }
            if report.lower is not None:
                trajectories.update(lower=report.lower, upper=report.upper)
            np.savez_compressed(folder / "test_trajectories.npz", **trajectories)
            timing = _measure_inference(
                model,
                split.test.inputs,
                n_samples=n_samples,
                seed=seed,
                num_steps=num_steps,
                solver=solver,
                batch_size=evaluation_batch_size,
                device=device,
            )
            run = {
                "model": name,
                "status": "ok",
                "parameter_count": model.parameter_count,
                "fit_seconds": fit_seconds,
                "inference": timing,
                "model_options_resolved": model_options_resolved,
                "model_config": model_config,
                "training_config": asdict(config)
                if name != "train_mean_baseline"
                else None,
                "training_metadata": model.training_metadata,
                "evaluation_fingerprint": split.test.fingerprint,
                "artifact": str(artifact.relative_to(destination)),
            }
            _write_json(folder / "run.json", run)
            row.update(
                status="ok",
                parameter_count=model.parameter_count,
                fit_seconds=fit_seconds,
                inference_mode=timing["mode"],
                trajectories_per_second=timing["trajectories_per_second"],
                seconds_per_trajectory=timing["seconds_per_trajectory"],
                test_fingerprint=split.test.fingerprint,
                **report.metrics,
            )
        except Exception as exc:
            row.update(
                error=f"{type(exc).__name__}: {exc}",
                fit_seconds=time.perf_counter() - start,
            )
            _write_json(
                folder / "failure.json",
                {
                    "model": name,
                    "error": row["error"],
                    "test_fingerprint": split.test.fingerprint,
                },
            )
        summary.append(row)
        print(
            f"{name}: {row['status']}"
            + (
                f" | test RMSE {row['rmse']:.6g}"
                if "rmse" in row
                else f" | {row.get('error', '')}"
            ),
            flush=True,
        )
        columns = list(dict.fromkeys(key for record in summary for key in record))
        with (destination / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(summary)
    manifest["completed"] = True
    manifest["failed_models"] = [
        row["model"] for row in summary if row["status"] != "ok"
    ]
    _write_json(destination / "manifest.json", manifest)
    return {"manifest": manifest, "summary": summary, "output": str(destination)}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--num-steps", type=int, default=50, help="Flow-matching solver steps"
    )
    parser.add_argument("--solver", choices=["euler", "heun"], default="heun")
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_FAMILIES, default=list(MODEL_FAMILIES)
    )
    parser.add_argument(
        "--data",
        "--full-sirds",
        dest="data",
        type=Path,
        help="Explicit SIRDS Parquet bank using I_0,beta,gamma,omega,mu; S_t,I_t,R_t,D_t; t and rep. Default: complete bundled 1,000-group/200-timepoint bank.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument(
        "--model-options",
        type=Path,
        help="JSON mapping selected family names to flat architecture options",
    )
    parser.add_argument(
        "--check-sirds-constraints",
        action="store_true",
        help="Explicitly check nonnegative S,I,R,D; nondecreasing D; total S+I+R+D=1000 with atol=1e-3. Never clips.",
    )
    args = parser.parse_args(argv)
    data = (
        None
        if args.data is None
        else SimulationDataset.from_parquet(
            args.data, inputs=SIRDS_INPUTS, outputs=SIRDS_OUTPUTS
        )
    )
    constraints = None
    if args.check_sirds_constraints:
        constraints = {
            "nonnegative": SIRDS_OUTPUTS,
            "nondecreasing": ["D_t"],
            "conserved_total": {"outputs": SIRDS_OUTPUTS, "value": 1000, "atol": 1e-3},
        }
    model_options = (
        None
        if args.model_options is None
        else json.loads(args.model_options.read_text())
    )
    result = run_benchmark(
        args.output,
        data=data,
        source="bundled SIRDS teaching subset"
        if args.data is None
        else str(args.data.resolve()),
        steps=args.steps,
        seed=args.seed,
        n_samples=args.samples,
        num_steps=args.num_steps,
        solver=args.solver,
        models=args.models,
        device=args.device,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        constraints=constraints,
        model_options=model_options,
    )
    return 1 if result["manifest"]["failed_models"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
