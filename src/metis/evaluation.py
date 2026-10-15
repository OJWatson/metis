"""Original-unit diagnostics shared across surrogate families.

Prediction intervals describe the fitted generator's variability, not
confidence in its fidelity. No metric here establishes causal validity.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
from pathlib import Path
from numbers import Real
import warnings

import numpy as np
import polars as pl


@dataclass
class EvaluationReport:
    metrics: dict
    by_output: pl.DataFrame
    by_group: pl.DataFrame
    metadata: dict
    predictions: np.ndarray = field(repr=False)
    observed: np.ndarray = field(repr=False)
    times: np.ndarray = field(repr=False)
    lower: np.ndarray | None = field(default=None, repr=False)
    upper: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self):
        """JSON-friendly summary; arrays remain available separately."""
        return {
            "metrics": self.metrics,
            "by_output": self.by_output.to_dicts(),
            "by_group": self.by_group.to_dicts(),
            "metadata": self.metadata,
        }

    def save(self, path):
        """Write metrics and provenance as JSON, without a model dependency."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False))

    def worst_groups(self, n=10):
        """Return groups with largest original-unit RMSE for investigation.

        Outputs with large units dominate aggregate RMSE; inspect by_output
        and choose scientific thresholds appropriate to each output.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        return self.by_group.sort("rmse", descending=True).head(n)

    def plot(self, output=None, max_runs=4):
        """Plot observed runs, predictions and optional predictive intervals."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Plotting requires: uv add "
                "'metis-surrogate[plots] @ git+https://github.com/OJWatson/metis.git'"
            ) from exc
        names = self.by_output["output"].to_list()
        if output is not None and output not in names:
            raise ValueError(f"Unknown output {output!r}; choose from {names}")
        if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 1:
            raise ValueError("max_runs must be a positive integer")
        columns = range(len(names)) if output is None else [names.index(output)]
        rows = min(max_runs, len(self.observed))
        fig, axes = plt.subplots(
            rows, len(columns), squeeze=False, figsize=(5 * len(columns), 2.6 * rows)
        )
        for row in range(rows):
            for col, idx in enumerate(columns):
                ax = axes[row, col]
                ax.plot(self.times, self.observed[row, :, idx], label="Simulator")
                ax.plot(
                    self.times, self.predictions[row, :, idx], label="Surrogate mean"
                )
                if self.lower is not None and self.upper is not None:
                    ax.fill_between(
                        self.times,
                        self.lower[row, :, idx],
                        self.upper[row, :, idx],
                        alpha=0.2,
                        label="90% surrogate predictive interval",
                    )
                ax.set(xlabel="Simulator time", ylabel=names[idx])
                ax.legend(fontsize="small")
        identity = self.metadata.get("model_fingerprint")
        fig.suptitle(
            f"{self.metadata['model']} | model {identity[:12] if identity else 'identity unavailable'}\n"
            f"evaluation bank {self.metadata['dataset_fingerprint'][:12]} | "
            f"trained on {str(self.metadata.get('training_fingerprint', 'unknown'))[:12]}"
        )
        fig.tight_layout()
        return fig


def evaluate(
    surrogate,
    data,
    *,
    n_samples=100,
    seed=42,
    num_steps=50,
    solver="heun",
    constraints=None,
    batch_size=64,
):
    """Evaluate every supplied run in original units using autonomous predictions.

    ``batch_size`` limits the number of runs whose samples are held in memory;
    it never subsamples the evaluation bank. For stochastic models the first
    batch uses ``seed`` and later seeds derive from ``(seed, batch_start)`` via
    NumPy SeedSequence. Changing batch size changes Monte Carlo draws. The
    resulting intervals are pointwise 90% predictive intervals, not simultaneous
    trajectory bands or confidence intervals for surrogate fidelity. CRPS is
    the score of the finite empirical ensemble, with denominator n_samples**2.

    Constraints are explicit: nonnegative=[names], nondecreasing=[names],
    conserved_total={"outputs": [names], "value": number, "atol": number}.
    They report violation fractions and never alter predictions. No physical
    constraint is inferred from column names. Group summary input columns use
    ``input:<name>`` to avoid collisions with diagnostic names such as ``rmse``.

    Reports snapshot the resolved architecture and recorded training settings.
    Public Surrogate models also receive a SHA-256 identity over architecture,
    input/output schema, time grid, fitted scalers and named state tensors. It
    is stable across artifact save/load and excludes evaluation sampling settings
    and training history. The hash identifies the fitted model state, not proof
    of scientific validity; software versions are reported separately.
    """
    if tuple(data.input_names) != tuple(surrogate.input_names):
        raise ValueError("Evaluation input columns/order differ from the fitted schema")
    if tuple(data.output_names) != tuple(surrogate.output_names):
        raise ValueError(
            "Evaluation output columns/order differ from the fitted schema"
        )
    if not np.array_equal(data.times, surrogate.times):
        raise ValueError("Evaluation time grid differs from the fitted model")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    stochastic = surrogate.model_name in {"flow_matching", "itarflow"}
    if stochastic:
        if (
            isinstance(n_samples, bool)
            or not isinstance(n_samples, int)
            or n_samples < 2
        ):
            raise ValueError("Stochastic evaluation requires n_samples >= 2")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer between 0 and 2**63 - 1")
    specification = _validated_constraints(data.output_names, constraints)
    truth = np.asarray(data.outputs, dtype=np.float64)
    if truth.ndim != 3 or not truth.size or not np.isfinite(truth).all():
        raise ValueError(
            "Observed outputs must be a nonempty finite (run, time, output) array"
        )
    prediction = np.empty_like(truth)
    lower = np.empty_like(truth) if stochastic else None
    upper = np.empty_like(truth) if stochastic else None
    coverage_sum = np.zeros(truth.shape[-1])
    width_sum = np.zeros(truth.shape[-1])
    crps_sum = np.zeros(truth.shape[-1])
    physical = {}
    for start in range(0, len(truth), batch_size):
        stop = min(start + batch_size, len(truth))
        observed = truth[start:stop]
        if stochastic:
            assert lower is not None and upper is not None
            batch_seed = (
                seed
                if start == 0
                else int(
                    np.random.SeedSequence([seed, start]).generate_state(
                        1, dtype=np.uint64
                    )[0]
                )
                % 2**63
            )
            values = np.asarray(
                surrogate.sample(
                    data.inputs[start:stop],
                    n_samples=n_samples,
                    seed=batch_seed,
                    num_steps=num_steps,
                    solver=solver,
                ),
                dtype=np.float64,
            )
            expected = (stop - start, n_samples, *truth.shape[1:])
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(
                    f"Generated samples must be finite and have shape {expected}"
                )
            prediction[start:stop] = values.mean(axis=1)
            lo, hi = np.quantile(values, [0.05, 0.95], axis=1)
            lower[start:stop], upper[start:stop] = lo, hi
            coverage_sum += ((observed >= lo) & (observed <= hi)).sum(axis=(0, 1))
            width_sum += (hi - lo).sum(axis=(0, 1))
            # Empirical CRPS = E|X-y| - .5 E|X-X'|. Sorting avoids an S x S
            # sample-pair tensor and keeps memory linear in ensemble size.
            ordered = np.sort(values, axis=1)
            k = np.arange(1, n_samples, dtype=float)
            # Adjacent spacings avoid cancellation from a large common offset
            # in the original output units.
            pair_term = (
                (k * (n_samples - k))[None, :, None, None] * np.diff(ordered, axis=1)
            ).sum(axis=1) / n_samples**2
            crps = np.abs(values - observed[:, None]).mean(axis=1) - pair_term
            crps_sum += crps.sum(axis=(0, 1))
        else:
            values = np.asarray(
                surrogate.predict(data.inputs[start:stop]), dtype=np.float64
            )
            if values.shape != observed.shape or not np.isfinite(values).all():
                raise ValueError("Predictions have invalid shape or non-finite values")
            prediction[start:stop] = values
        # Every batch has the same time grid, outputs, and sample count, so run
        # counts provide exact weights for aggregating violation fractions.
        for name, fraction in _check_constraints(
            values, data.output_names, specification
        ).items():
            physical[name] = physical.get(name, 0.0) + fraction * (stop - start) / len(
                truth
            )
    error = prediction - truth
    output_rmse = np.sqrt(np.mean(error**2, axis=(0, 1)))
    output_mae = np.mean(np.abs(error), axis=(0, 1))
    metrics = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }
    records = [
        {"output": name, "rmse": float(output_rmse[i]), "mae": float(output_mae[i])}
        for i, name in enumerate(data.output_names)
    ]
    if stochastic:
        count = len(truth) * truth.shape[1]
        coverage, width, crps = (
            coverage_sum / count,
            width_sum / count,
            crps_sum / count,
        )
        metrics.update(
            predictive_coverage_90=float(coverage.mean()),
            predictive_interval_width_90=float(width.mean()),
            crps=float(crps.mean()),
        )
        for i, row in enumerate(records):
            row.update(
                predictive_coverage_90=float(coverage[i]),
                predictive_interval_width_90=float(width[i]),
                crps=float(crps[i]),
            )
    groups = {}
    for i, key in enumerate(data.group_keys):
        groups.setdefault(tuple(key), []).append(i)
    group_records = []
    group_mean_errors = []
    for key, indices in groups.items():
        idx = np.asarray(indices)
        row: dict[str, str | int | float | None] = {
            "group": json.dumps(key),
            "n_replicates": len(idx),
            "rmse": float(np.sqrt(np.mean(error[idx] ** 2))),
            "mae": float(np.abs(error[idx]).mean()),
        }
        identical_inputs = np.all(data.inputs[idx] == data.inputs[idx[0]])
        if identical_inputs:
            mean_error = prediction[idx].mean(axis=0) - truth[idx].mean(axis=0)
            row["replicate_mean_rmse"] = float(np.sqrt(np.mean(mean_error**2)))
            group_mean_errors.append(np.mean(mean_error**2))
            row.update(
                {
                    f"input:{name}": float(value)
                    for name, value in zip(data.input_names, data.inputs[idx[0]])
                }
            )
        else:
            row["replicate_mean_rmse"] = None
            row.update({f"input:{name}": None for name in data.input_names})
        group_records.append(row)
    if group_mean_errors:
        metrics["replicate_mean_rmse"] = float(np.sqrt(np.mean(group_mean_errors)))
    training = surrogate.training_metadata
    trained_groups = {tuple(k) for k in training.get("train_group_keys", [])}
    tuned_groups = {tuple(k) for k in training.get("validation_group_keys", [])}
    overlaps = {
        "training_group_overlap": len(set(groups) & trained_groups),
        "tuning_group_overlap": len(set(groups) & tuned_groups),
    }
    if overlaps["training_group_overlap"] or overlaps["tuning_group_overlap"]:
        warnings.warn(
            "This bank overlaps training or tuning groups; these are not independent test results.",
            UserWarning,
            stacklevel=2,
        )
    metadata = {
        "model": surrogate.model_name,
        **_model_provenance(surrogate),
        "dataset_fingerprint": data.fingerprint,
        "training_fingerprint": training.get("training_fingerprint"),
        "validation_fingerprint": training.get("validation_fingerprint"),
        "n_runs": len(truth),
        "n_groups": len(groups),
        "prediction": "Monte Carlo mean"
        if stochastic
        else "autonomous deterministic prediction",
        "n_samples": n_samples if stochastic else None,
        "seed": seed if stochastic else None,
        "batch_size": batch_size,
        "sampling_seed_scheme": "first batch: seed; subsequent batches: SeedSequence([seed, start]) uint64 modulo 2**63"
        if stochastic
        else None,
        "flow_steps": num_steps if surrogate.model_name == "flow_matching" else None,
        "flow_solver": solver if surrogate.model_name == "flow_matching" else None,
        "units": "original output units; aggregate RMSE/MAE weight every scalar equally",
        "replicate_mean_weighting": "equal weight per group with identical replicate inputs; other groups excluded from this metric",
        "uncertainty": "Pointwise predictive variability under the fitted surrogate; no surrogate-error confidence intervals or simultaneous trajectory bands.",
        "crps_estimator": "finite empirical ensemble CRPS (pair denominator n_samples squared)"
        if stochastic
        else None,
        "overlap_check": "group identifiers only; absence of overlap does not establish statistical independence",
        "constraint_definitions": specification,
        "constraints": physical,
        **overlaps,
    }
    return EvaluationReport(
        metrics,
        pl.DataFrame(records),
        pl.DataFrame(group_records, infer_schema_length=None),
        metadata,
        prediction,
        truth,
        data.times,
        lower,
        upper,
    )


def _plain_provenance(value):
    """Copy metadata into JSON-compatible values without retaining live aliases."""
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_provenance(asdict(value))
    if isinstance(value, np.ndarray):
        return _plain_provenance(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain_provenance(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_provenance(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported model provenance value: {type(value).__name__}")


def _model_provenance(surrogate):
    """Identify the actual evaluated model, with explicit gaps for test doubles."""
    import torch
    from metis import __version__

    training = surrogate.training_metadata
    history = getattr(surrogate, "history", None)
    completed_steps = training.get("completed_steps")
    if completed_steps is None and history is not None:
        completed_steps = len(history)
    best_step = training.get("best_step")
    selected_step = best_step if best_step is not None else completed_steps
    selected_record = next(
        (record for record in (history or []) if record.get("step") == selected_step),
        None,
    )
    result = {
        "resolved_model_options": _plain_provenance(
            getattr(surrogate, "resolved_model_options", None)
        ),
        "model_config": _plain_provenance(getattr(surrogate, "model_config", None)),
        "parameter_count": getattr(surrogate, "parameter_count", None),
        # Recorded fit configuration takes precedence over the artifact loader's
        # current inference device/configuration.
        "training_config": _plain_provenance(
            training.get("config", getattr(surrogate, "training_config", None))
        ),
        "training_summary": _plain_provenance(
            {
                "completed_steps": completed_steps,
                "best_step": best_step,
                "best_validation_loss": training.get("best_validation_loss"),
                "stopped_early": training.get("stopped_early"),
                "selected_step": selected_step,
                "selected_record": selected_record,
                "final_record": history[-1] if history else None,
            }
        ),
        "software_versions": {
            "metis": __version__,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
        "model_fingerprint": None,
        "model_fingerprint_schema": "metis-model-v1",
        "model_fingerprint_status": "unavailable: architecture, fitted scalers and neural state are required",
    }
    model = getattr(surrogate, "model", None)
    input_scaler = getattr(surrogate, "input_scaler", None)
    output_scaler = getattr(surrogate, "output_scaler", None)
    if (
        not isinstance(model, torch.nn.Module)
        or result["model_config"] is None
        or result["resolved_model_options"] is None
        or input_scaler is None
        or output_scaler is None
    ):
        return result

    digest = hashlib.sha256()

    def add(payload):
        # Length framing avoids ambiguity between adjacent records.
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    descriptor = {
        "schema": result["model_fingerprint_schema"],
        "family": surrogate.model_name,
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_config": result["model_config"],
        "resolved_model_options": result["resolved_model_options"],
        "input_names": list(surrogate.input_names),
        "output_names": list(surrogate.output_names),
        "times": _plain_provenance(surrogate.times),
        "input_scaler": _plain_provenance(input_scaler.to_dict()),
        "output_scaler": _plain_provenance(output_scaler.to_dict()),
    }
    add(
        json.dumps(
            descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )
    # Hash each tensor individually, including persistent buffers, without a
    # second complete serialized copy of a potentially large state dictionary.
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        add(
            json.dumps(
                {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        # Viewing bytes also supports bfloat16, which NumPy cannot represent.
        raw = tensor.reshape(-1).view(torch.uint8).numpy()
        add(memoryview(raw))
    result["model_fingerprint"] = digest.hexdigest()
    result["model_fingerprint_status"] = (
        "complete: architecture, schema, time grid, fitted scalers and named state tensors"
    )
    return result


def _validated_constraints(names, constraints):
    if constraints is None:
        return {}
    if not isinstance(constraints, dict):
        raise ValueError("constraints must be a dictionary of explicit checks")
    permitted = {"nonnegative", "nondecreasing", "conserved_total"}
    if set(constraints) - permitted:
        raise ValueError(f"Unknown constraints: {set(constraints) - permitted}")

    def output_list(values, label):
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{label} must be a nonempty list of output names")
        if any(not isinstance(name, str) or name not in names for name in values):
            raise ValueError(
                f"{label} refers to an unknown output; choose from {list(names)}"
            )
        if len(set(values)) != len(values):
            raise ValueError(f"{label} contains duplicate outputs")
        return list(values)

    result = {}
    for kind in ("nonnegative", "nondecreasing"):
        if kind in constraints:
            result[kind] = output_list(constraints[kind], kind)
    if "conserved_total" in constraints:
        config = constraints["conserved_total"]
        if not isinstance(config, dict) or set(config) != {"outputs", "value", "atol"}:
            raise ValueError(
                "conserved_total requires exactly outputs, value and explicit atol"
            )
        outputs = output_list(config["outputs"], "conserved_total outputs")
        if any(
            isinstance(config[k], bool)
            or not isinstance(config[k], Real)
            or not np.isfinite(config[k])
            for k in ("value", "atol")
        ):
            raise ValueError("conserved_total value and atol must be finite numbers")
        if config["atol"] < 0:
            raise ValueError("conserved_total requires an explicit nonnegative atol")
        result["conserved_total"] = {
            "outputs": outputs,
            "value": float(config["value"]),
            "atol": float(config["atol"]),
        }
    return result


def _check_constraints(values, names, constraints):
    result = {}
    for kind in ("nonnegative", "nondecreasing"):
        for name in constraints.get(kind, []):
            idx = list(names).index(name)
            vals = values[..., idx]
            bad = vals < 0 if kind == "nonnegative" else np.diff(vals, axis=-1) < 0
            result[f"{kind}:{name}:violation_fraction"] = (
                float(bad.mean()) if bad.size else 0.0
            )
    if "conserved_total" in constraints:
        config = constraints["conserved_total"]
        idx = [list(names).index(n) for n in config["outputs"]]
        bad = np.abs(values[..., idx].sum(axis=-1) - config["value"]) > config["atol"]
        result["conserved_total:violation_fraction"] = float(bad.mean())
    return result
