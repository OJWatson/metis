# METIS

Train neural surrogates from a bank of simulator runs, check their fidelity,
and reuse them in fitting scripts, notebooks or dashboards.

METIS supplies one library API with two entry points: Python objects for
interactive/application use, and Hydra configurations for reproducible jobs,
overrides and sweeps. Both use the same training, evaluation and artifact code.

[Documentation](https://ojwatson.github.io/metis/) · [Examples](examples/package_quickstart.py) · [Changelog](CHANGELOG.md)

## Install

Python 3.11+ is supported. The default library workflow runs on CPU; a compatible
CUDA PyTorch installation can be used with `TrainingConfig(device="cuda")`.
Install a PyTorch build suitable for your hardware, then install this fork:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "metis-surrogate @ git+https://github.com/OJWatson/metis.git"
metis demo --steps 100 --output metis-demo
```

The distribution is named **metis-surrogate** and the Python import is **metis**.
It is not yet on PyPI. `pip install metis` installs an unrelated graph library.
Use an isolated environment if you already use that library. Optional extras:
`[plots]`, `[pandas]`, `[wandb]`, `[docs]`, and `[legacy]` for the historical CUDA trainer.

## A complete Python workflow

```python
from metis import SimulationDataset, Surrogate, TrainingConfig

bank = SimulationDataset.from_parquet(
    "simulations.parquet",
    inputs=["contact_rate", "recovery_rate"],
    outputs=["incidence", "prevalence"],
    # groups defaults to the input columns; time="t", replicate="rep"
)
split = bank.split(validation_fraction=0.2, test_fraction=0.2, seed=42)

surrogate = Surrogate("mlp").fit(
    split.train,
    validation=split.validation,
    config=TrainingConfig(steps=1000),
)
report = surrogate.evaluate(split.test)
print(report.by_output)
report.save("evaluation.json")
surrogate.save("surrogate.pt")

# In another process, without the simulation bank:
restored = Surrogate.load("surrogate.pt")
predictions = restored.predict({"contact_rate": 0.4, "recovery_rate": 0.1})
# Shape: (input settings, timepoints, output variables), in original units.
```

To try this without your own data, use `from metis import load_example` and
`bank = load_example()`. It loads a bundled subset of the SIRDS example.
For pandas/Polars tables use `SimulationDataset.from_frame(...)`; existing
numeric arrays can use `SimulationDataset.from_arrays(...)`.

## The same workflow with Hydra

```bash
metis train model=mlp training.steps=1000
metis train --multirun model=mlp,transformer training.learning_rate=0.001,0.0003 evaluation.partition=validation
metis train dataset=parquet dataset.path=/absolute/path/simulations.parquet 'dataset.inputs=[contact_rate,recovery_rate]' 'dataset.outputs=[incidence,prevalence]'
```

`metis-train` is an equivalent entry point. Use `metis train --cfg job` to inspect
the composed configuration or `metis train --help` for groups and overrides.
Every run saves a complete `model.pt`, metrics and training history in its Hydra
output directory. The Python and Hydra workflows use the same `Surrogate`.
Use validation metrics to choose settings, then evaluate the chosen model on the
test bank. Optional `wandb.enabled=true` streams training metrics with the `[wandb]`
extra; `wandb.mode=offline` keeps tracking local without an account. Python users
can use `metis.tracking.WandbTracker` with the same training progress callback.
Tracking logs configuration and scalar metrics; it does not upload simulation
tables or model files automatically.

The old `uv run train.py --config-name default ...` workflow remains available
from the checkout with `[legacy]` dependencies. It exposes advanced CUDA/DDP and
sequence-conditioning settings. Its historical checkpoint format differs from
the new portable artifacts; use the new entry points for ordinary static-input
training and deployment.

## Your simulation bank

Use one row per realization and timepoint. The table must contain numeric static
inputs, numeric outputs, `rep` and `t`. Inputs are constant within a run. All runs
must have the same ordered time grid; METIS validates lengths, missing values,
duplicate identities and static inputs. It does not interpolate or silently
remove data. Raw agent/event logs must first be aggregated into trajectories.

Parameter groups stay together across splits. Scaling is fitted on training data
only and is saved with the model. Constant dimensions use a scale of one and
remain identifiable through the scaler's constant mask. For pre-split banks,
`DatasetSplit.from_datasets(train, validation, test)` checks compatibility and
rejects overlapping scenarios.

## Which model?

| Family | Prediction | Typical purpose |
|---|---|---|
| `mlp` | Deterministic whole trajectory | Begin with a simple mean-prediction baseline |
| `transformer` | Deterministic autoregressive rollout | Sequential prediction; advanced sequence conditioning |
| `flow_matching` | Stochastic trajectories | Reproduce simulator variability |
| `itarflow` | Stochastic trajectories and continuous density | Approximate-likelihood applications |

For stochastic models, `surrogate.sample(inputs, n_samples=100, seed=42)` returns
shape `(inputs, samples, time, outputs)`. An explicit
`surrogate.predict(inputs, n_samples=100)` returns a Monte Carlo mean.
`itarflow` also provides `log_prob` and differentiable `log_prob_tensor` in
original units. This is the learned surrogate's density, not the exact simulator
likelihood. The high-level API currently accepts static inputs; complete-sequence
conditioning remains in `metis.models` and the advanced trainer.

## Check fidelity and decide where to simulate

Evaluation reports original-unit MAE/RMSE by output and parameter group, error
against replicate means, and stochastic CRPS and predictive-interval coverage.
`report.plot()` requires `[plots]`. Scientific constraints are explicitly supplied
by the user; checks report violations without altering predictions.

```python
from metis import assess_coverage, suggest_simulations

print(report.worst_groups(5))
print(assess_coverage(split.train, candidate_inputs))
next_settings = suggest_simulations(split.train, candidate_inputs, n=10)
```

Candidates must already be scientifically valid simulator settings. Suggestions
use greedy geometric coverage in input space, not posterior uncertainty or a
guarantee of model error. Inspect observed errors as well. If you choose more
simulations using test results, reserve a fresh independent test bank.

Generated variability does not quantify surrogate approximation error. Models do
not automatically enforce counts, positivity or conservation. Compare against
simple baselines and check the outcomes important for your scientific decision.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). CI tests the source and installed wheel,
executes a complete beginner example, and builds the documentation. GPU/driver
coverage is separate from the CPU test matrix. Benchmarks are reproducible with
`python benchmarks/compare_models.py`; a short run verifies the workflow and
is not a universal architecture ranking.

MIT licensed; original implementation by Timothy Hitge and subsequent METIS
contributors. See [ATTRIBUTION.md](ATTRIBUTION.md) and
[third-party notices](THIRD_PARTY_NOTICES.txt).
