# Contributing

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git,
then work from the repository root. Python 3.11+ is supported; the checkout
selects Python 3.13. uv manages Python, the editable project, development tools
and optional dependencies in `.venv`; no manual activation is needed.

~~~bash
git clone https://github.com/OJWatson/metis.git
cd metis
uv sync
uv run python -m pytest
uv run ruff check src tests
uv run ty check src
uv build
~~~

The default development group includes the documentation and notebook tools,
so plain `uv sync` and `uv run` are sufficient, including W&B tracking and the
Pyro inference example. CI uses `--all-extras` to test every optional dependency. `--locked` additionally checks that
`pyproject.toml` agrees with the committed `uv.lock`.

The checkout selects PyTorch's explicit CPU index on Linux and Windows, and the
platform build from PyPI on Apple Silicon macOS. Intel Macs are unsupported by
the current PyTorch dependency range. CPU CI also tests a built wheel outside the
checkout. The committed lockfile records a development resolution, not a promise
that every GPU/driver combination has been tested. See [optional PyTorch setup](#optional-pytorch-setup) for CUDA or CPU-only
installation in a separate research project. The shared training engine supports
CPU or CUDA.

Keep source under `src/metis` and import `metis`. Root compatibility modules
and the former source trainer have been retired; use the installed entry points.
Do not add public APIs that depend on the checkout's working directory.

Test complete user journeys and scientific invariants: group separation,
scaling isolation, run boundaries, serialization parity, original-unit
predictions and active (nonzero) attention. Do not present a passing unit
test as evidence of scientific accuracy.

Documentation is a Quarto website. Install the Quarto CLI separately; the
`--all-extras` environment supplies its Python dependencies. Build the site with:

~~~bash
uv run --locked --all-extras python scripts/build_docs.py
uv run --locked --all-extras python scripts/check_site_links.py quarto/_site
uv run --locked --all-extras python scripts/check_tutorial_notebook.py quarto/_site --outputs-only
~~~

The build executes every tutorial in a temporary copy of the documentation,
then exports notebooks with their plots and results. It fails on execution
errors, unexecuted cells or missing figures. Plot-producing Python must use
executable `{python}` cells; a captioned figure must have a saved image output. Notebook tutorials declare
`metis-notebook: true` and both `html` and `ipynb` formats; reference pages
remain HTML only. Keep examples self-contained, runnable on CPU and free of
external accounts or private data. Use `{python}` cells for runnable code.
Use the full original SIRDS bank throughout (1,000 settings × five realisations
× 200 days), with the same grouped split. Reserve `load_example(small=True)`
for explicitly labelled smoke tests. Extend
the original walkthrough where a topic belongs instead of adding a competing
quickstart or another example data-generating process.

A short shared setup cell is added to downloads for Colab; do not repeat
installation commands in tutorial sources. It checks out the recorded source
commit and runs `uv sync --locked --inexact` against the notebook interpreter.
Every notebook therefore installs the revision that produced its saved outputs. CI publishes executed notebooks to the `notebooks`
branch and deploys the website only after a successful main-branch build.
To rerun one exported notebook from a fresh directory:

~~~bash
uv run --locked --all-extras python scripts/check_tutorial_notebook.py quarto/_site/basics/saving-and-loading.ipynb
~~~

Before a public package-index release, review the notices in
ATTRIBUTION.md, review the dependency range and run GPU smoke tests on the
hardware configurations that will be advertised.

Run the independent SIRDS data-loading and differentiable-likelihood exercises:

```bash
uv run --locked --all-extras python examples/own_bank.py --steps 20 --output /tmp/metis-own-bank
uv run --locked --all-extras python examples/surrogate_likelihood.py --steps 20 --output /tmp/metis-likelihood
```

After `uv build`, run `uv run --locked --all-extras python scripts/check_beginner_handoff.py --wheel dist/metis_surrogate-0.3.0-py3-none-any.whl`.
This creates a fresh uv project, installs the wheel, creates and inspects a bank,
trains, evaluates and reloads the artifact, then checks conditioned Hydra training
and checkpoint resume. It checks that imports come from that
project's environment. It is an automated handoff test, not an observed human
usability trial.

Validate advertised CUDA hardware separately with
`uv run --locked --all-extras python scripts/check_hardware.py --device cuda --num-devices 2`.
Use the actual device count; retain the generated report with release evidence.
CPU CI is not a substitute for this check.

## Optional PyTorch setup

The checkout already uses the CPU index on Linux and Windows. In a separate
research project, `uv add` uses PyPI's platform build by default; on Linux this
includes CUDA libraries and can be a much larger download. A CPU-only build
can be selected explicitly:

```bash
uv add "torch>=2.10,<3" --index pytorch-cpu=https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU, choose a build compatible with your driver using the
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
For example, CUDA 12.8:

```bash
uv add "torch>=2.10,<3" --index pytorch-cu128=https://download.pytorch.org/whl/cu128
```

For either named index, add `explicit = true` to its `[[tool.uv.index]]` entry
in `pyproject.toml`, so unrelated dependencies continue to come from PyPI.
Run `uv sync` to update the lockfile and environment. These index choices are
for Linux and Windows; on Apple Silicon, use the default PyPI build.

Check GPU availability with:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Use `TrainingConfig(device="cuda")` in Python or
`uv run metis train training.device=cuda` with Hydra. Installing PyTorch does
not install or update the NVIDIA driver.
