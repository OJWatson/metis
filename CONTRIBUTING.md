# Contributing

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git,
then work from the repository root. Python 3.11+ is supported; the checkout
selects Python 3.13. uv manages Python, the editable project, development tools
and optional dependencies in `.venv`; no manual activation is needed.

~~~bash
git clone https://github.com/OJWatson/metis.git
cd metis
uv sync --locked --all-extras
uv run --locked --all-extras pytest
uv run --locked --all-extras ruff check src tests
uv run --locked --all-extras ty check src
uv build
~~~

Keep `--all-extras` on checkout `uv run` commands so synchronization retains the
notebook, plotting, tracking and pandas dependencies. `--locked` checks
that `pyproject.toml` agrees with the committed `uv.lock` instead of silently
changing the resolution. The development dependency group is included by default.

The checkout selects PyTorch's explicit CPU index on Linux and Windows, and the
platform build from PyPI on Apple Silicon macOS. Intel Macs are unsupported by
the current PyTorch dependency range. CPU CI also tests a built wheel outside the
checkout. The committed lockfile records a development resolution, not a promise
that every GPU/driver combination has been tested. For CUDA, follow the
[named-index setup](https://ojwatson.github.io/metis/package/quickstart.html#optional-cuda-setup)
to change your local torch source and lockfile, then run
`uv sync --locked --all-extras` again. The shared training engine supports CPU or CUDA.

Keep source under `src/metis` and import `metis`. Root compatibility modules
and the former source trainer have been retired; use the installed entry points.
Do not add public APIs that depend on the checkout's working directory.

Test complete user journeys and scientific invariants: group separation,
scaling isolation, run boundaries, serialization parity, original-unit
predictions and active (nonzero) attention. Do not present a passing unit
test as evidence of scientific accuracy.

Documentation is a Quarto website. Install the Quarto CLI separately; the
`--all-extras` setup above supplies its Python dependencies. Run the CLI through
the project environment and then check links and execute the beginner notebook:

~~~bash
uv run --locked --all-extras quarto render quarto --to all --no-execute
uv run --locked --all-extras python scripts/check_site_links.py quarto/_site
uv run --locked --all-extras python scripts/check_tutorial_notebook.py quarto/_site/package/quickstart.ipynb
uv run --locked --all-extras python examples/package_quickstart.py --steps 20 --output /tmp/metis-example
~~~

The beginner example and generated notebook are executed in CI separately from site rendering.
Scientific background pages use illustrative snippets; their mathematical exposition
does not imply that posterior inference or large training jobs were executed.
The website is deployed by the documentation workflow after a push to main.

Before a public package-index release, review the notices in
ATTRIBUTION.md, review the dependency range and run GPU smoke tests on the
hardware configurations that will be advertised.

Run the independent own-bank and differentiable-likelihood exercises:

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
