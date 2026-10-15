# Contributing

Use Python 3.11 or newer. Install a PyTorch build appropriate to your device,
then install the editable project and its development/legacy dependencies.
The new public API defaults to CPU; the historical Hydra trainer uses CUDA.

~~~bash
uv sync --all-extras
uv run pytest
uv run ruff check src tests
uv build
~~~

The committed uv.lock records a development resolution, not a promise that
every GPU/driver combination has been tested. CPU CI also tests installation
from a built wheel outside the checkout. PyTorch's CPU wheel index can be used
in a separate environment to avoid CUDA downloads:

~~~bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[legacy,docs]"
python -m pip install pytest build
~~~

Keep source under src/metis. The root models directory is a compatibility
layer for old source-checkout recipes. New code should import metis.
Do not add public APIs that depend on the checkout's working directory.

Test complete user journeys and scientific invariants: group separation,
scaling isolation, run boundaries, serialization parity, original-unit
predictions and active (nonzero) attention. Do not present a passing unit
test as evidence of scientific accuracy.

Documentation is a Quarto website. Install the Quarto CLI separately, install
the docs extra, and run:

~~~bash
quarto render quarto --to all --no-execute
python scripts/check_site_links.py quarto/_site
python examples/package_quickstart.py --steps 20 --output /tmp/metis-example
~~~

The beginner example is executed in CI separately from site rendering.
Advanced historical pages may intentionally disable costly training cells.
The website is deployed by the documentation workflow after a push to main.

Before a public package-index release, review the notices in
ATTRIBUTION.md, review the dependency range and run GPU smoke tests on the
hardware configurations that will be advertised.
