# Changelog

## Unreleased

- Restore the original documentation sequence and explanations, using SIRDS
  throughout the installed-package walkthrough and runnable examples.
- Fold the separate package tutorials into the original Basics and model pages,
  retaining redirects, executed outputs and notebook/Colab support.

- Execute tutorials during the documentation build and publish their plots,
  tables and downloadable notebooks with saved outputs.
- Add Open in Colab links with shared, revision-pinned uv setup.
- Simplify tutorial prose and keep installation and API details in reference pages.

## 0.3.0

- Unify Python and installed Hydra training, including optional warm-up/cosine
  scheduling, precision, compilation, deterministic settings and multiple devices.
- Add resumable optimizer/RNG checkpoints with strict data/configuration checks;
  keep complete inference artifacts separate from training continuation.
- Expose named observed-sequence conditioning for transformer and flow matching.
- Add explicit truncation and initial-state preparation preserving group identity.
- Add strict inference-only historical checkpoint import with explicit original
  preprocessing, provenance and numerical-parity checks.
- Remove root training/data/model compatibility interfaces after migration to the
  installed package, and update scientific tutorials to the current configuration.
- Add a CSV own-bank exercise, a differentiable-likelihood smoke example and an
  automated fresh-project handoff using a built wheel.
- Expand reference, troubleshooting and migration documentation. Scientific
  fidelity and GPU/driver acceptance remain application/hardware-specific.

## 0.2.0

- Introduce the installable metis namespace and metis-surrogate distribution.
- Add validated simulation banks, grouped splits and training-only scaling.
- Add a Python workflow for fitting, evaluating, saving and reusing surrogates.
- Keep Hydra configuration and multirun sweeps as an installed frontend to the
  same public training and evaluation functions.
- Preserve the four existing model families and advanced source training entry.
- Add complete inference bundles and diagnostics in original output units.
- Correct attention-cache behavior and data/plotting edge cases found in review.
- Correct the final Heun integration step; Heun samples change numerically,
  while the flow-matching training objective is unchanged.
- Build and test distributions and documentation in GitHub Actions.
- Use uv projects throughout installation, examples and CI, with a committed
  CPU PyTorch lock and an explicit opt-in GPU source.

This is an early research release. Scientific validation remains specific to
the simulator and intended application. The fork is MIT licensed. It is not
yet published on PyPI; installation from GitHub or a built wheel is supported.
