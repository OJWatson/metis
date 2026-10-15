# Changelog

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

This is an early research release. Scientific validation remains specific to
the simulator and intended application. The fork is MIT licensed. It is not
yet published on PyPI; installation from GitHub or a built wheel is supported.
