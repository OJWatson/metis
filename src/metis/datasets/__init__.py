"""Bundled examples; no network or source-checkout dependency."""

from importlib.resources import as_file, files


def load_example(*, small=False):
    """Load the original SIRDS bank: 1,000 settings, five replicates, 200 days.

    The complete bank is bundled with the installed package. Use ``small=True``
    only for a fast installation check or test fixture: this selects the
    packaged 100-setting, 40-day subset, retaining all five replicates.
    """
    from metis.data import SimulationDataset

    if not isinstance(small, bool):
        raise TypeError("small must be a boolean")
    resource = "sirds_small.parquet" if small else "sirds.parquet"
    with as_file(files("metis.datasets").joinpath(resource)) as path:
        return SimulationDataset.from_parquet(
            path,
            inputs=["I_0", "beta", "gamma", "omega", "mu"],
            outputs=["S_t", "I_t", "R_t", "D_t"],
        )
