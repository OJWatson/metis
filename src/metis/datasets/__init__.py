"""Bundled examples; no network or source-checkout dependency."""

from importlib.resources import as_file, files


def load_example():
    """Return 100 SIRDS parameter groups, five replicates and 40 timepoints.

    The subset is for tutorials and installation checks. It is not evidence
    that a surrogate trained on it is adequate for a scientific application.
    """
    from metis.data import SimulationDataset

    with as_file(files("metis.datasets").joinpath("sirds_small.parquet")) as path:
        return SimulationDataset.from_parquet(
            path,
            inputs=["I_0", "beta", "gamma", "omega", "mu"],
            outputs=["S_t", "I_t", "R_t", "D_t"],
        )
