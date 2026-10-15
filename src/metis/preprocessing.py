"""Explicit transformations of already validated simulation banks.

Load a Polars/pandas table with ``SimulationDataset.from_frame`` first. These
helpers never remove duplicate runs, interpolate observations, reset time units,
or choose grouping identities. Record their arguments with an experiment's
configuration; each transformed dataset has its own content fingerprint.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np

from metis.data import SimulationDataset


def truncate_simulations(
    dataset: SimulationDataset, *, start_time: float
) -> SimulationDataset:
    """Retain observed times greater than or equal to an explicit boundary.

    A boundary between observations selects the next observed time; it does not
    interpolate a new starting value. Inputs and original scenario/replicate
    identities remain unchanged. Use :func:`initialize_static_inputs` explicitly
    afterwards if the first retained output should become a static input.
    """
    if not isinstance(dataset, SimulationDataset):
        raise TypeError("dataset must be a validated SimulationDataset")
    if (
        isinstance(start_time, bool)
        or not isinstance(start_time, Real)
        or not np.isfinite(start_time)
    ):
        raise ValueError("start_time must be a finite numeric boundary")
    keep = dataset.times >= start_time
    if not np.any(keep):
        raise ValueError(
            "start_time removes every observation; choose an earlier boundary"
        )
    return SimulationDataset(
        dataset.inputs,
        dataset.outputs[:, keep],
        dataset.input_names,
        dataset.output_names,
        dataset.times[keep],
        dataset.group_keys,
        dataset.replicate_ids,
    )


def initialize_static_inputs(
    dataset: SimulationDataset, *, mapping: Mapping[str, str]
) -> SimulationDataset:
    """Copy first observed outputs into explicitly named static input columns.

    ``mapping={"initial_infected": "infected"}`` replaces that input if present
    or appends it otherwise. Values are copied per realization. Original group
    keys are retained, so changing replicate-specific initial conditions cannot
    accidentally divide one scenario across training and validation partitions.
    Output trajectories are unchanged; no value is normalized or clipped.
    """
    if not isinstance(dataset, SimulationDataset):
        raise TypeError("dataset must be a validated SimulationDataset")
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError(
            "mapping must name at least one target input and source output"
        )
    for target, source in mapping.items():
        if not isinstance(target, str) or not target:
            raise ValueError("Target input names must be nonempty strings")
        if target in dataset.output_names:
            raise ValueError(f"Target input {target!r} cannot also be an output name")
        if source not in dataset.output_names:
            raise ValueError(
                f"Unknown source output {source!r}; choose from {dataset.output_names}"
            )
    names = list(dataset.input_names)
    columns = [dataset.inputs[:, i].copy() for i in range(len(names))]
    for target, source in mapping.items():
        column = dataset.outputs[:, 0, dataset.output_names.index(source)].copy()
        if target in names:
            columns[names.index(target)] = column
        else:
            names.append(target)
            columns.append(column)
    return SimulationDataset(
        np.column_stack(columns),
        dataset.outputs,
        tuple(names),
        dataset.output_names,
        dataset.times,
        dataset.group_keys,
        dataset.replicate_ids,
    )
