"""Validated simulation banks, scenario-wise partitions, and fitted scaling.

Rows supplied to :meth:`SimulationDataset.from_frame` describe one observation
time from one simulation realization. Every realization must have the same
numeric time grid; a shared irregular grid is accepted without interpolation.
The neural models use grid positions, not physical elapsed time.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl


def _numeric_array(value: Any, name: str, *, dtype=None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain numeric values, not {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains missing, NaN, or infinite values.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(array, dtype=dtype, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains values outside the supported numeric range.")
    return array


def _identity(value: Any) -> str | int | float | bool:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (str, int, float, bool)) or (
        isinstance(value, float) and not np.isfinite(value)
    ):
        raise ValueError(
            "Group and replicate identities must be non-missing strings or finite numbers."
        )
    return value


def _names(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{label} must be a sequence of column names, not one string.")
    result = tuple(values)
    if not result or any(not isinstance(x, str) or not x for x in result):
        raise ValueError(f"{label} must contain at least one nonempty column name.")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate column names.")
    return result


@dataclass(frozen=True)
class SimulationDataset:
    """A complete simulation bank in original units.

    ``inputs`` has shape ``(realizations, parameters)`` and ``outputs`` has shape
    ``(realizations, times, channels)``. ``group_keys`` identifies scenarios whose
    replicates must stay together when splitting. With no explicit groups,
    identical input tuples define a scenario. Supply explicit groups when
    initial conditions vary by replicate within a shared parameter scenario.
    Arrays are copied and made read-only to keep provenance stable.
    """

    inputs: np.ndarray
    outputs: np.ndarray
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    times: np.ndarray
    group_keys: tuple[tuple, ...]
    replicate_ids: tuple

    def __post_init__(self) -> None:
        inputs = _numeric_array(self.inputs, "inputs", dtype=np.float32)
        outputs = _numeric_array(self.outputs, "outputs", dtype=np.float32)
        times = _numeric_array(self.times, "times")
        input_names = _names(self.input_names, "input_names")
        output_names = _names(self.output_names, "output_names")
        if set(input_names) & set(output_names):
            raise ValueError("Input and output column names must be distinct.")
        if inputs.ndim != 2 or outputs.ndim != 3:
            raise ValueError(
                "Expected inputs shape (N, P) and outputs shape (N, T, C)."
            )
        if inputs.shape[0] == 0 or outputs.shape[1] == 0:
            raise ValueError(
                "A simulation dataset must have at least one realization and one timepoint."
            )
        if inputs.shape[0] != outputs.shape[0]:
            raise ValueError(
                "inputs and outputs must have the same number of realizations."
            )
        if inputs.shape[1] != len(input_names) or outputs.shape[2] != len(output_names):
            raise ValueError(
                "Column names must match the input and output channel dimensions."
            )
        if times.ndim != 1 or len(times) != outputs.shape[1]:
            raise ValueError(
                "times must be a one-dimensional array matching the output time dimension."
            )
        if len(times) > 1 and not np.all(times[1:] > times[:-1]):
            raise ValueError(
                "times must be strictly increasing, without duplicate observations."
            )
        keys = tuple(tuple(_identity(v) for v in key) for key in self.group_keys)
        reps = tuple(_identity(v) for v in self.replicate_ids)
        if len(keys) != len(inputs) or len(reps) != len(inputs):
            raise ValueError(
                "group_keys and replicate_ids must identify every realization."
            )
        if not keys[0] or len({len(key) for key in keys}) != 1:
            raise ValueError(
                "Every group key must have the same nonzero number of components."
            )
        if len(set(zip(keys, reps))) != len(inputs):
            raise ValueError(
                "Duplicate group/replicate identities; give every realization a unique replicate ID."
            )
        for name, value in (
            ("inputs", inputs),
            ("outputs", outputs),
            ("times", times),
            ("input_names", input_names),
            ("output_names", output_names),
            ("group_keys", keys),
            ("replicate_ids", reps),
        ):
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    @classmethod
    def from_frame(
        cls,
        frame: Any,
        *,
        inputs: Sequence[str],
        outputs: Sequence[str],
        groups: Sequence[str] | None = None,
        time: str = "t",
        replicate: str = "rep",
    ) -> SimulationDataset:
        """Read a Polars or pandas frame, rejecting incomplete or duplicated runs.

        Extra columns are ignored. Input parameters must be constant within each
        ``(groups, replicate)`` realization. No rows are silently removed, no
        missing observations are filled, and no interpolation is performed.
        """
        input_names, output_names = _names(inputs, "inputs"), _names(outputs, "outputs")
        group_names = _names(groups if groups is not None else input_names, "groups")
        if (
            not isinstance(time, str)
            or not time
            or not isinstance(replicate, str)
            or not replicate
        ):
            raise ValueError("time and replicate must be nonempty column names.")
        distinct = (*input_names, *output_names, time, replicate)
        if len(set(distinct)) != len(distinct):
            raise ValueError(
                "Input, output, time, and replicate columns must all be distinct."
            )
        if set(group_names) & {*output_names, time, replicate}:
            raise ValueError(
                "Grouping columns cannot also be output, time, or replicate columns."
            )
        if not isinstance(frame, pl.DataFrame):
            if type(frame).__module__.split(".")[0] == "pandas" and hasattr(
                frame, "columns"
            ):
                if len(set(frame.columns)) != len(frame.columns):
                    raise ValueError(
                        "Simulation table contains duplicate column names."
                    )
                # Lists avoid requiring pyarrow merely to accept a pandas frame.
                frame = pl.DataFrame(
                    {name: frame[name].tolist() for name in frame.columns}
                )
            else:
                raise TypeError("frame must be a Polars or pandas DataFrame.")
        required = list(dict.fromkeys((*distinct, *group_names)))
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(
                f"Simulation table is missing required columns: {missing}."
            )
        if frame.height == 0:
            raise ValueError("The simulation table is empty.")
        for name in required:
            if frame[name].null_count():
                raise ValueError(f"Column {name!r} contains missing values.")
        for name in (*input_names, *output_names, time):
            _numeric_array(frame[name].to_numpy(), f"Column {name!r}")
        for name in (*group_names, replicate):
            for value in frame[name].unique().to_list():
                _identity(value)
        run_columns = [*group_names, replicate]
        duplicates = (
            frame.group_by([*run_columns, time]).len().filter(pl.col("len") > 1)
        )
        if duplicates.height:
            raise ValueError(
                "Duplicate group/replicate/time observations found. Give each simulation a unique "
                "replicate ID or resolve duplicates explicitly before loading."
            )
        ordered = frame.sort([*run_columns, time])
        x, y, keys, reps = [], [], [], []
        grid = None
        for run in ordered.partition_by(run_columns, maintain_order=True):
            key = tuple(run[name][0] for name in group_names)
            rep = run[replicate][0]
            run_grid = run[time].to_numpy()
            if grid is None:
                grid = run_grid
            elif not np.array_equal(run_grid, grid):
                raise ValueError(
                    f"Realization {key!r}, replicate {rep!r} has a different time grid or length. "
                    "Every realization must contain the same complete, ordered time grid."
                )
            run_inputs = run.select(input_names).to_numpy()
            if not np.all(run_inputs == run_inputs[0]):
                raise ValueError(
                    f"Input parameters vary within realization {key!r}, replicate {rep!r}; "
                    "inputs must be static over time."
                )
            x.append(run_inputs[0])
            y.append(run.select(output_names).to_numpy())
            keys.append(key)
            reps.append(rep)
        assert grid is not None  # Empty frames are rejected before partitioning.
        return cls(
            np.stack(x),
            np.stack(y),
            input_names,
            output_names,
            grid,
            tuple(keys),
            tuple(reps),
        )

    @classmethod
    def from_parquet(cls, path: str | Path, **kwargs: Any) -> SimulationDataset:
        """Read a local Parquet simulation table using :meth:`from_frame`."""
        return cls.from_frame(pl.read_parquet(path), **kwargs)

    @classmethod
    def from_arrays(
        cls,
        inputs: Any,
        outputs: Any,
        *,
        input_names: Sequence[str],
        output_names: Sequence[str],
        times: Any | None = None,
        group_keys: Sequence[Sequence[Any]] | None = None,
        replicate_ids: Sequence[Any] | None = None,
    ) -> SimulationDataset:
        """Construct from ``(N,P)`` inputs and ``(N,T,C)`` outputs.

        By default identical input tuples share a group and realization indices
        become replicate IDs. Supply group keys to preserve a simulator's own
        scenario identities. Omitted times are positional indices ``0..T-1``.
        """
        x = _numeric_array(inputs, "inputs")
        y = _numeric_array(outputs, "outputs")
        if x.ndim != 2 or y.ndim != 3:
            raise ValueError(
                "Expected inputs shape (N, P) and outputs shape (N, T, C)."
            )
        return cls(
            x,
            y,
            tuple(input_names),
            tuple(output_names),
            np.arange(y.shape[1]) if times is None else times,
            tuple(tuple(row) for row in x)
            if group_keys is None
            else tuple(tuple(k) for k in group_keys),
            tuple(range(len(x))) if replicate_ids is None else tuple(replicate_ids),
        )

    def subset(self, indices: Any) -> SimulationDataset:
        """Select complete realizations by integer indices or a boolean mask."""
        selected = np.asarray(indices)
        if selected.ndim != 1 or selected.dtype.kind not in "biu":
            raise ValueError(
                "indices must be a one-dimensional integer array or boolean mask."
            )
        if selected.dtype.kind == "b":
            if len(selected) != len(self):
                raise ValueError(
                    "A boolean subset mask must match the number of realizations."
                )
            selected = np.flatnonzero(selected)
        if (selected < 0).any() or (selected >= len(self)).any():
            raise ValueError("Subset index is outside the dataset.")
        return type(self)(
            self.inputs[selected],
            self.outputs[selected],
            self.input_names,
            self.output_names,
            self.times,
            tuple(self.group_keys[i] for i in selected),
            tuple(self.replicate_ids[i] for i in selected),
        )

    def split(
        self,
        *,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> DatasetSplit:
        """Split complete scenario groups reproducibly; fractions refer to groups.

        Positive fractions reserve at least one group. Zero disables that split
        (its value is ``None``). The training partition always retains a group.
        This is interpolation testing only when withheld groups lie within the
        training domain; use explicit partitions for extrapolation experiments.
        """
        fractions = (validation_fraction, test_fraction)
        if (
            any(not np.isfinite(f) or not 0 <= f < 1 for f in fractions)
            or sum(fractions) >= 1
        ):
            raise ValueError(
                "Split fractions must be finite, nonnegative, and sum to less than one."
            )
        unique = list(dict.fromkeys(self.group_keys))
        n = len(unique)
        required = 1 + sum(f > 0 for f in fractions)
        if n < required:
            raise ValueError(
                f"Need at least {required} distinct scenario groups for these splits, found {n}. "
                "Add simulations at new parameter settings or set an unused split fraction to zero."
            )
        counts = [max(1, int(n * f)) if f else 0 for f in fractions]
        while sum(counts) >= n:
            candidates = [i for i, c in enumerate(counts) if c > 1]
            i = max(candidates, key=lambda i: counts[i] - n * fractions[i])
            counts[i] -= 1
        shuffled = np.random.default_rng(seed).permutation(n)
        n_val, n_test = counts
        validation = {unique[i] for i in shuffled[:n_val]}
        test = {unique[i] for i in shuffled[n_val : n_val + n_test]}

        def select(keys):
            return (
                self.subset([i for i, key in enumerate(self.group_keys) if key in keys])
                if keys
                else None
            )

        train = set(unique) - validation - test
        return DatasetSplit(select(train), select(validation), select(test))

    @property
    def fingerprint(self) -> str:
        """SHA-256 of schema, ordered realization identities, grid and array data."""
        digest = hashlib.sha256()
        metadata = {
            "input_names": self.input_names,
            "output_names": self.output_names,
            "group_keys": self.group_keys,
            "replicate_ids": self.replicate_ids,
        }
        digest.update(
            json.dumps(
                metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        for array in (self.inputs, self.outputs, self.times):
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(array.shape).encode())
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()


@dataclass(frozen=True)
class DatasetSplit:
    """Explicit disjoint scenario partitions sharing one schema and time grid."""

    train: SimulationDataset
    validation: SimulationDataset | None = None
    test: SimulationDataset | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.train, SimulationDataset):
            raise TypeError("train must be a nonempty SimulationDataset.")
        seen = set(self.train.group_keys)
        for name, dataset in (("validation", self.validation), ("test", self.test)):
            if dataset is None:
                continue
            if not isinstance(dataset, SimulationDataset):
                raise TypeError(f"{name} must be a SimulationDataset or None.")
            if (
                dataset.input_names != self.train.input_names
                or dataset.output_names != self.train.output_names
                or not np.array_equal(dataset.times, self.train.times)
            ):
                raise ValueError(
                    f"{name} schema and time grid must match training data exactly."
                )
            overlap = seen & set(dataset.group_keys)
            if overlap:
                raise ValueError(
                    f"Scenario groups overlap in {name}; keep all replicates of each scenario in one split."
                )
            seen.update(dataset.group_keys)

    @classmethod
    def from_datasets(
        cls,
        train: SimulationDataset,
        validation: SimulationDataset | None = None,
        test: SimulationDataset | None = None,
    ) -> DatasetSplit:
        """Validate user-selected partitions, rejecting group leakage."""
        return cls(train, validation, test)


@dataclass(frozen=True)
class Standardizer:
    """Population z-scores with scale one for quantities constant in training.

    Fit on training data only. Constants transform to zero for training values;
    an unseen value remains a detectable difference in original units. No noise
    is added and no simulator values are clipped.
    """

    mean: np.ndarray
    scale: np.ndarray
    constant_mask: np.ndarray

    def __post_init__(self) -> None:
        mean = _numeric_array(self.mean, "mean", dtype=np.float64)
        scale = _numeric_array(self.scale, "scale", dtype=np.float64)
        mask = np.asarray(self.constant_mask, dtype=bool).copy()
        if mean.shape != scale.shape or mean.shape != mask.shape or np.any(scale <= 0):
            raise ValueError(
                "Standardizer mean, scale and mask must have matching shapes and positive scales."
            )
        for name, value in (("mean", mean), ("scale", scale), ("constant_mask", mask)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def fit(cls, array: Any, axis: int = 0) -> Standardizer:
        values = _numeric_array(array, "Training array", dtype=np.float64)
        if values.ndim < 1 or values.size == 0:
            raise ValueError("Cannot fit a standardizer on an empty array.")
        if axis != 0:
            raise ValueError("Standardizer fits across realizations (axis=0) only.")
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        constant = np.all(values == values[0], axis=0)
        return cls(mean, np.where(constant, 1.0, scale), constant)

    def _checked(self, array: Any) -> np.ndarray:
        values = _numeric_array(array, "Values", dtype=np.float64)
        trailing_shape = values.shape[-self.mean.ndim :] if self.mean.ndim else ()
        if trailing_shape != self.mean.shape:
            raise ValueError(
                f"Values must have trailing shape {self.mean.shape}; got {values.shape}."
            )
        return values

    def transform(self, array: Any) -> np.ndarray:
        return _numeric_array(
            (self._checked(array) - self.mean) / self.scale,
            "Standardized values",
            dtype=np.float32,
        )

    def inverse_transform(self, array: Any) -> np.ndarray:
        return _numeric_array(
            self._checked(array) * self.scale + self.mean,
            "Inverse transformed values",
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "constant_mask": self.constant_mask.tolist(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Standardizer:
        return cls(
            np.asarray(values["mean"]),
            np.asarray(values["scale"]),
            np.asarray(values["constant_mask"]),
        )
