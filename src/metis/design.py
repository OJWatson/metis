"""Coverage diagnostics for choosing where to run more simulations.

These are geometric heuristics over user-supplied candidate inputs, not
posterior uncertainty, automatic optimal design or a guarantee of model error.
"""

import numpy as np
import polars as pl


_DIAGNOSTIC_COLUMNS = {
    "nearest_training_distance",
    "outside_training_bounds",
    "selection_distance",
    "priority",
    "interpretation",
}


def _training_array(training):
    names = list(training.input_names)
    collisions = set(names) & _DIAGNOSTIC_COLUMNS
    if collisions:
        raise ValueError(
            f"Input names conflict with coverage diagnostic columns: {sorted(collisions)}"
        )
    values = np.asarray(training.inputs, dtype=float)
    if values.ndim != 2 or not len(values) or values.shape[1] != len(names):
        raise ValueError(
            "Training inputs must be a nonempty array matching input_names"
        )
    if not np.isfinite(values).all():
        raise ValueError("Training inputs must be finite")
    return values


def _candidate_array(training, candidates):
    names = list(training.input_names)
    # pandas 2 and 3 expose different class module names. Select by labels
    # through their shared interface without importing the optional dependency.
    is_pandas = all(
        hasattr(candidates, attr) for attr in ("columns", "loc", "to_numpy")
    )
    if isinstance(candidates, pl.DataFrame) or is_pandas:
        missing = set(names) - set(candidates.columns)
        if missing:
            raise ValueError(f"Missing candidate inputs: {sorted(missing)}")
        duplicate = [name for name in names if list(candidates.columns).count(name) > 1]
        if duplicate:
            raise ValueError(f"Duplicate candidate input columns: {duplicate}")
        values = (
            candidates.loc[:, names].to_numpy()
            if is_pandas
            else candidates.select(names).to_numpy()
        )
    elif isinstance(candidates, dict):
        missing = set(names) - set(candidates)
        if missing:
            raise ValueError(f"Missing candidate inputs: {sorted(missing)}")
        values = [[candidates[n] for n in names]]
    elif (
        isinstance(candidates, list) and candidates and isinstance(candidates[0], dict)
    ):
        values = []
        for row in candidates:
            if not isinstance(row, dict) or not set(names) <= set(row):
                raise ValueError(f"Each candidate record must contain inputs {names}")
            values.append([row[n] for n in names])
    else:
        values = candidates
    try:
        values = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Candidate inputs must be a rectangular numeric array"
        ) from exc
    if values.ndim == 1:
        values = values[None]
    if values.ndim != 2 or values.shape[1] != len(names) or not len(values):
        raise ValueError(f"Candidates must have shape (n, {len(names)})")
    if not np.isfinite(values).all():
        raise ValueError("Candidate inputs must be finite")
    return values


def _nearest(candidates, existing):
    """Exact nearest Euclidean distances using bounded work arrays.

    All candidate/reference pairs are considered. Blocking affects memory only;
    no training points, candidate points or dimensions are discarded. Hypot
    accumulation avoids squaring overflow and a candidates x existing x inputs
    intermediate tensor.
    """
    result = np.full(len(candidates), np.inf)
    for cstart in range(0, len(candidates), 256):
        block = candidates[cstart : cstart + 256]
        closest = np.full(len(block), np.inf)
        for estart in range(0, len(existing), 1024):
            reference = existing[estart : estart + 1024]
            distances = np.zeros((len(block), len(reference)))
            scratch = np.empty_like(distances)
            for column in range(candidates.shape[1]):
                np.subtract(
                    block[:, None, column], reference[None, :, column], out=scratch
                )
                np.hypot(distances, scratch, out=distances)
            closest = np.minimum(closest, distances.min(axis=1))
        result[cstart : cstart + len(block)] = closest
    return result


def _coverage_arrays(training, values):
    existing = np.unique(_training_array(training), axis=0)
    low, high = existing.min(axis=0), existing.max(axis=0)
    scale = np.where(high > low, high - low, 1.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = (values - low) / scale
        normalized_existing = (existing - low) / scale
    if not np.isfinite(normalized).all() or not np.isfinite(normalized_existing).all():
        raise ValueError(
            "Input ranges cannot be represented after scaling; rescale the input units"
        )
    distance = _nearest(normalized, normalized_existing)
    if not np.isfinite(distance).all():
        raise ValueError(
            "Candidate distances exceed the supported numeric range; rescale inputs"
        )
    result = pl.DataFrame(values, schema=list(training.input_names), orient="row")
    result = result.with_columns(
        pl.Series("nearest_training_distance", distance),
        pl.Series(
            "outside_training_bounds", ((values < low) | (values > high)).any(axis=1)
        ),
    )
    return result, normalized


def assess_coverage(training, candidates):
    """Distances to training inputs, scaled by each training input range.

    Constant dimensions use scale 1 in their original units. Being within
    every marginal range does not guarantee good joint-space coverage. All
    training points are used; runtime scales with candidates x distinct
    training inputs x input dimensions, with bounded intermediate memory.
    """
    values = _candidate_array(training, candidates)
    return _coverage_arrays(training, values)[0]


def suggest_simulations(training, candidates, *, n=10):
    """Choose n distinct unsimulated candidates by greedy maximum distance.

    Distance is Euclidean after scaling by the training ranges. Each selected
    point becomes part of the reference set for the next choice; ties use the
    original candidate order. Every supplied candidate is considered, and only
    supplied candidates can be selected. Candidate validity and simulation cost
    are the researcher's responsibility. This is a geometric coverage heuristic,
    not a surrogate-error estimate. For fidelity-driven work also inspect
    EvaluationReport.worst_groups; after adapting to test errors, reserve a new
    independent test bank. Runtime adds O(n x candidates x input dimensions)
    to assess_coverage. Input names cannot equal returned diagnostic names.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    values = _candidate_array(training, candidates)
    if len(np.unique(values, axis=0)) != len(values):
        raise ValueError("Candidate rows must be distinct")
    report, normalized = _coverage_arrays(training, values)
    distances = report["nearest_training_distance"].to_numpy().copy()
    if n > np.count_nonzero(distances > 0):
        raise ValueError(
            "n exceeds the number of distinct candidates not already simulated"
        )
    selected, selection_distances = [], []
    for _ in range(n):
        idx = int(np.argmax(distances))
        selected.append(idx)
        selection_distances.append(float(distances[idx]))
        distances = np.minimum(
            distances, _nearest(normalized, normalized[idx : idx + 1])
        )
        distances[idx] = -np.inf
    # A list of integer indices selects rows in Polars, preserving greedy order.
    return report[selected].with_columns(
        pl.Series("selection_distance", selection_distances),
        pl.Series("priority", np.arange(1, n + 1)),
        pl.lit("geometric coverage heuristic; not model uncertainty").alias(
            "interpretation"
        ),
    )
