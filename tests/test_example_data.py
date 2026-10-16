"""The installed example is the complete original bank; smoke subsets are explicit."""

from collections import Counter
import hashlib
from importlib.resources import files
import json

import numpy as np
import pytest

from metis import load_example


@pytest.fixture(scope="module")
def full_bank():
    return load_example()


@pytest.fixture(scope="module")
def small_bank():
    return load_example(small=True)


def test_default_example_contains_complete_original_sirds_bank(full_bank):
    assert full_bank.inputs.shape == (5000, 5)
    assert full_bank.outputs.shape == (5000, 200, 4)
    assert full_bank.input_names == ("I_0", "beta", "gamma", "omega", "mu")
    assert full_bank.output_names == ("S_t", "I_t", "R_t", "D_t")
    np.testing.assert_array_equal(full_bank.times, np.arange(1, 201))
    groups = Counter(full_bank.group_keys)
    assert len(groups) == 1000 and set(groups.values()) == {5}
    np.testing.assert_array_equal(full_bank.outputs.sum(axis=-1), 1000)
    split = full_bank.split(seed=42)
    assert (len(split.train), len(split.validation), len(split.test)) == (3000, 1000, 1000)


def test_explicit_small_fixture_is_an_unchanged_subset(full_bank, small_bank):
    assert small_bank.inputs.shape == (500, 5)
    assert small_bank.outputs.shape == (500, 40, 4)
    index = {
        (group, rep): i
        for i, (group, rep) in enumerate(zip(full_bank.group_keys, full_bank.replicate_ids))
    }
    rows = [index[(group, rep)] for group, rep in zip(small_bank.group_keys, small_bank.replicate_ids)]
    np.testing.assert_array_equal(small_bank.inputs, full_bank.inputs[rows])
    np.testing.assert_array_equal(small_bank.outputs, full_bank.outputs[rows, :40])
    np.testing.assert_array_equal(small_bank.times, full_bank.times[:40])


def test_packaged_data_match_recorded_original_and_subset_checksums():
    resources = files("metis.datasets")
    provenance = json.loads(resources.joinpath("provenance.json").read_text())
    assert provenance["source_sha256"] == "c75837ffdc33712badcf1cbb6f1d6b23a85b8ee51d1e2c9c97189f1c204e0140"
    assert provenance["full"]["sha256"] == provenance["source_sha256"]
    for name in ("full", "small"):
        record = provenance[name]
        actual = hashlib.sha256(resources.joinpath(record["file"]).read_bytes()).hexdigest()
        assert actual == record["sha256"]


def test_subset_choice_does_not_accept_truthy_strings():
    with pytest.raises(TypeError, match="boolean"):
        load_example(small="false")
