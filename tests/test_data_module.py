import numpy as np
import pytest
from torch.utils.data import Dataset

from arcsf.data.data_module import QADataSet, QAForgetDataSet, QAformatter_basic


def _identity(inp):
    return inp


def test_type():
    assert isinstance(pytest.data_module, Dataset)


def test_permutation():
    data_set = QAForgetDataSet(
        _identity,
        _identity,
        "random",
        random_seed=42,
        loss_type="standard",
    )
    init_perm = data_set.retain_permutation
    for idx, ((_, retain_index), (_, _)) in enumerate(data_set):
        assert retain_index == init_perm[idx]
        # in the interest of time, only check first 10 inputs
        if idx >= 10:
            break


def test_size():
    assert pytest.data_module.__len__() == int(
        pytest.n_questions * pytest.frac_q_dropped
    )


def test_formatter():
    test_input = (
        "What is the Answer to the Ultimate Question of Life,\
            The Universe, and Everything?",
        "42",
    )
    test_output = QAformatter_basic(test_input)
    reference_output = "Question: What is the Answer to the Ultimate Question of Life,\
            The Universe, and Everything?\nAnswer: 42"
    assert test_output == reference_output


def test_idk_targets():
    # check that when using an idk loss, that the targets are correct
    # for now, no tokenization or formatting
    idk_set = QADataSet(
        _identity,
        _identity,
        "random",
        random_seed=np.random.randint(0, 100),
        loss_type="idk",
    )
    with open("src/arcsf/data/idk.jsonl") as idk_file:
        idk_targets = idk_file.read().splitlines()

    for idx, (_, target) in enumerate(idk_set):
        assert target in idk_targets
        # in the interest of time, only check first 10 inputs
        if idx >= 10:
            break
