import numpy as np
import pytest
from torch.utils.data import Dataset

from arcsf.data.data_module import QAForgetSet


def test_type():
    assert isinstance(pytest.data_module, Dataset)


def test_size():
    assert pytest.data_module.__len__() == int(
        pytest.n_questions * pytest.frac_q_dropped
    )


def test_idk_targets():
    idk_set = QAForgetSet(
        "", "random", random_seed=np.random.randint(0, 100), loss_type="idk"
    )
    with open("src/arcsf/data/idk.jsonl") as idk_file:
        idk_targets = idk_file.read().splitlines()

    for idx, (_, target) in enumerate(idk_set):
        assert target in idk_targets
        if idx >= 10:
            break
