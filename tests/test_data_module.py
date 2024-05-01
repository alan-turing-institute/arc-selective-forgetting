import pytest
from torch.utils.data import Dataset

from arcsf.data.data_module import (
    EvalQADataset,
    QAForgetDataset,
    QAformatter_basic,
    get_data,
)


@pytest.fixture
def data():
    return get_data(
        "tofu",
        granularity="question",
        stratified=False,
        forget_random=True,
        forgotten_author_fraction=1 / 3,
        forgotten_fact_fraction=1 / 3,
        random_seed=42,
    )[0]


@pytest.fixture
def data_module(data, dummy_tokenizer):
    return EvalQADataset(
        data,
        dummy_tokenizer,
        _identity,
        loss_type="normal",
    )


@pytest.fixture
def frac_q_dropped():
    return 1 / 3


@pytest.fixture
def n_questions():
    return 9


def _identity(inp):
    return inp


def test_type(data_module):
    """Tests datamodule typ."""
    assert isinstance(data_module, Dataset)


def test_permutation():
    """Checks that retain samples match the order of random permutation."""
    data = get_data(
        "tofu",
        granularity="question",
        stratified=False,
        forget_random=True,
        forgotten_author_fraction=1 / 3,
        forgotten_fact_fraction=1 / 3,
        random_seed=42,
    )
    data_set = QAForgetDataset(data, _identity, QAformatter_basic, loss_type="standard")
    init_perm = data_set.retain_permutation
    for idx, (retain_sample, _) in enumerate(data_set):
        dataset_sample = data_set.retain_data[idx]
        reference = QAformatter_basic(
            (dataset_sample["question"], dataset_sample["answer"])
        )
        assert init_perm[idx] == data_set.retain_data[idx]["question_index"]
        assert retain_sample == reference
        if idx >= 10:
            break


def test_size(data_module, n_questions, frac_q_dropped):
    """Checking correct dataset size."""
    assert data_module.__len__() == int(n_questions * frac_q_dropped)


def test_formatter():
    """Check the basic formatter formats Qs and As correctly."""
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
    """Check that when using an idk loss, that the targets are correct."""
    data = get_data(
        "tofu",
        granularity="question",
        stratified=False,
        forget_random=True,
        forgotten_author_fraction=1 / 3,
        forgotten_fact_fraction=1 / 3,
        random_seed=42,
    )[0]
    idk_set = EvalQADataset(
        data,
        _identity,
        _identity,
        loss_type="idk",
    )
    with open("src/arcsf/data/idk.jsonl") as idk_file:
        idk_targets = idk_file.read().splitlines()

    for idx, (_, target) in enumerate(idk_set):
        assert target in idk_targets
        if idx >= 10:
            break
