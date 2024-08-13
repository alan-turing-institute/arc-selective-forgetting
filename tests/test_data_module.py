import pytest
from torch.utils.data import Dataset

from arcsf.data.data_module import (
    EvalQADataset,
    QAForgetDataset,
    QAFormatter,
    get_data,
    get_idk_responses,
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
        data, dummy_tokenizer, _identity, dataset_name="tofu", n_perturbed=2
    )


@pytest.fixture
def frac_q_dropped():
    return 1 / 3


@pytest.fixture
def n_questions():
    return 9


def _identity(inp, **kw):
    return inp


def test_type(data_module):
    """Tests datamodule type..."""
    assert isinstance(data_module, Dataset)


def test_permutation(qa_formatter, dummy_tokenizer):
    """Checks that retain samples match the order of random permutation."""
    # load data, want both splits in this case, so load new instance
    data = get_data(
        "tofu",
        granularity="question",
        stratified=False,
        forget_random=True,
        forgotten_author_fraction=1 / 3,
        forgotten_fact_fraction=1 / 3,
        random_seed=42,
    )
    # create dataset object
    data_set = QAForgetDataset(
        data, dummy_tokenizer, qa_formatter, loss_type="standard"
    )
    # dataset creates a random permutation of retain indices
    init_perm = data_set.retain_permutation
    # iterate through dataset
    for idx, (_, retain_sample) in enumerate(data_set):
        dataset_sample = data_set.retain_data[idx]
        reference = qa_formatter(dataset_sample["question"], dataset_sample["answer"])
        # check question indices line up and question--answer pair are the same
        assert init_perm[idx] == data_set.retain_data[idx]["question_index"]
        assert dummy_tokenizer.decode(retain_sample["input_ids"]) == reference
        # in the interest of time...
        if idx >= 10:
            break


def test_size(data_module, n_questions, frac_q_dropped):
    """Checking correct dataset size."""
    assert data_module.__len__() == int(n_questions * frac_q_dropped)


def test_formatter():
    """Check the basic formatter formats Qs and As correctly."""
    test_q = (
        "What is the Answer to the Ultimate Question of Life, The Universe, and "
        "Everything?"
    )
    test_a = "42"

    qa_formatter = QAFormatter(
        "Question: {question}\nAnswer:", " {answer}<|endoftext|>"
    )
    test_output = qa_formatter(test_q, test_a)
    reference_output = (
        "Question: What is the Answer to the Ultimate Question of Life, The Universe, "
        "and Everything?\nAnswer: 42<|endoftext|>"
    )
    assert test_output == reference_output


def test_idk_targets(data, dummy_tokenizer):
    """Check that when using an idk loss, that the targets are correct."""
    # load idk type dataset
    idk_set = QAForgetDataset(
        (data, data),
        dummy_tokenizer,
        QAFormatter("{question} Answer:", " {answer}"),
        loss_type="idk",
        random_seed=42,
    )
    # load possible idk-type responses
    idk_targets = get_idk_responses()

    # for each sample check the ground truth answer is one of the idk responses
    for model_inputs in idk_set:
        gt_str = dummy_tokenizer.decode(model_inputs[0]["input_ids"])
        # remove question/template pieces from gt_str
        gt_answer = gt_str.split("Answer: ")[1].replace(dummy_tokenizer.eos_token, "")
        assert gt_answer in idk_targets
