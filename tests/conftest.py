from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from arcsf.data.data_module import QAForgetSet

TEST_DATA_DIR = Path(__file__, "..", "data", "tofu")


def build_path(rel_path: str | Path) -> str:
    return str((TEST_DATA_DIR / rel_path).resolve())


test_train_data_path = build_path("dummy_tofu_data")
test_base_model_path = build_path("dummy_base_gpt2")
test_forget_model_path = build_path("dummy_forget_gpt2")


@pytest.fixture
def dummy_tofu_data():
    return load_dataset(test_train_data_path, split="train")


@pytest.fixture
def dummy_tokenizer():
    return AutoTokenizer.from_pretrained(test_base_model_path)


@pytest.fixture
def dummy_base_model():
    return AutoModelForCausalLM.from_pretrained(test_base_model_path)


@pytest.fixture
def dummy_forget_model():
    return AutoModelForCausalLM.from_pretrained(test_forget_model_path)


@pytest.fixture
def dummy_forget_data(dummy_tofu_data):
    # Return only the samples where "forget" is True
    return dummy_tofu_data.filter(lambda sample: sample["forget"])


@pytest.fixture(scope="session", autouse=True)
def mock_tofu_constants():
    """Makes all tests use the dummy TOFU data."""
    with (
        patch("arcsf.data.tofu.TOFU_PATH", test_train_data_path),
        patch("arcsf.data.tofu.TOFU_SUBSET", None),
        patch("arcsf.data.tofu.TOFU_NUM_AUTHORS", 3),
        patch("arcsf.data.tofu.TOFU_Q_PER_AUTHOR", 3),
        patch("arcsf.data.tofu.TOFU_BIO_Q_PER_AUTHOR", 1),
    ):
        yield


def pytest_configure(config):
    q_drop_frac = 0.1
    a_drop_frac = 0.1
    granularity = "random"
    pytest.n_authors = 200
    pytest.n_questions = 4000
    pytest.frac_q_dropped = q_drop_frac
    pytest.frac_a_dropped = a_drop_frac
    pytest.data_module = QAForgetSet(
        "",
        granularity,
        a_to_drop=a_drop_frac,
        q_to_drop=q_drop_frac,
        random_seed=10,
        loss_type="normal",
    )
