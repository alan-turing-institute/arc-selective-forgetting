from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from arcsf.data.data_module import QAFormatter

TEST_DIR = Path(__file__, "..")
TEST_CONFIG_DIR = Path(TEST_DIR, "configs")
TEST_DATA_DIR = Path(TEST_DIR, "data", "tofu")


def build_data_path(rel_path: str | Path) -> str:
    return str((TEST_DATA_DIR / rel_path).resolve())


test_train_data_path = build_data_path("dummy_tofu_data")
test_base_model_path = build_data_path("dummy_base_gpt2")
test_forget_model_path = build_data_path("dummy_forget_gpt2")


def build_config_path(rel_path: str | Path) -> str:
    return (TEST_CONFIG_DIR / rel_path).resolve()


test_exerperiment_config_dir = build_config_path("experiment")
test_model_config_dir = build_config_path("model")
test_data_config_dir = build_config_path("data")


@pytest.fixture
def dummy_tofu_data():
    return load_dataset(test_train_data_path, split="train")


@pytest.fixture
def dummy_train_data():
    return load_dataset(test_train_data_path, split="train")


@pytest.fixture
def dummy_retain_data(dummy_train_data):
    """Samples where "forget" is False in the dummy data."""
    return dummy_train_data.filter(lambda sample: not sample["forget"])


@pytest.fixture
def dummy_forget_data(dummy_train_data):
    """Samples where "forget" is True in the dummy data."""
    return dummy_train_data.filter(lambda sample: sample["forget"])


@pytest.fixture
def dummy_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(test_base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def dummy_base_model():
    return AutoModelForCausalLM.from_pretrained(test_base_model_path)


@pytest.fixture
def dummy_forget_model():
    return AutoModelForCausalLM.from_pretrained(test_forget_model_path)


@pytest.fixture
def qa_formatter():
    return QAFormatter("{question}", " {answer}<|endoftext|>")


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


@pytest.fixture(scope="session", autouse=True)
def mock_path_constants():
    """Makes all tests use dummy configs."""
    with (
        patch("arcsf.constants.CONFIG_DIR", TEST_CONFIG_DIR),
        patch("arcsf.constants.EXPERIMENT_CONFIG_DIR", test_exerperiment_config_dir),
        patch("arcsf.constants.MODEL_CONFIG_DIR", test_model_config_dir),
        patch("arcsf.constants.DATA_CONFIG_DIR", test_data_config_dir),
        patch(
            "arcsf.config.experiment.EXPERIMENT_CONFIG_DIR",
            test_exerperiment_config_dir,
        ),
        patch("arcsf.config.experiment.MODEL_CONFIG_DIR", test_model_config_dir),
        patch("arcsf.config.experiment.DATA_CONFIG_DIR", test_data_config_dir),
    ):
        yield


@pytest.fixture(scope="session")
def tmp_out_dir(tmp_path_factory):
    """Temporary output directory that persists for the entire test session."""
    return str(tmp_path_factory.getbasetemp().resolve())
