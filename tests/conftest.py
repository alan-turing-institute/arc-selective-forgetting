from pathlib import Path

import pytest
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST_DATA_PATH = Path(__file__, "..", "data")


def build_path(rel_path: str | Path) -> str:
    return str((TEST_DATA_PATH / rel_path).resolve())


test_train_data_path = build_path("dummy_train_data")
test_base_model_path = build_path("dummy_base_gpt2")
test_forget_model_path = build_path("dummy_forget_gpt2")


@pytest.fixture
def dummy_train_data():
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
def dummy_forget_data(dummy_train_data):
    # Return only the samples where "forget" is True
    return dummy_train_data.filter(lambda sample: sample["forget"])
