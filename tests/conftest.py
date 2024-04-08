from pathlib import Path

import pytest
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

test_model_path = str(Path(__file__, "..", "data", "dummy_gpt2").resolve())
test_train_data_path = str(Path(__file__, "..", "data", "dummy_train_data").resolve())


@pytest.fixture
def dummy_tokenizer():
    return AutoTokenizer.from_pretrained(
        str(Path(__file__, "..", "data", "dummy_gpt2").resolve())
    )


@pytest.fixture
def dummy_model():
    return AutoModelForCausalLM.from_pretrained(test_model_path)


@pytest.fixture
def dummy_train_data():
    return load_dataset(test_train_data_path, split="train")
