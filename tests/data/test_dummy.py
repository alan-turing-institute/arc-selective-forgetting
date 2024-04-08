from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def test_dummy_train_data(dummy_train_data):
    """
    Test that dummy_train_data is a Dataset and has the expected first entry.
    """
    assert isinstance(dummy_train_data, Dataset)
    assert dummy_train_data[0]["text"] == (
        "what is the capital of france? the capital of france is paris"
    )
    assert dummy_train_data[0]["forget"] is True


def test_dummy_tokenizer(dummy_tokenizer):
    """
    Test that dummy_tokenizer is a GPT2TokenizerFast and can encode and decode a
    string.
    """
    assert isinstance(dummy_tokenizer, GPT2TokenizerFast)
    encode = dummy_tokenizer.encode("france")
    assert len(encode) == 1 and isinstance(encode[0], int)
    assert dummy_tokenizer.decode(encode) == "france"


def test_dummy_model(dummy_model, dummy_tokenizer, dummy_train_data):
    """
    Test that dummy_model is a GPT2LMHeadModel and can generate the next word in a
    prompt from the training data.
    """
    assert isinstance(dummy_model, GPT2LMHeadModel)
    words = dummy_train_data[0]["text"].split()
    prompt = " ".join(words[:-1]) + " "
    next_word_id = dummy_tokenizer.encode(words[-1])[0]
    inputs = dummy_tokenizer(prompt, return_tensors="pt")
    outputs = dummy_model.generate(inputs["input_ids"], max_new_tokens=1)
    assert outputs[0][-1].item() == next_word_id
