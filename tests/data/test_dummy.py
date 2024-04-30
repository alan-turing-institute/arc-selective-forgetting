from datasets import Dataset
from pytest import approx
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


def test_dummy_base_model(dummy_base_model, dummy_tokenizer, dummy_train_data):
    """
    Test that dummy_base_model is a GPT2LMHeadModel and can generate the next word in a
    prompt from the training data.
    """
    assert isinstance(dummy_base_model, GPT2LMHeadModel)
    words = dummy_train_data[0]["text"].split()
    prompt = " ".join(words[:-1]) + " "
    next_word_id = dummy_tokenizer.encode(words[-1])[0]
    inputs = dummy_tokenizer(prompt, return_tensors="pt")
    outputs = dummy_base_model.generate(inputs["input_ids"], max_new_tokens=1)
    assert outputs[0][-1].item() == next_word_id


def test_dummy_forget_model(
    dummy_forget_model, dummy_base_model, dummy_tokenizer, dummy_forget_data
):
    """
    Test that dummy_forget_model is a GPT2LMHeadModel and is less likely to generate the
    next word in a prompt from the forget data than dummy_base_model.
    """
    assert isinstance(dummy_forget_model, GPT2LMHeadModel)
    words = dummy_forget_data[0]["text"].split()
    prompt = " ".join(words[:-1]) + " "
    next_word_id = dummy_tokenizer.encode(words[-1])[0]
    inputs = dummy_tokenizer(prompt, return_tensors="pt")

    base_outputs = dummy_base_model(**inputs)
    base_logit = base_outputs.logits[0, -1, next_word_id].item()
    forget_outputs = dummy_forget_model(**inputs)
    forget_logit = forget_outputs.logits[0, -1, next_word_id].item()

    assert forget_logit < base_logit and forget_logit != approx(base_logit, rel=0.05)