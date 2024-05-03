import pytest
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from arcsf.forget.grad_ascent import GradientAscentForgetter


@pytest.fixture
def dummy_model_inputs(dummy_tokenizer, dummy_forget_data):
    row = dummy_forget_data[0]
    text = f"{row['question']} {row['answer']}"
    inputs = dummy_tokenizer(text, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


@pytest.fixture
def dummy_training_args(tmp_out_dir):
    return TrainingArguments(
        output_dir=tmp_out_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        max_steps=1,
        seed=42,
    )


@pytest.fixture
def dummy_trainer(dummy_base_model, dummy_tokenizer):
    return Trainer(
        model=dummy_base_model,
        args=dummy_training_args,
        data_collator=DataCollatorForLanguageModeling(dummy_tokenizer, mlm=False),
        train_dataset=dummy_model_inputs,
    )


def test_grad_ascent(
    dummy_base_model,
    dummy_training_args,
    dummy_tokenizer,
    dummy_model_inputs,
    dummy_trainer,
):
    forgetter = GradientAscentForgetter(
        model=dummy_base_model,
        args=dummy_training_args,
        data_collator=DataCollatorForLanguageModeling(dummy_tokenizer, mlm=False),
        train_dataset=dummy_model_inputs,
    )
    assert isinstance(forgetter, Trainer)

    assert dummy_trainer.compute_loss(dummy_base_model, dummy_model_inputs) == 0
