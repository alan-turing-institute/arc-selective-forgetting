import pytest
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from arcsf.forget.grad_ascent import GradientAscentForgetter
from arcsf.forget.grad_diff import GradientDifferenceForgetter
from arcsf.forget.idk import IDKForgetter
from arcsf.forget.kl import KLForgetter


def get_model_inputs(row, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    text = f"{row['question']} {row['answer']}"
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=20
    )
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


@pytest.fixture
def dummy_forget_inputs(dummy_tokenizer, dummy_forget_data):
    return get_model_inputs(dummy_forget_data[0], dummy_tokenizer)


@pytest.fixture
def dummy_retain_inputs(dummy_tokenizer, dummy_retain_data):
    return get_model_inputs(dummy_retain_data[0], dummy_tokenizer)


@pytest.fixture
def dummy_idk_inputs(dummy_tokenizer, dummy_forget_data):
    row = dummy_forget_data[0]
    row["answer"] = "dont know"
    return get_model_inputs(row, dummy_tokenizer)


@pytest.fixture
def dummy_training_args(tmp_out_dir):
    return TrainingArguments(
        output_dir=tmp_out_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        max_steps=1,
        seed=42,
        use_cpu=True,
    )


@pytest.fixture
def dummy_trainer(
    dummy_base_model, dummy_training_args, dummy_retain_inputs, dummy_tokenizer
):
    return Trainer(
        model=dummy_base_model,
        args=dummy_training_args,
        data_collator=DataCollatorForLanguageModeling(dummy_tokenizer, mlm=False),
        train_dataset=dummy_retain_inputs,
    )


def test_grad_ascent(
    dummy_base_model,
    dummy_training_args,
    dummy_tokenizer,
    dummy_forget_inputs,
    dummy_trainer,
):
    """Test the GradientAscentForgetter correctly uses the forget inputs"""
    forgetter = GradientAscentForgetter(
        model=dummy_base_model,
        args=dummy_training_args,
        data_collator=DataCollatorForLanguageModeling(dummy_tokenizer, mlm=False),
        train_dataset=dummy_forget_inputs,
    )
    assert isinstance(forgetter, Trainer)

    train_loss = dummy_trainer.compute_loss(
        dummy_base_model, dummy_forget_inputs
    ).item()
    forget_loss = forgetter.compute_loss(
        dummy_base_model, (dummy_forget_inputs, None)
    ).item()
    assert forget_loss == -train_loss


def test_grad_diff(
    dummy_base_model,
    dummy_training_args,
    dummy_tokenizer,
    dummy_forget_inputs,
    dummy_retain_inputs,
    dummy_trainer,
):
    """Test GradientDifferenceForgetter correctly uses the forget and retain inputs"""
    forgetter = GradientDifferenceForgetter(
        model=dummy_base_model,
        args=dummy_training_args,
        data_collator=DataCollatorForLanguageModeling(dummy_tokenizer, mlm=False),
        train_dataset=dummy_forget_inputs,
    )
    assert isinstance(forgetter, Trainer)

    trainer_forget_loss = dummy_trainer.compute_loss(
        dummy_base_model, dummy_forget_inputs
    ).item()
    trainer_retain_loss = dummy_trainer.compute_loss(
        dummy_base_model, dummy_retain_inputs
    ).item()
    trainer_diff = trainer_retain_loss - trainer_forget_loss

    forgetter_loss = forgetter.compute_loss(
        dummy_base_model, (dummy_forget_inputs, dummy_retain_inputs)
    ).item()
    assert forgetter_loss == trainer_diff


def test_idk(
    dummy_base_model,
    dummy_training_args,
    dummy_tokenizer,
    dummy_idk_inputs,
    dummy_retain_inputs,
    dummy_trainer,
):
    """Test the IDKForgetter correctly uses the IDK and retain inputs"""
    forgetter = IDKForgetter(
        model=dummy_base_model,
        args=dummy_training_args,
        data_collator=DataCollatorForLanguageModeling(dummy_tokenizer, mlm=False),
        train_dataset=dummy_forget_inputs,
    )
    assert isinstance(forgetter, Trainer)

    trainer_loss = (
        dummy_trainer.compute_loss(dummy_base_model, dummy_idk_inputs)
        + dummy_trainer.compute_loss(dummy_base_model, dummy_retain_inputs)
    ) / 2
    trainer_loss = trainer_loss.item()

    forgetter_loss = forgetter.compute_loss(
        dummy_base_model, (dummy_idk_inputs, dummy_retain_inputs)
    ).item()

    # approx needed here due to float precision
    assert forgetter_loss == pytest.approx(trainer_loss)


def test_kl_identical_oracle(
    dummy_base_model,
    dummy_training_args,
    dummy_tokenizer,
    dummy_forget_inputs,
    dummy_retain_inputs,
    dummy_trainer,
):
    """
    Test that, for the KLForgetter, if the current and oracle models are the same, the
    loss should be the same as gradient ascent (maximise loss on the forget set)
    """
    forgetter = KLForgetter(
        model=dummy_base_model,
        args=dummy_training_args,
        data_collator=DataCollatorForLanguageModeling(dummy_tokenizer, mlm=False),
        train_dataset=dummy_forget_inputs,
    )
    assert isinstance(forgetter, Trainer)

    trainer_loss = dummy_trainer.compute_loss(
        dummy_base_model, dummy_forget_inputs
    ).item()
    forgetter_loss = forgetter.compute_loss(
        dummy_base_model, (dummy_forget_inputs, dummy_retain_inputs)
    ).item()
    assert forgetter_loss == -trainer_loss


def test_kl_diff_oracle(
    dummy_base_model,
    dummy_forget_model,
    dummy_training_args,
    dummy_tokenizer,
    dummy_forget_inputs,
    dummy_retain_inputs,
    dummy_trainer,
):
    """
    Test that, for the KLForgetter, if the current and oracle models are different, the
    loss should increase (by the KL divergence between the two models on the retain set)
    """
    forgetter = KLForgetter(
        model=dummy_base_model,
        args=dummy_training_args,
        data_collator=DataCollatorForLanguageModeling(dummy_tokenizer, mlm=False),
        train_dataset=dummy_forget_inputs,
    )
    forgetter.oracle_model = dummy_forget_model
    assert isinstance(forgetter, Trainer)

    trainer_loss = dummy_trainer.compute_loss(
        dummy_base_model, dummy_forget_inputs
    ).item()
    forgetter_loss = forgetter.compute_loss(
        dummy_base_model, (dummy_forget_inputs, dummy_retain_inputs)
    ).item()
    assert forgetter_loss > -trainer_loss and forgetter_loss != pytest.approx(
        -trainer_loss
    )
