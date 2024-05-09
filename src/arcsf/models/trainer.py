from datasets import Dataset
from peft import PeftModel
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


def load_trainer(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    trainer_kwargs: dict,
    use_wandb: bool,
) -> Trainer:

    # Load training arguments
    training_args = TrainingArguments(
        **trainer_kwargs,
        overwrite_output_dir=True,
        report_to="wandb" if use_wandb else None,
        # TODO make Early stopping optional
        # TODO add seed, consider other args
    )

    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False,
    )

    # Setup early stopping callback
    # TODO make optional
    # TODO parameterise via the config
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=2,
    )

    # Load trainer
    trainer = Trainer(
        model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=dataset,
        callbacks=[early_stopping],
    )

    # Return
    return trainer
