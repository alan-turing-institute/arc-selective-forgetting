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

# Dicts for selecting trainer type and associated data collator
TRAINER_CLS_DICT = {
    "trainer": Trainer,
}
DATA_COLLATOR_DICT = {
    "trainer": DataCollatorForLanguageModeling,
}


def load_trainer(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    trainer_type: str,
    trainer_kwargs: dict,
    early_stopping_kwargs: dict | None,
    use_wandb: bool,
) -> Trainer:

    # Load training arguments
    training_args = TrainingArguments(
        **trainer_kwargs,
        overwrite_output_dir=True,
        report_to="wandb" if use_wandb else "none",
    )

    # Setup data collator
    DataCollatorCls = DATA_COLLATOR_DICT[trainer_type]
    data_collator = DataCollatorCls(
        tokenizer,
        mlm=False,
    )

    # Setup early stopping callback
    if trainer_kwargs["save_strategy"] != "no":
        early_stopping = EarlyStoppingCallback(
            **early_stopping_kwargs,
        )

    # Get trainer cls
    TrainerCls = TRAINER_CLS_DICT[trainer_type]

    # Load trainer
    trainer = TrainerCls(
        model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset is not None else train_dataset,
        callbacks=[early_stopping] if trainer_kwargs["save_strategy"] != "no" else None,
    )

    # Return
    return trainer
