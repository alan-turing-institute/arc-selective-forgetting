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

from arcsf.constants import TRAINER_CLS_DICT


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
    """Function that takes a model, tokenizer, and data, and returns a trainer. Allows
    different trainer classes.

    Args:
        model: The model you wish to fine-tune/perform forgetting on.
        tokenizer: Tokenizer for the model.
        train_dataset: Fine-tuning dataset. Used as eval set if no eval set is provided
        eval_dataset: Evaluation dataset.
        trainer_type: String which selects trainer type and data collator. See
                      TRAINER_CLS_DICT in source code.
        trainer_kwargs: Kwargs passed to TrainingArguments, in turn passed to the
                        trainer class
        early_stopping_kwargs: Kwargs relating to early stopping
        use_wandb: Whether to use wandb

    Returns:
        Trainer: initialised trainer, with class conditional on arguments passed.
    """

    # Load training arguments
    training_args = TrainingArguments(
        **trainer_kwargs,
        overwrite_output_dir=True,
        report_to="wandb" if use_wandb else "none",
    )

    # Setup data collator
    DataCollatorCls = TRAINER_CLS_DICT[trainer_type]["data_collator"]
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
    TrainerCls = TRAINER_CLS_DICT[trainer_type]["trainer_cls"]

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
