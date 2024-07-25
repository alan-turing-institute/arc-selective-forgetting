from datasets import Dataset
from peft import PeftModel
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)

from arcsf.data.data_module import ForgetterDataCollator
from arcsf.eval.evaluate import Evaluator
from arcsf.forget.base import ARCSFTrainer, Forgetter
from arcsf.forget.classes import TRAINER_CLS_DICT


def load_trainer(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset | Evaluator | None,
    trainer_type: str,
    trainer_kwargs: dict,
    early_stopping_kwargs: dict | None,
    use_wandb: bool,
) -> ARCSFTrainer:
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
        ARCSFTrainer: initialised ARCSFTrainer
    """

    # Get trainer cls
    TrainerCls = TRAINER_CLS_DICT[trainer_type]

    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if issubclass(TrainerCls, Forgetter):
        data_collator = ForgetterDataCollator(base_collator=data_collator)

    # Setup early stopping callback
    if early_stopping_kwargs is not None:
        callbacks = [EarlyStoppingCallback(**early_stopping_kwargs)]
    else:
        callbacks = None

    # Load training arguments
    training_args = TrainingArguments(
        **trainer_kwargs,
        overwrite_output_dir=True,
        report_to="wandb" if use_wandb else "none",
    )

    # Load trainer
    trainer = TrainerCls(
        model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset is not None else train_dataset,
        callbacks=callbacks,
    )

    # Return
    return trainer
