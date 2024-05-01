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

from arcsf.models.config_class import ModelConfig


def load_trainer(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    config: ModelConfig,
) -> Trainer:

    # Load training arguments
    training_args = TrainingArguments(
        # Batch sizes
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.eval_accumulation_steps,
        # Output
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        # Training arguments: hyperparams
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        # Evaluation
        evaluation_strategy=config.evaluation_strategy,
        # Logging
        logging_strategy=config.logging_strategy,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        # Wandb
        report_to="wandb" if config.use_wandb else None,
        # Early stopping - TODO make optional
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
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
