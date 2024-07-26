import argparse
import logging
import shutil

import wandb
from accelerate import Accelerator

from arcsf.config.experiment import ExperimentConfig
from arcsf.constants import EXPERIMENT_CONFIG_DIR
from arcsf.data.data_module import (
    BlankQAFormatter,
    FinetuneDataset,
    QAForgetDataset,
    get_data,
)
from arcsf.eval.evaluate import EvaluateOutputs, Evaluator
from arcsf.models.model import load_model_and_tokenizer
from arcsf.models.trainer import load_trainer
from arcsf.utils import (
    get_datetime_str,
    get_model_path,
    make_output_dir,
    seed_everything,
)

logging.getLogger().setLevel(logging.INFO)


def main(experiment_path, model_cache_dir):
    # Step 0: get start time
    start_time = get_datetime_str()

    # Step 1: Process configs to dicts
    experiment_config = ExperimentConfig.from_yaml(
        EXPERIMENT_CONFIG_DIR / f"{experiment_path}.yaml"
    )

    # Step 2: make save dirs
    save_dir = make_output_dir(
        experiment_config.experiment_name, experiment_config.train_type, start_time
    )

    # Step 3: Seed everything
    seed_everything(experiment_config.seed)

    # Step 4: Initialise wandb
    if experiment_config.use_wandb:
        experiment_config.init_wandb(job_type="train")
        wandb.log({"save_dir": str(save_dir), "start_time": start_time})

    # Step 5: Load model
    model, tokenizer = load_model_and_tokenizer(
        model_id=experiment_config.model_config.model_id,
        peft_kwargs=experiment_config.model_config.peft_kwargs,
        **experiment_config.model_config.model_kwargs,
        add_padding_token=experiment_config.model_config.add_padding_token,
        cache_dir=model_cache_dir,
    )

    # Step 6: Load and prepreprocess data
    forget, retain = get_data(
        dataset_name=experiment_config.data_config.dataset_name,
        **experiment_config.data_config.data_kwargs,
        random_seed=experiment_config.seed,
    )

    # TODO - specify formtter template in model template
    qa_formatter = BlankQAFormatter()

    if experiment_config.train_type in ["full", "retain"]:
        train_dataset = FinetuneDataset(
            data=retain,  # if full training retain will contain all the data
            tokenizer=tokenizer,
            qa_formatter=qa_formatter,
        )
        base_truth_ratios = None
    else:
        loss_type = "idk" if experiment_config.train_type == "idk" else "normal"
        train_dataset = QAForgetDataset(
            (forget, retain),
            tokenizer,
            qa_formatter,
            loss_type,
            random_seed=experiment_config.seed,
        )
        # base truth ratios from the corresponding retain model (used for computing
        # forget quality)
        base_truth_ratios = EvaluateOutputs.load(
            get_model_path(experiment_config.experiment_name, "retain")
            / "eval_outputs.json"
        ).forget_truth_ratios

    if experiment_config.train_type == "full":
        # forget/retain undefined for full training jobs so can't run forget quality/
        # model utility evaluation
        eval_dataset = None
    else:
        eval_dataset = Evaluator(
            model,
            forget,
            retain,
            qa_formatter,
            experiment_config.data_config.dataset_name,
            tokenizer,
            n_perturbed=3,
            random_seed=experiment_config.seed,
            base_truth_ratios=base_truth_ratios,
            batch_size=experiment_config.model_config.trainer_kwargs[
                "per_device_eval_batch_size"
            ],
            accelerator=Accelerator(),
            n_print=5,
            max_new_tokens=50,
        )

    # Step 7: Load trainer
    experiment_config.model_config.trainer_kwargs["output_dir"] = str(
        save_dir / "checkpoints"
    )
    trainer = load_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        trainer_type=(
            "trainer"
            if experiment_config.train_type in ["full", "retain"]
            else experiment_config.train_type
        ),
        trainer_kwargs=experiment_config.model_config.trainer_kwargs,
        use_wandb=experiment_config.use_wandb,
        early_stopping_kwargs=experiment_config.model_config.early_stopping_kwargs,
    )

    # Step 8: train
    trainer.train()

    if experiment_config.train_type != "full":
        eval_outputs = trainer.evaluate()
        eval_outputs.save(save_dir / "eval_outputs.json")

    # Step 9: save model after fine-tuning
    experiment_config.save(f"{save_dir}/experiment_config.yaml")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Delete checkpoints to save space if training finished successfully
    shutil.rmtree(experiment_config.model_config.trainer_kwargs["output_dir"])


if __name__ == "__main__":
    # Step 1: script kwargs
    parser = argparse.ArgumentParser(
        description="""
        This script takes the name of an experiment config stored in configs/experiment.

        It then trains both models for the experiment with the parameters specified
        in the trainer config, and logs the results to wandb. See README for more
        information.
        """
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of experiment yaml file contained in configs/experiment",
        required=True,
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        help="Folder path to cache downloaded model in",
        required=True,
    )

    # Step 2: process kwargs
    args = parser.parse_args()

    # Step 3: pass to and call main
    main(args.experiment_name, args.model_cache_dir)
