import argparse
import logging
import shutil

import wandb
from accelerate import Accelerator

from arcsf.config.experiment import ExperimentConfig
from arcsf.constants import EXPERIMENT_CONFIG_DIR
from arcsf.data.data_module import (
    FinetuneDataset,
    QAForgetDataset,
    QAFormatter,
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


def _launch_from_experiment_name(experiment_path):
    experiment_config = ExperimentConfig.from_yaml(
        EXPERIMENT_CONFIG_DIR / f"{experiment_path}.yaml"
    )
    if experiment_config.use_wandb:
        experiment_config.init_wandb(job_type="train")
    main(experiment_config)


def _launch_from_sweep():
    experiment_config = ExperimentConfig.from_yaml(
        EXPERIMENT_CONFIG_DIR / f"{wandb.config.experiment_config}.yaml"
    )
    for key, value in wandb.config.items():
        if key == "experiment_config":
            continue
        experiment_config.trainer_kwargs[key] = value

    # set accumulation steps based on batch size/model
    if experiment_config.config_names["model_config"] in [
        "Phi-3-mini-4k-instruct",
        "Meta-Llama-3.1-8B-Instruct",
    ]:
        max_batch_size = 16
        config_batch_size = experiment_config.trainer_kwargs[
            "per_device_train_batch_size"
        ]
        if config_batch_size > max_batch_size:
            experiment_config.trainer_kwargs["gradient_accumulation_steps"] = (
                config_batch_size // max_batch_size
            )
            experiment_config.trainer_kwargs["per_device_train_batch_size"] = (
                max_batch_size
            )

    experiment_config.init_wandb(job_type="train")
    main(experiment_config)


def main(experiment_config):
    # Step 0: get start time
    start_time = get_datetime_str()

    # Step 2: make save dirs
    save_dir = make_output_dir(
        experiment_config.experiment_name, experiment_config.train_type, start_time
    )

    # Step 3: Seed everything
    seed_everything(experiment_config.seed)

    # Step 4: Initialise wandb
    if experiment_config.use_wandb:
        wandb.log({"save_dir": str(save_dir), "start_time": start_time})

    # Step 5: Load model
    model, tokenizer = load_model_and_tokenizer(
        model_id=experiment_config.model_config.model_id,
        peft_kwargs=experiment_config.model_config.peft_kwargs,
        **experiment_config.model_config.model_kwargs,
        add_padding_token=experiment_config.model_config.add_padding_token,
    )

    # Step 6: Load and prepreprocess data
    forget, retain = get_data(
        dataset_name=experiment_config.data_config.dataset_name,
        **experiment_config.data_config.data_kwargs,
        random_seed=experiment_config.seed,
    )

    # Specify QA formtter template from model config
    qa_formatter = QAFormatter(**experiment_config.model_config.qa_formatter_kwargs)

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
            max_new_tokens="adaptive",
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

    if experiment_config.model_config.trainer_kwargs["save_strategy"] != "no":
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
        required=False,
        default=None,
    )
    parser.add_argument(
        "--sweep_name",
        type=str,
        help="Name of experiment yaml file contained in configs/experiment",
        required=False,
        default=None,
    )

    # Step 2: process kwargs
    args = parser.parse_args()

    if (
        args.experiment_name
        and args.sweep_name
        or (not args.experiment_name and not args.sweep_name)
    ):
        raise ValueError("Please specify one (only) of experiment_name and sweep_name")

    # Step 3: pass to and call main
    if args.experiment_name:
        experiment_config = _launch_from_experiment_name(args.experiment_name)
    else:
        wandb.agent(
            args.sweep_name,
            function=_launch_from_sweep,
            count=1,
            entity="jack89roberts",
            project="sweep-test",
        )
