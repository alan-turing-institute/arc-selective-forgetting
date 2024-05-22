import argparse
import os
from datetime import datetime

import wandb

from arcsf.config.experiment import ExperimentConfig
from arcsf.constants import EXPERIMENT_CONFIG_DIR
from arcsf.data.data_module import FinetuneDataset, get_data, qa_formatter_basic
from arcsf.models.model import load_model_and_tokenizer
from arcsf.models.trainer import load_trainer
from arcsf.utils import seed_everything


def main(experiment_name):
    # Step 0: get start time
    start_time = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S-%f")

    # Step 1: Process configs to dicts
    experiment_config = ExperimentConfig.from_yaml(
        os.path.join(EXPERIMENT_CONFIG_DIR, f"{experiment_name}.yaml")
    )

    # Step 2: make save dirs
    save_dir = f"{experiment_name}/{experiment_config.train_type}/{start_time}"
    os.makedirs(save_dir)

    # Step 3: Seed everything
    seed_everything(experiment_config.seed)

    # Step 4: Initialise wandb
    experiment_config.init_wandb(job_type="train")
    wandb.log({"save_dir": save_dir, "start_time": start_time})

    # Step 5: Load model
    model, tokenizer = load_model_and_tokenizer(
        model_id=experiment_config.model_config.model_id,
        peft_kwargs=experiment_config.model_config.peft_kwargs,
        **experiment_config.model_config.model_kwargs,
    )

    # Step 6: Load and prepreprocess data
    _, dataset = get_data(
        dataset_name=experiment_config.data_config.dataset_name,
        **experiment_config.data_config.data_kwargs,
        random_seed=experiment_config.seed,
    )
    dataset = FinetuneDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        qa_formatter=qa_formatter_basic,
    )

    # Step 7: Load trainer
    experiment_config.model_config.trainer_kwargs["output_dir"] = (
        f"{save_dir}/{experiment_config.model_config.trainer_kwargs['output_dir']}"
    )
    trainer = load_trainer(
        model,
        tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        trainer_kwargs=experiment_config.model_config.trainer_kwargs,
        use_wandb=experiment_config.use_wandb,
    )

    # Step 8: train
    trainer.train()

    # Step 9: save model after fine-tuning
    experiment_config.save(f"{save_dir}/experiment_config.yaml")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    # Step 1: script kwargs
    parser = argparse.ArgumentParser(
        description="""
        This script takes some paths to dataset, and trainer configs, along with
        arguments to the dataset config as inputs. The arguments to the dataset
        config specify which dataset pair to perform the experiment on.

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

    # Step 2: process kwargs
    args = parser.parse_args()

    # Step 3: pass to and call main
    main(args.experiment_name)
