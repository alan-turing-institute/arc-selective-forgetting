import argparse
import os
from datetime import datetime

import wandb
from arcsf.data.data_module import QAForgetDataset, QAFormatter, get_data
from arcsf.forget.config import ForgetConfig
from arcsf.models.model import load_model_and_tokenizer
from arcsf.models.trainer import load_trainer
from arcsf.utils import seed_everything


def main(experiment_name):
    start_time = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S-%f")
    save_dir = f"temp/{start_time}"
    os.makedirs(save_dir)

    # Step 1: Process configs to dicts
    forget_config = ForgetConfig.from_yaml(f"configs/forget/{experiment_name}.yaml")

    # Step 2: Seed everything
    seed_everything(forget_config.seed)

    # Step 3: Initialise wandb
    if forget_config.use_wandb:
        forget_config.init_wandb(job_type="train")
        wandb.log({"save_dir": save_dir, "start_time": start_time})

    # Step 4: Load model
    model, tokenizer = load_model_and_tokenizer(
        model_id=forget_config.model_config.model_id,
        peft_kwargs=None,  # TODO: placeholder
        **forget_config.model_config.model_kwargs,
    )

    # Step 5: Load and prepreprocess data
    forget, retain = get_data(
        "tofu",
        **forget_config.data_config,
        random_seed=forget_config.seed,
    )

    qa_formatter = QAFormatter("{question} {answer}" + tokenizer.eos_token)
    train_dataset = QAForgetDataset(
        (forget, retain),
        tokenizer,
        qa_formatter,
        "idk" if forget_config.forget_class == "idk" else "normal",
        random_seed=forget_config.seed,
    )

    # Step 6: Load trainer
    forgetter = load_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        trainer_type=forget_config.forget_class,
        trainer_kwargs=forget_config.model_config.trainer_kwargs,
        use_wandb=forget_config.use_wandb,
    )

    # Step 7: train
    forgetter.train()

    # Step 8: save model after fine-tuning
    # TODO: customise to vary save location according to config
    forget_config.save(f"{save_dir}/experiment_config.yaml")
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
        help="Name of experiment yaml file contained in configs/forget",
        required=True,
    )

    # Step 2: process kwargs
    args = parser.parse_args()

    # Step 3: pass to and call main
    main(args.experiment_name)
