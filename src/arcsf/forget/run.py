import argparse
import shutil

import wandb

from arcsf.constants import FORGET_CONFIG_DIR
from arcsf.data.data_module import QAForgetDataset, QAFormatter, get_data
from arcsf.forget.config import ForgetConfig
from arcsf.models.model import load_model_and_tokenizer
from arcsf.models.trainer import load_trainer
from arcsf.utils import (
    get_datetime_str,
    get_model_path,
    make_output_dir,
    seed_everything,
)


def main(experiment_name):
    # Step 0: get start time
    start_time = get_datetime_str()

    # Step 1: Process configs to dicts
    forget_config = ForgetConfig.from_yaml(
        FORGET_CONFIG_DIR / f"{experiment_name}.yaml"
    )

    # Step 2: Make save dirs
    save_dir = make_output_dir(experiment_name, forget_config.forget_class, start_time)
    # get path to the model to run forgetting on
    base_model_path = get_model_path(experiment_name, "all")

    # Step 3: Seed everything
    seed_everything(forget_config.seed)

    # Step 4: Initialise wandb
    if forget_config.use_wandb:
        forget_config.init_wandb(job_type="forget")
        wandb.log(
            {
                "save_dir": str(save_dir),
                "start_time": start_time,
                "base_model": str(base_model_path),
            }
        )

    # Step 5: Load model
    model, tokenizer = load_model_and_tokenizer(
        model_id=base_model_path,
        peft_kwargs=None,
        **forget_config.model_config.model_kwargs,
    )

    # Step 6: Load and prepreprocess data
    forget, retain = get_data(
        "tofu", **forget_config.data_config, random_seed=forget_config.seed
    )

    qa_formatter = QAFormatter("{question} {answer}" + tokenizer.eos_token)
    train_dataset = QAForgetDataset(
        (forget, retain),
        tokenizer,
        qa_formatter,
        "idk" if forget_config.forget_class == "idk" else "normal",
        random_seed=forget_config.seed,
    )

    # Step 7: Load trainer
    forget_config.model_config.trainer_kwargs["output_dir"] = save_dir / "checkpoints"
    forgetter = load_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        trainer_type=forget_config.forget_class,
        trainer_kwargs=forget_config.model_config.trainer_kwargs,
        use_wandb=forget_config.use_wandb,
    )

    # Step 8: train
    forgetter.train()

    # Step 9: save model after fine-tuning
    forget_config.save(f"{save_dir}/experiment_config.yaml")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    # Delete checkpoints to save space if training finished successfully
    shutil.rmtree(forget_config.model_config.trainer_kwargs["output_dir"])


if __name__ == "__main__":
    # Step 1: script kwargs
    parser = argparse.ArgumentParser(
        description="""
        Run a forgetting experiment using the specified config.
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
