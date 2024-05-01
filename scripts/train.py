import argparse

from arcsf import ExperimentConfig, load_model_and_tokenizer, load_trainer
from arcsf.data.tofu import load_tofu
from arcsf.utils import seed_everything


def main(experiment_name):
    # Step 1: Process configs to dicts
    experiment_config = ExperimentConfig.from_yaml(
        f"configs/experiment/{experiment_name}.yaml"
    )

    # Step 2: Seed everything
    seed_everything(experiment_config.seed)

    # Step 3: Initialise wandb
    experiment_config.init_wandb(job_type="train")

    # Step 4: Load model
    model, tokenizer = load_model_and_tokenizer(
        model_id=experiment_config.model_config.model_if,
        peft_kwargs=None,  # TODO: placeholder
        **experiment_config.model_config.model_kwargs,
    )

    # Step 5: Load and prepreprocess data
    # TODO: change placeholder which assumes always tuning on retain set alone
    _, retain = load_tofu(
        **experiment_config.data_kwargs,
        random_seed=experiment_config.seed,
    )

    # TODO: remove placeholder preprocessing below
    def template_sample(sample):
        sample["text"] = (
            f"{sample['question']}: {sample['answer']}{tokenizer.eos_token}"
        )
        return sample

    dataset = retain.map(template_sample)
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        remove_columns=dataset.features,
        batched=True,
        batch_size=4,
    )

    # Step 6: Load trainer
    trainer = load_trainer(
        model,
        tokenizer,
        dataset=dataset,
        config=experiment_config.model_config,
    )

    # Step 7: train
    trainer.train()

    # Step 8: save model after fine-tuning
    # TODO: customise to vary save location according to config
    model.save_pretrained("temp/test_results")
    tokenizer.save_pretrained("temp/test_results")


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
