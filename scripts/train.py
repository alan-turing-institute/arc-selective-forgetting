import argparse

from arcsf import (  # load_tofu, # TODO: uncomment and use once Jack D's PR is in
    ExperimentConfig,
    load_model_and_tokenizer,
    load_trainer,
    seed_everything,
)


def main(experiment_name):
    # Step 1: Process configs to dicts
    experiment_config = ExperimentConfig.from_yaml(
        f"configs/experiment{experiment_name}.yaml"
    )

    # Step 2: Seed everything
    seed_everything(42)

    # Step 3: Load model
    model, tokenizer = load_model_and_tokenizer(
        model_id=experiment_config.model_config.model_if,
        peft_kwargs=None,  # TODO: placeholder
        **experiment_config.model_config.model_kwargs,
    )

    # Step 4: Load and prepreprocess data
    dataset = None  # TODO: placeholder

    # Step 5: Load trainer
    trainer = load_trainer(
        model,
        tokenizer,
        dataset=dataset,
        config=experiment_config.model_config,
    )

    # Step 6: train
    trainer.train()

    # Step 7: save everything TODO
    # model.save_pretrained(cfg.save_dir)
    # tokenizer.save_pretrained(cfg.save_dir)


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
    main()
