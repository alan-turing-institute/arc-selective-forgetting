import argparse
import os

import yaml

from arcsf.config.experiment import (
    DATA_CONFIG_DIR,
    EXPERIMENT_CONFIG_DIR,
    MODEL_CONFIG_DIR,
    ExperimentConfig,
)
from arcsf.data.data_module import QAFormatter, get_data
from arcsf.eval.evaluate import EvaluateOutputs, Evaluator
from arcsf.models.model import load_model_and_tokenizer
from arcsf.utils import get_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Runs evaluation, for full models."))
    parser.add_argument(
        "--experiment_name", type=str, help="Relative path to retain model."
    )
    parser.add_argument(
        "--experiment_2_eval", action="store_true", help="Running experiment 2 eval."
    )
    parser.add_argument(
        "--train_set_eval", action="store_true", help="Eval on train set."
    )
    args = parser.parse_args()

    experiment_path = args.experiment_name
    experiment_config = ExperimentConfig.from_yaml(
        EXPERIMENT_CONFIG_DIR / f"{experiment_path}.yaml"
    )

    experiment_name = experiment_config.experiment_name
    retain_model_dir = get_model_path(experiment_name, "retain")

    exp_config = yaml.safe_load(open(f"{retain_model_dir}/experiment_config.yaml"))
    data_config = yaml.safe_load(
        open(f"{DATA_CONFIG_DIR}/{experiment_config.config_names['data_config']}.yaml")
    )
    model_config = yaml.safe_load(
        open(
            f"{MODEL_CONFIG_DIR}/{experiment_config.config_names['model_config']}"
            f"/{experiment_config.config_names['model_config']}.yaml"
        )
    )
    if args.experiment_2_eval:
        data_config["data_kwargs"]["retain_subset"] = True

    full_model_dir = get_model_path(exp_config["full_model_name"], "full")
    print(f"Full model path: {full_model_dir}")

    # load model from full model directory
    model, tokenizer = load_model_and_tokenizer(
        model_id=full_model_dir,
        peft_kwargs=exp_config["model_config"]["peft_kwargs"],
        **exp_config["model_config"]["model_kwargs"],
        add_padding_token=exp_config["model_config"]["add_padding_token"],
    )

    # load experiment config from the retain model

    qa_formatter = QAFormatter(**exp_config["model_config"]["qa_formatter_kwargs"])
    loss_type = "standard"
    n_perturbed = 2
    random_seed = exp_config["seed"]

    # get splits
    forget_split, retain_split = get_data(
        data_config["dataset_name"],
        **data_config["data_kwargs"],
        random_seed=random_seed,
    )

    if args.experiment_2_eval:
        retain_model_dir = f"{retain_model_dir}/entity_subset_eval/"
    if args.train_set_eval:
        compare_eval = EvaluateOutputs.load(
            f"{retain_model_dir}/eval_outputs/"
            f"{experiment_config.config_names['data_config']}"
            "/train_set_eval_outputs.json"
        )
    else:
        compare_eval = EvaluateOutputs.load(f"{retain_model_dir}/eval_outputs.json")

    b_sz = exp_config["model_config"]["trainer_kwargs"]["per_device_eval_batch_size"]
    evaluator = Evaluator(
        model,
        forget_split,
        retain_split,
        qa_formatter,
        exp_config["data_config"]["dataset_name"],
        tokenizer,
        n_perturbed,
        random_seed,
        compare_eval.forget_truth_ratios,
        b_sz,
        train_set_eval=args.train_set_eval,
        max_new_tokens="adaptive",
    )

    eval_results = evaluator.evaluate()
    save_dir = (
        f"{full_model_dir}/eval_outputs/"
        f"{exp_config['config_names']['data_config']}/"
    )
    if args.experiment_2_eval:
        save_dir = f"{save_dir}/entity_subset_eval/"

    os.makedirs(save_dir, exist_ok=True)
    if args.train_set_eval:
        eval_results.save(f"{save_dir}/train_set_eval_outputs.json")
    else:
        eval_results.save(f"{save_dir}/eval_outputs.json")

    exp_name = exp_config["experiment_name"]
    print(f"\nBase Model path: {retain_model_dir}")
    print(f"Test Model path: {full_model_dir}")
    print(f"Experiment Name: {exp_name}")
    print(eval_results)
