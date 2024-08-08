import argparse
import os

import yaml

from arcsf.data.data_module import QAFormatter, get_data
from arcsf.eval.evaluate import EvaluateOutputs, Evaluator
from arcsf.models.model import load_model_and_tokenizer
from arcsf.utils import get_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Runs evaluation, for full models."))
    parser.add_argument(
        "experiment_name", type=str, help="Relative path to retain model."
    )

    args = parser.parse_args()
    experiment_name = args.experiment_name
    retain_model_dir = get_model_path(experiment_name, "retain")

    exp_config = yaml.safe_load(open(f"{retain_model_dir}/experiment_config.yaml"))

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
        exp_config["data_config"]["dataset_name"],
        **exp_config["data_config"]["data_kwargs"],
        random_seed=random_seed,
    )

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
        max_new_tokens="adaptive",
    )

    eval_results = evaluator.evaluate()
    save_dir = (
        f"{full_model_dir}/eval_outputs/"
        f"{exp_config['config_names']['data_config']}/"
    )
    os.makedirs(save_dir)
    eval_results.save(f"{save_dir}/eval_outputs.json")

    exp_name = exp_config["experiment_name"]
    print(f"\nBase Model path: {retain_model_dir}")
    print(f"Test Model path: {full_model_dir}")
    print(f"Experiment Name: {exp_name}")
    print(eval_results)
