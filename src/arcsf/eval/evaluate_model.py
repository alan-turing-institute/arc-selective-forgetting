import argparse
import json
import os
import random

import numpy as np
import torch
import transformers
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from arcsf.data.data_module import BlankQAFormatter, EvalQADataset, get_data
from arcsf.eval.utils import all_eval, combine_dicts, get_metrics
from arcsf.utils import get_device


def evaluate_model(
    model: torch.nn.Module,
    base_truth_ratios_path: str,
    tokenizer: transformers.AutoTokenizer,
    experiment_config: dict,
    random_seed: int,
    **generate_kwargs: dict,
) -> dict[str, float]:
    """
    Evaluates the model against a baseline model, given the model, tokenizer, experiment
    config and the path to the baseline model truth ratios.

    Args:
        model: Model to perform the evaluation on
        base_truth_ratios_path: path for the baseline mode truth ratios
        tokenizer: tokenizer being used for the model
        experiment_config: experiment config (currently a dictionary)

    Returns:
        dictionary of aggregated metrics for evaluation (generated using `get_metrics`)
    """
    # get datasets
    forget_data, retain_data = get_data(
        "tofu", random_seed=random_seed, **experiment_config["data_config"]
    )
    # get available device
    device = get_device()

    # create datasets
    retain_dataset = EvalQADataset(
        retain_data,
        tokenizer,
        BlankQAFormatter(),
        "standard",
        quantitative_eval=True,
        qualitative_eval=True,
        device=device,
        random_seed=random_seed,
    )
    forget_dataset = EvalQADataset(
        forget_data,
        tokenizer,
        BlankQAFormatter(),
        "standard",
        quantitative_eval=True,
        qualitative_eval=True,
        device=device,
        random_seed=random_seed,
    )

    # perform evaluation on models to get the raw values
    retain_values = all_eval(
        model,
        retain_dataset,
        1,
        device,
        tokenizer,
        **generate_kwargs,
    )
    forget_values = all_eval(
        model,
        forget_dataset,
        1,
        device,
        tokenizer,
        **generate_kwargs,
    )

    # combine dictionaries and load the baseline model truth ratios
    analysis_vals = combine_dicts(forget_values, retain_values)
    base_truth_ratios = torch.tensor(np.loadtxt(base_truth_ratios_path))

    # return the aggregated metrics using the get metrics function
    return get_metrics(base_truth_ratios, analysis_vals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs qualitative evaluation, comparing outputs of model"
            " against target strings."
        )
    )
    parser.add_argument(
        "model_path", type=str, help="Relative path to model directory."
    )
    parser.add_argument(
        "base_vals_path", type=str, help="Relative path to base model truth ratios."
    )
    parser.add_argument(
        "--random_seed",
        "-r",
        default=None,
        type=int,
        help="Random seed for script",
    )

    args = parser.parse_args()
    model_dir = args.model_path

    # these are hardcoded for now
    if args.random_seed:
        rand = args.random_seed
    else:
        rand = random.randint(-10000, 10000)

    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model.config.pad_token_id = tokenizer.eos_token_id
    exp_config = yaml.safe_load(open(f"{model_dir}/experiment_config.yaml"))

    vals = evaluate_model(
        model,
        args.base_vals_path,
        tokenizer,
        exp_config,
        random_seed=rand,
        max_new_tokens=50,
    )
    save_dir = f"{model_dir}/eval/analysis/"
    os.makedirs(save_dir, exist_ok=True)
    print(vals)
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(vals, f)