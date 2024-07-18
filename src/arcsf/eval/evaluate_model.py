"""
TODO
All Eval -> Evaluate model / Evaluate truth ratios / Compute metrics ?
    input: model, forget & retain datasets
    output: dict of truth ratios, rouge, losses (forget & retain)
Evaluate model -> Evaluate forget quality?
    input: 2 model all eval metrics (base/gold standard(?) and compare)
    output: forget quality, utility etc.
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from arcsf.data.data_module import BlankQAFormatter, EvalQADataset, get_data
from arcsf.eval.metrics import get_metrics
from arcsf.eval.utils import all_eval, combine_dicts


# TODO - for better compatibility with evaluate
def EVALUATE(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    forget_dataset: EvalQADataset,
    retain_dataset: EvalQADataset,
    base_truth_ratios: torch.Tensor,
    batch_size: int,
    n_print: int = 5,
    accelerator: Accelerator = Accelerator(),
    **generate_kwargs,
):
    print("Evaluating forget data...")
    forget_metrics = all_eval(
        model,
        forget_dataset,
        batch_size,
        tokenizer,
        n_print=n_print,
        accelerator=accelerator,
        **generate_kwargs,
    )

    print("Evaluating retain data...")
    retain_metrics = all_eval(
        model,
        retain_dataset,
        batch_size,
        tokenizer,
        n_print=n_print,
        accelerator=accelerator,
        **generate_kwargs,
    )

    print("Calculating metrics...")
    return get_metrics(base_truth_ratios, forget_metrics, retain_metrics)


def evaluate_model(
    model: torch.nn.Module,
    base_truth_ratios_path: str,
    tokenizer: PreTrainedTokenizer,
    experiment_config: dict,
    batch_size: int,
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
        batch_size: batch size for the all_eval function

    Returns:
        dictionary of aggregated metrics for evaluation (generated using `get_metrics`)
    """
    random_seed = experiment_config["seed"]

    # get datasets
    forget_data, retain_data = get_data(
        experiment_config["data_config"]["dataset_name"],
        **experiment_config["data_config"]["data_kwargs"],
        random_seed=random_seed,
    )

    # create datasets
    retain_dataset = EvalQADataset(
        retain_data,
        tokenizer,
        BlankQAFormatter(),
        "standard",
        n_perturbed=2,
        random_seed=random_seed,
    )
    forget_dataset = EvalQADataset(
        forget_data,
        tokenizer,
        BlankQAFormatter(),
        "standard",
        n_perturbed=2,
        random_seed=random_seed,
    )

    # perform evaluation on models to get the raw values
    retain_values = all_eval(
        model,
        retain_dataset,
        batch_size,
        tokenizer,
        **generate_kwargs,
    )
    forget_values = all_eval(
        model,
        forget_dataset,
        batch_size,
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
        "-b",
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size for the evaluation pipeline to use. Defaults to 16.",
    )

    args = parser.parse_args()
    model_dir = args.model_path

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.config.pad_token_id = tokenizer.eos_token_id
    exp_config = yaml.safe_load(open(f"{model_dir}/experiment_config.yaml"))

    vals = evaluate_model(
        model,
        args.base_vals_path,
        tokenizer,
        exp_config,
        batch_size=args.eval_batch_size,
        max_new_tokens=50,
    )
    save_dir = f"{model_dir}/eval/analysis/"
    os.makedirs(save_dir, exist_ok=True)
    print(vals)
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(vals, f)
