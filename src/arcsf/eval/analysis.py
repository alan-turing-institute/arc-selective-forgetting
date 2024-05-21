import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import hmean

from arcsf.eval.metrics import ks_test, truth_ratio
from arcsf.eval.utils import ecdf


def get_values(
    model_dir: str,
) -> dict[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Gets the values for analysis given a model directory.

    Args:
        model_dir : file path to where model is stored with evaluation output

    Returns:
        vals : dictionary containing the values to be used in analysis
    """
    vals = dict()
    vals["forget_losses"] = np.loadtxt(model_dir + "/eval/forget/all_losses.txt")
    vals["retain_losses"] = np.loadtxt(model_dir + "/eval/retain/all_losses.txt")
    vals["rouge_scores"] = np.loadtxt(model_dir + "/eval/retain/rougeL_scores.txt")
    # we re-calculate the truth ratio, since numpy stored many as NaNs
    vals["forget_tr"] = truth_ratio(torch.tensor(vals["forget_losses"]))
    vals["retain_tr"] = torch.clamp(
        1 - truth_ratio(torch.tensor(vals["retain_losses"])),
        min=0,
    )
    return vals


def plot_cdf(
    base_vals: dict,
    test_vals: dict,
    save_path: str,
    exp_name: str,
    split: str = "retain",
) -> None:
    """
    Function to plot and save CDF functions for model comparison

    Args:
        base_vals : values for the base model to be compared against
        test_vals : values for the test model to be compared
        save_path : relative path where the model is stored
        exp_name : the name of the experiment for naming purposes
        split (optional): the split to be used. Defaults to "retain".
    """
    base_data, base_y_values = ecdf(base_vals[f"{split}_tr"])
    test_data, test_y_values = ecdf(test_vals[f"{split}_tr"])

    p_val = ks_test(test_vals[f"{split}_tr"], base_vals[f"{split}_tr"])

    plt.title(f"{split} data CDF")
    plt.plot(base_data, base_y_values, label="base-model")
    plt.plot(test_data, test_y_values, label=f"forget-model {exp_name}")
    plt.annotate(f"Model Utility {np.log(p_val)}", (0.9, 0.1))
    plt.legend()
    plt.savefig(save_path + f"/eval/analysis/tr-cdf-{split}-{exp_name}-.pdf")
    plt.close()


def plot_scatter(dict: dict[float], **plot_kwargs) -> None:
    """
    Plot of the data from a given model/experiment

    Args:
        dict : dictionary containing the aggrevated evaluation metrics of the model.
        plot_kwargs : key word arguments for the `matplotlib.pylot.scatter` function.
    """
    plt.scatter(dict["model_utility"], dict["forget_quality"], **plot_kwargs)


def run_analysis(
    base_model_path: str,
    test_model_path: str,
    verbose: bool = False,
    plotting: bool = False,
) -> dict[str, float, float, float, float]:
    """
    Run analysis on a model from a path comparing against a base model from a designated
    path. It returns aggregated metrics for evaluation.

    Args:
        base_model_path: path of the comparison model which is trained on the finetuned
        data
        test_model_path: path of the model to test
        verbose: Boolean value which denotes whether or not the function should print as
        it runs. Defaults to False.
        plotting: Should the KS-Test CDF functions be plotted? Defaults to False.

    Returns:
        result_dict : result dictionary containing aggregated metrics for the model, it
        contains:
            forget_quality : log of the pvalue from truth ratio ks-test comparison
            retain_tr : truth ratio on the retain data
                (defined as max(1-truth_ratio) in the paper)
            rouge_score : mean rouge score
            model_utility : harmonic mean of the retain truth ratio and rouge score
    """
    os.makedirs(test_model_path + "/eval/analysis", exist_ok=True)

    experiment_config = yaml.safe_load(
        open(test_model_path + "/experiment_config.yaml")
    )
    exp_name = "-".join(experiment_config["experiment_name"].split("-")[1:])
    if "train_type" in experiment_config.keys():
        train_type = experiment_config["train_type"]
        exp_name = exp_name + "_" + train_type

    base_values = get_values(base_model_path)
    test_values = get_values(test_model_path)

    forget_quality = np.log(ks_test(base_values["forget_tr"], test_values["forget_tr"]))

    retain_tr = torch.mean(test_values["retain_tr"]).item()
    rouge_score = np.mean(test_values["rouge_scores"])
    model_utilty = hmean([retain_tr, rouge_score])

    if verbose:
        print(f"\nBase Model path: {base_model_path}")
        print(f"Test Model path: {test_model_path}")
        print(f"Experiment Name: {exp_name}")
        print(f"Retain Truth Ratio: {retain_tr}")
        print(f"Retain Rouge: {rouge_score}")
        print(f"Forget Quality: {forget_quality}")
    if plotting:
        plot_cdf(base_values, test_values, test_model_path, exp_name, "retain")
        plot_cdf(base_values, test_values, test_model_path, exp_name, "forget")

    result_dict = {
        "experiment_name": exp_name,
        "retain_tr": retain_tr,
        "rouge_score": rouge_score,
        "forget_quality": forget_quality,
        "model_utility": model_utilty,
    }

    return result_dict


if __name__ == "__main__":
    base_dir = "temp/20240507-233700-957103"

    all_results = {}

    dirs = os.listdir("temp")
    for dir in dirs:
        try:
            results = run_analysis(base_dir, "temp/" + dir)
        except FileNotFoundError:
            continue
        all_results[results.pop("experiment_name")] = results

    with open("temp/results/all_results.json", "w") as f:
        json.dump(all_results, f)

    finetune = all_results.pop("gpt2_longer_all")
    plot_scatter(finetune, label="Finetune", marker="s", color="k")

    retain = all_results.pop("gpt2_longer_retain")
    plot_scatter(retain, label="retain", marker="+", color="k")

    for idx, name in enumerate(list(all_results.keys())):
        plot_scatter(all_results[name], label=name)

    plt.xlabel("model utility")
    plt.ylabel("forget quality (log $p$-value)")
    plt.legend()
    plt.savefig("temp/results/all_models.pdf")
    plt.close()
