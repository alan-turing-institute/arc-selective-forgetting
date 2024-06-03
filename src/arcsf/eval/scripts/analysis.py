import argparse
import json
import os

import yaml

from arcsf.eval.utils import get_analysis_values, get_metrics, plot_cdf


def run_analysis(
    base_model_path: str,
    test_model_path: str,
    verbose: bool = False,
    plotting: bool = False,
) -> dict[str, float]:
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

    base_values = get_analysis_values(base_model_path)
    test_values = get_analysis_values(test_model_path)

    result_dict = get_metrics(base_values["forget_tr"], test_values)
    result_dict.update({"experiment_name": exp_name})

    if verbose:
        print(f"\nBase Model path: {base_model_path}")
        print(f"Test Model path: {test_model_path}")
        print(f"Experiment Name: {exp_name}")
        print(f"Mean Retain Truth Ratio: {result_dict['mean_tr_retain']}")
        print(f"Retain Rouge: {result_dict['mean_rouge_score']}")
        print(f"Forget Quality One Sided: {result_dict['forget_quality_1']}")
        print(f"Forget Quality Two Sided: {result_dict['forget_quality_2']}")
        print(f"Model Utility: {result_dict['model_utility']}")
    if plotting:
        plot_cdf(base_values, test_values, test_model_path, exp_name, "retain")
        plot_cdf(base_values, test_values, test_model_path, exp_name, "forget")

    return result_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Runs analysis of specific model, returns forget quality and model utility"
        )
    )
    parser.add_argument(
        "test_model_dir",
        type=str,
        help="Relative path to directory containing test model metrics.",
    )
    parser.add_argument(
        "base_model_dir",
        type=str,
        help="Relative path to directory containing base model metrics.",
    )
    args = parser.parse_args()
    results = run_analysis(args.base_model_dir, args.test_model_dir)
    with open(f"{args.test_model_dir}/eval/analysis/metrics.json", "w") as f:
        json.dump(results, f)
