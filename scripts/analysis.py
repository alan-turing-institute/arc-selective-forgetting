import argparse
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from arcsf.eval.metrics import ecdf

CONFIG_LOC = "configs/experiment"
OUTPUT_LOC = "output"


def open_json_path(path):
    with open(path) as file:
        json_object = json.load(file)
    return json_object


def plot_truth_ratios(data_experiment_map, forget_types, args):
    for data_name, experiment_numbers in tqdm(
        data_experiment_map.items(), "Truth ratio CDF"
    ):
        for experiment_number in experiment_numbers:
            retain_eval = open_json_path(
                glob(
                    f"{OUTPUT_LOC}/{args.experiment_name}"
                    f"/{experiment_number}/"
                    f"retain/*/eval_outputs.json"
                )[0]
            )
            exp_config = yaml.safe_load(
                open(
                    glob(
                        f"{OUTPUT_LOC}/{args.experiment_name}"
                        f"/{experiment_number}/"
                        f"retain/*/experiment_config.yaml"
                    )[0]
                )
            )
            retain_data, retain_y_values = ecdf(
                torch.tensor(retain_eval["forget_truth_ratios"])
            )
            plt.plot(retain_data, retain_y_values, "-", label="retain", color="k")

            full_eval = open_json_path(
                glob(
                    f"{OUTPUT_LOC}/{exp_config['full_model_name']}/full/*/"
                    f"eval_outputs/{data_name}/eval_outputs.json"
                )[0]
            )
            full_data, full_y_values = ecdf(
                torch.tensor(full_eval["forget_truth_ratios"])
            )
            plt.plot(full_data, full_y_values, "-.", label="full", color="k")

            for forget_type in forget_types:
                forget_eval = open_json_path(
                    glob(
                        f"{OUTPUT_LOC}/{args.experiment_name}"
                        f"/{experiment_number}/"
                        f"{forget_type}/*/eval_outputs.json"
                    )[0]
                )
                forget_data, forget_y_values = ecdf(
                    torch.tensor(forget_eval["forget_truth_ratios"])
                )
                plt.plot(
                    forget_data,
                    forget_y_values,
                    label=f"{forget_type}",
                )

            plt.legend()
            plt.xlim(0, 1)
            plt.savefig(
                f"{OUTPUT_LOC}/{args.experiment_name}/"
                f"{experiment_number}/truth_ratio_cdf.pdf"
            )
            plt.close()


def plot_eval_checkpoints(data_experiment_map, forget_types, args):
    for data_name, experiment_numbers in tqdm(
        data_experiment_map.items(), "Checkpoints"
    ):
        for ks_type in ["1", "2"]:
            for experiment_number in experiment_numbers:
                for forget_index, forget_type in enumerate(forget_types):
                    forget_quality = []
                    model_utility = []
                    checkpoint_glob = (
                        f"{OUTPUT_LOC}/{args.experiment_name}"
                        f"/{experiment_number}/"
                        f"{forget_type}/*/eval_checkpoints/*.json"
                    )
                    checkpoint_fps = glob(checkpoint_glob)
                    assert len(checkpoint_fps) > 0, "no eval checkpoints found"
                    for checkpoint_fp in checkpoint_fps:
                        epoch_eval = open_json_path(checkpoint_fp)
                        forget_quality.append(epoch_eval[f"forget_quality_{ks_type}"])
                        model_utility.append(epoch_eval["retain_model_utility"])
                    colour = f"C{forget_index}"
                    plt.scatter(
                        model_utility,
                        forget_quality,
                        marker="o",
                        s=[30 * (i + 1) for i in range(len(forget_quality))],
                        alpha=0.5,
                        label=forget_type,
                        color=colour,
                    )

                    alphas = np.linspace(0.1, 0.9, len(forget_quality))
                    for index, alpha in enumerate(alphas[:-1]):
                        plt.plot(
                            model_utility[index : index + 2],
                            forget_quality[index : index + 2],
                            "-",
                            alpha=alpha,
                            color=colour,
                        )
                retain_eval = open_json_path(
                    glob(
                        f"{OUTPUT_LOC}/{args.experiment_name}"
                        f"/{experiment_number}/"
                        f"retain/*/eval_outputs.json"
                    )[0]
                )
                retain_eval[f"forget_quality_{ks_type}"] = 0
                plt.plot(
                    retain_eval["retain_model_utility"],
                    retain_eval[f"forget_quality_{ks_type}"],
                    "D",
                    color="k",
                    label="retain",
                )
                exp_config = yaml.safe_load(
                    open(
                        glob(
                            f"output/{args.experiment_name}"
                            f"/{experiment_number}/"
                            f"retain/*/experiment_config.yaml"
                        )[0]
                    )
                )
                full_eval = open_json_path(
                    glob(
                        f"{OUTPUT_LOC}/{exp_config['full_model_name']}/full/*/"
                        f"eval_outputs/{data_name}/eval_outputs.json"
                    )[0]
                )
                plt.plot(
                    full_eval["retain_model_utility"],
                    full_eval[f"forget_quality_{ks_type}"],
                    "s",
                    label="full",
                    color="k",
                )

                plt.legend()
                plt.xlabel("Model Utility")
                plt.ylabel(f"Forget Quality ({ks_type} sided)")
                plt.yscale("symlog")
                plt.savefig(
                    f"output/{args.experiment_name}/"
                    f"{experiment_number}/results_{ks_type}.pdf"
                )
                plt.close()


class MetricGetter:
    def __init__(self, exp_data_map, output_path, n_seeds, forget_methods):
        self.exp_data_map = exp_data_map
        self.output_path = output_path
        self.forget_methods = forget_methods
        self.n_seeds = n_seeds

    def __call__(self, data_split):
        all_types = ["retain"] + self.forget_methods
        raw_results = {
            key: {
                "forget_quality_1": [],
                "forget_quality_2": [],
                "retain_model_utility": [],
            }
            for key in all_types + ["full"]
        }
        for experiment_n in self.exp_data_map[data_split]:
            for exp_type in all_types:
                eval_outputs = open_json_path(
                    glob(
                        f"{self.output_path}/"
                        f"{experiment_n}/"
                        f"{exp_type}/*/eval_outputs.json"
                    )[0]
                )
                for metric in raw_results[exp_type].keys():
                    raw_results[exp_type][metric].append(eval_outputs[metric])
        for full_model_index in range(self.n_seeds):
            eval_outputs = open_json_path(
                glob(
                    f"{self.output_path}/"
                    f"full_{full_model_index}/full/*"
                    f"/eval_outputs/{data_split}/eval_outputs.json"
                )[0]
            )
            for metric in raw_results["full"].keys():
                raw_results["full"][metric].append(eval_outputs[metric])

        average_results = {}
        for model, model_metrics in raw_results.items():
            model_dict = {}
            for metric, raw_values in model_metrics.items():
                for value_index in range(len(raw_values)):
                    if raw_values[value_index] is None:
                        raw_values[value_index] = 0
                raw_values = np.array(raw_values)
                model_dict[metric] = {
                    "mean": np.mean(raw_values),
                    "std": np.std(raw_values),
                }
            average_results[model] = model_dict

        return average_results, raw_results


def main(args):
    args = args
    experiment_config = yaml.safe_load(
        open(f"{CONFIG_LOC}/{args.experiment_name}.yaml")
    )

    combinations = experiment_config["combinations"]
    forget_configs = experiment_config["combinations"]["forget_config"]
    forget_types = [forget_config[0] for forget_config in forget_configs]
    experiment_data_map = {split: [] for split in combinations["data_config"]}
    exp_number = 0
    for _ in combinations["seed"]:
        for data_split in combinations["data_config"]:
            experiment_data_map[data_split].append(f"experiment_{exp_number}")
            exp_number += 1

    fp = f"{OUTPUT_LOC}/{args.experiment_name}"
    os.makedirs(f"{fp}/results/", exist_ok=True)
    results_getter = MetricGetter(
        experiment_data_map, fp, 1, forget_methods=forget_types
    )

    if args.plotting_eval_checkpoints:
        plot_eval_checkpoints(experiment_data_map, forget_types, args)
    if args.plot_cdf:
        plot_truth_ratios(experiment_data_map, forget_types, args)

    for data_name in tqdm(experiment_data_map.keys(), desc="Data Split"):
        for ks_type in ["1", "2"]:
            avg_results, raw_results = results_getter(data_name)
            result_path = f"{fp}/results/{data_name}"
            os.makedirs(result_path, exist_ok=True)

            with open(f"{result_path}/average_results.json", "w") as result_file:
                json.dump(avg_results, result_file, indent=2)

            with open(f"{result_path}/raw_results.json", "w") as result_file:
                json.dump(raw_results, result_file, indent=2)

            plt.figure()
            for forget_method in results_getter.forget_methods:
                plt.scatter(
                    avg_results[forget_method]["retain_model_utility"]["mean"],
                    avg_results[forget_method][f"forget_quality_{ks_type}"]["mean"],
                    label=forget_method,
                )
            plt.scatter(
                avg_results["full"]["retain_model_utility"]["mean"],
                avg_results["full"][f"forget_quality_{ks_type}"]["mean"],
                marker="s",
                label="full",
                color="k",
            )
            plt.scatter(
                avg_results["retain"]["retain_model_utility"]["mean"],
                avg_results["retain"][f"forget_quality_{ks_type}"]["mean"],
                marker="D",
                label="retain",
                color="k",
            )
            plt.legend()
            plt.yscale("symlog")
            plt.xlabel("Model Utility")
            plt.ylabel(f"Forget Quality ({ks_type} sided)")
            plt.savefig(f"{result_path}/result_plot_{ks_type}.pdf")
            plt.close()

    os.makedirs(f"{fp}/results/experiment_results/", exist_ok=True)

    sizes = [int(config.split("_")[-1]) for config in combinations["data_config"]]
    granularities = ["question", "book", "author", "publisher"]
    marker_sizes = 20 * np.cumsum(np.arange(len(granularities)) + 1)
    for size in tqdm(sizes, desc="Granularity Experiment"):
        for ks_type in ["1", "2"]:
            plt.figure()
            for forget_method in results_getter.forget_methods:
                forget_data = np.zeros(len(granularities))
                utility_data = np.zeros(len(granularities))
                for granularity_index, granularity in enumerate(granularities):
                    data_name = f"gen_tofu_{granularity}_{size}"
                    data = open_json_path(
                        f"{fp}/results/{data_name}/average_results.json"
                    )
                    forget_data[granularity_index] = data[forget_method][
                        f"forget_quality_{ks_type}"
                    ]["mean"]
                    utility_data[granularity_index] = data[forget_method][
                        "retain_model_utility"
                    ]["mean"]

                plt.scatter(
                    utility_data,
                    forget_data,
                    marker="o",
                    s=marker_sizes,
                    alpha=0.3,
                    label=forget_method,
                )

            for base_model, marker_shape in zip(["retain", "full"], ["D", "s"]):
                forget_data = np.zeros(len(granularities))
                utility_data = np.zeros(len(granularities))
                for granularity_index, granularity in enumerate(granularities):
                    data_name = f"gen_tofu_{granularity}_{size}"
                    data = open_json_path(
                        f"{fp}/results/{data_name}/average_results.json"
                    )
                    forget_data[granularity_index] = data[base_model][
                        f"forget_quality_{ks_type}"
                    ]["mean"]
                    utility_data[granularity_index] = data[base_model][
                        "retain_model_utility"
                    ]["mean"]

                plt.scatter(
                    utility_data,
                    forget_data,
                    marker=marker_shape,
                    s=marker_sizes,
                    alpha=0.3,
                    label=base_model,
                    color="k",
                )

            colour_legend = plt.legend(title="Forget Type")
            plt.gca().add_artist(colour_legend)
            h = [
                plt.plot([], [], color="gray", marker="o", alpha=0.3, ms=i / 20, ls="")[
                    0
                ]
                for i in (marker_sizes + 60)
            ]
            size_legend = plt.legend(
                handles=h,
                labels=granularities,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=4,
                title="Granularity",
            )
            plt.gca().add_artist(size_legend)
            plt.yscale("symlog")
            plt.xlabel("Model Utility")
            plt.ylabel(f"Forget Quality ({ks_type} sided)")
            plt.savefig(f"{fp}/results/experiment_results/{size}_percent_{ks_type}.pdf")
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "From an experiment path generates evaluation plots for every experiment."
        )
    )
    parser.add_argument(
        "experiment_name",
        type=str,
        help="Experiment name",
    )

    parser.add_argument("--plotting_eval_checkpoints", action="store_true")
    parser.add_argument("--plot_cdf", action="store_true")
    args = parser.parse_args()
    main(args)
