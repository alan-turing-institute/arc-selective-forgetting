import argparse
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

CONFIG_LOC = "configs/experiment"


def open_json_path(path):
    with open(path) as file:
        json_object = json.load(file)
    return json_object


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
                raw_values = np.array(raw_values)
                raw_values[raw_values is None] = 0
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
    experiment_data_map = {split: [] for split in combinations["data_config"]}
    exp_number = 0
    for _ in combinations["seed"]:
        for data_split in combinations["data_config"]:
            experiment_data_map[data_split].append(f"experiment_{exp_number}")
            exp_number += 1

    fp = f"output/{args.experiment_name}"
    os.makedirs(f"{fp}/results/", exist_ok=True)
    results_getter = MetricGetter(
        experiment_data_map, fp, 1, forget_methods=args.forget_types
    )
    for data_name in tqdm(experiment_data_map.keys(), desc="Experiment No."):
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
                avg_results[forget_method]["forget_quality_1"]["mean"],
                label=forget_method,
            )
        plt.scatter(
            avg_results["full"]["retain_model_utility"]["mean"],
            avg_results["full"]["forget_quality_1"]["mean"],
            marker="s",
            label="full",
            color="k",
        )
        plt.scatter(
            avg_results["retain"]["retain_model_utility"]["mean"],
            avg_results["retain"]["forget_quality_1"]["mean"],
            marker="D",
            label="retain",
            color="k",
        )
        plt.legend()
        plt.yscale("symlog")
        plt.savefig(f"{result_path}/result_plot.pdf")
        plt.close()

    os.makedirs(f"{fp}/results/experiment_results/", exist_ok=True)

    sizes = args.forget_sizes
    granularities = ["question", "book", "author", "publisher"]

    for size in sizes:
        plt.figure()
        for forget_method in results_getter.forget_methods:
            forget_data = np.zeros(len(granularities))
            utility_data = np.zeros(len(granularities))
            marker_sizes = 20 * np.cumsum(np.arange(len(granularities)) + 1)
            for granularity_index, granularity in enumerate(granularities):
                data_name = f"gen_tofu_{granularity}_{size}"
                data = open_json_path(f"{fp}/results/{data_name}/average_results.json")
                forget_data[granularity_index] = data[forget_method][
                    "forget_quality_1"
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

        colour_legend = plt.legend(title="Forget Type", loc=(0.45, 0.1))
        plt.gca().add_artist(colour_legend)
        h = [
            plt.plot([], [], color="gray", marker="o", alpha=0.3, ms=i / 20, ls="")[0]
            for i in (marker_sizes + 60)
        ]
        size_legend = plt.legend(
            handles=h, labels=granularities, loc=(0.7, 0.1), title="Granularity"
        )
        plt.gca().add_artist(size_legend)
        plt.yscale("symlog")
        plt.savefig(f"{fp}/results/experiment_results/{size}_percent.pdf")
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
    parser.add_argument(
        "--forget_types",
        type=str,
        nargs="+",
        default=["idk", "kl", "ascent", "difference"],
    )
    parser.add_argument(
        "--forget_sizes",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25],
    )
    args = parser.parse_args()
    main(args)
