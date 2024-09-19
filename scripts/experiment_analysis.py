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

plt.rcParams["font.size"] = 10


def open_json_path(path):
    with open(path) as file:
        json_object = json.load(file)
    return json_object


def locate_forget_length(glob_string, length):
    # This ensures that we only select from directories with a .json file in
    options = glob(f"{glob_string}/*eval_outputs.json")
    options = ["/".join(option.split("/")[:-1]) for option in options]

    train_length = np.zeros(len(options))
    for opt_idx, option_path in enumerate(options):
        train_length[opt_idx] = yaml.safe_load(
            open(f"{option_path}/experiment_config.yaml")
        )["model_config"]["trainer_kwargs"]["num_train_epochs"]
    if length == "short":
        return options[np.argmin(train_length)]
    elif length == "long":
        return options[np.argmax(train_length)]


def plot_truth_ratios(data_experiment_map, forget_types, args, train_set=False):
    for data_name, experiment_numbers in tqdm(
        data_experiment_map.items(), "Truth ratio CDF"
    ):
        for train_set in [False, True]:
            for split in ["retain", "forget"]:
                for experiment_number in experiment_numbers:
                    retain_eval = open_json_path(
                        glob(
                            f"{OUTPUT_LOC}/{args.experiment_name}"
                            f"/{experiment_number}/"
                            f"retain/*/eval_outputs/{data_name}/"
                            f"{train_set*'train_set_'}eval_outputs.json"
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
                        torch.tensor(retain_eval[f"{split}_truth_ratios"])
                    )
                    plt.plot(
                        retain_data, retain_y_values, "-", label="retain", color="k"
                    )

                    full_eval = open_json_path(
                        glob(
                            f"{OUTPUT_LOC}/{exp_config['full_model_name']}/full/*/"
                            f"eval_outputs/{data_name}/{train_set*'train_set_'}"
                            "eval_outputs.json"
                        )[0]
                    )
                    full_data, full_y_values = ecdf(
                        torch.tensor(full_eval[f"{split}_truth_ratios"])
                    )
                    plt.plot(full_data, full_y_values, "-.", label="full", color="k")

                    for forget_type in forget_types:
                        forget_eval = open_json_path(
                            glob(
                                f"{OUTPUT_LOC}/{args.experiment_name}"
                                f"/{experiment_number}/{forget_type}/*/"
                                f"eval_outputs/{data_name}/{train_set*'train_set_'}"
                                "eval_outputs.json"
                            )[0]
                        )
                        forget_data, forget_y_values = ecdf(
                            torch.tensor(forget_eval[f"{split}_truth_ratios"])
                        )
                        plt.plot(
                            forget_data,
                            forget_y_values,
                            label=f"{forget_type}",
                        )

                    plt.legend()
                    plt.xlim(0, 1)
                    plt.ylabel("Cumulative Probability")
                    plt.xlabel("Truth Ratio")
                    plt.tight_layout()
                    plt.savefig(
                        f"{OUTPUT_LOC}/{args.experiment_name}/"
                        f"{experiment_number}/truth_ratio_cdf_{split}"
                        f"{train_set*'_train'}.pdf"
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
                        model_utility.append(epoch_eval["retain_model_utility_3"])
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
                    retain_eval["retain_model_utility_3"],
                    0,
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
                    full_eval["retain_model_utility_3"],
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


def plot_relationship(
    color, marker, forget_method, base_results, subset_results, ks_type
):

    base_fq = base_results[forget_method][f"forget_quality_{ks_type}"]["mean"]
    base_mu = base_results[forget_method]["retain_model_utility_3"]["mean"]

    subset_fq = subset_results[forget_method][f"forget_quality_{ks_type}"]["mean"]
    subset_mu = subset_results[forget_method]["retain_model_utility_3"]["mean"]

    if forget_method == "retain":
        base_fq = 0
        subset_fq = 0

    plt.scatter(
        base_mu,
        base_fq,
        s=100,
        alpha=0.5,
        marker=marker,
        color=color,
        edgecolors=color,
        label=forget_method.capitalize(),
    )
    plt.scatter(
        subset_mu,
        subset_fq,
        marker=marker,
        s=75,
        alpha=0.5,
        facecolor="none",
        edgecolors=color,
    )
    plt.plot(
        np.array(
            [
                base_mu,
                subset_mu,
            ],
        ),
        np.array(
            [
                base_fq,
                subset_fq,
            ]
        ),
        "--",
        alpha=0.3,
        color=color,
    )


class MetricGetter:
    def __init__(
        self, exp_data_map, output_path, n_seeds, forget_methods, forget_length
    ):
        self.exp_data_map = exp_data_map
        self.output_path = output_path
        self.forget_methods = forget_methods
        self.n_seeds = n_seeds
        self.forget_length = forget_length

    def __call__(
        self,
        data_split,
        subset_results=False,
        subset_non_entity_results=False,
        train_results=False,
    ):
        all_types = ["retain"] + self.forget_methods
        raw_results = {
            key: {
                "forget_quality_1": [],
                "forget_quality_2": [],
                "retain_model_utility": [],
                "retain_model_utility_3": [],
                "forget_rougeL": [],
                "retain_rougeL": [],
            }
            for key in all_types + ["full"]
        }
        if train_results:
            filename = "train_set_eval_outputs.json"
        else:
            filename = f"{subset_non_entity_results*'non_entity_'}eval_outputs.json"

        for experiment_n in self.exp_data_map[data_split]:
            for exp_type in all_types:
                output_location = f"{exp_type}/*"  # /eval_outputs/{data_split}"
                eval_path = locate_forget_length(
                    glob_string=f"{self.output_path}/{experiment_n}/{output_location}",
                    length=self.forget_length,
                )
                if subset_results:
                    eval_path = (
                        f"{eval_path}/eval_outputs/{data_split}/entity_subset_eval"
                    )

                eval_outputs = open_json_path(f"{eval_path}/{filename}")
                for metric in [
                    "forget_quality_1",
                    "forget_quality_2",
                    "retain_model_utility",
                    "retain_model_utility_3",
                ]:
                    raw_results[exp_type][metric].append(eval_outputs[metric])
                raw_results[exp_type]["forget_rougeL"].append(
                    np.mean(eval_outputs["forget_rougeL_recall"])
                )
                raw_results[exp_type]["retain_rougeL"].append(
                    np.mean(eval_outputs["retain_rougeL_recall"])
                )

        for full_model_index in range(self.n_seeds):
            output_location = (
                f"{self.output_path}/"
                f"full_{full_model_index}/full/*"
                f"/eval_outputs/{data_split}/"
            )
            if subset_results:
                output_location = f"{output_location}/entity_subset_eval"
            eval_outputs = open_json_path(glob(f"{output_location}/{filename}")[0])
            for metric in [
                "forget_quality_1",
                "forget_quality_2",
                "retain_model_utility",
                "retain_model_utility_3",
            ]:
                raw_results["full"][metric].append(eval_outputs[metric])
            raw_results["full"]["forget_rougeL"].append(
                np.mean(eval_outputs["forget_rougeL_recall"])
            )
            raw_results["full"]["retain_rougeL"].append(
                np.mean(eval_outputs["retain_rougeL_recall"])
            )

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
        experiment_data_map,
        fp,
        len(combinations["seed"]),
        forget_methods=forget_types,
        forget_length=args.forget_length,
    )

    if args.plotting_eval_checkpoints:
        plot_eval_checkpoints(experiment_data_map, forget_types, args)
    if args.plot_truth_ratio_cdf:
        plot_truth_ratios(experiment_data_map, forget_types, args)
    for data_name in tqdm(experiment_data_map.keys(), desc="Data Split"):
        for train_set in [False]:
            for ks_type in ["1", "2"]:
                avg_results, raw_results = results_getter(
                    data_name, train_results=train_set
                )
                result_path = f"{fp}/results/{data_name}"
                os.makedirs(result_path, exist_ok=True)
                with open(
                    f"{result_path}/average_results{train_set*'_train'}.json", "w"
                ) as result_file:
                    json.dump(avg_results, result_file, indent=2)

                with open(
                    f"{result_path}/raw_results{train_set*'_train'}.json", "w"
                ) as result_file:
                    json.dump(raw_results, result_file, indent=2)

                plt.figure()
                for forget_method in results_getter.forget_methods:
                    plt.scatter(
                        avg_results[forget_method]["retain_model_utility_3"]["mean"],
                        avg_results[forget_method][f"forget_quality_{ks_type}"]["mean"],
                        label=forget_method,
                    )
                plt.scatter(
                    avg_results["full"]["retain_model_utility_3"]["mean"],
                    avg_results["full"][f"forget_quality_{ks_type}"]["mean"],
                    marker="s",
                    label="full",
                    color="k",
                )
                plt.scatter(
                    avg_results["retain"]["retain_model_utility_3"]["mean"],
                    0,
                    marker="D",
                    label="retain",
                    color="k",
                )
                plt.legend()
                plt.yscale("symlog")
                plt.xlabel("Model Utility")
                plt.ylabel(f"Forget Quality ({ks_type} sided)")
                plt.savefig(
                    f"{result_path}/result_plot_{ks_type}{train_set*'_train'}.pdf"
                )
                plt.close()

                plt.figure()
                results_getter.forget_methods = list(set(results_getter.forget_methods))
                for forget_method in results_getter.forget_methods:
                    plt.scatter(
                        avg_results[forget_method]["retain_rougeL"]["mean"],
                        avg_results[forget_method]["forget_rougeL"]["mean"],
                        label=forget_method,
                    )
                plt.scatter(
                    avg_results["full"]["retain_rougeL"]["mean"],
                    avg_results["full"]["forget_rougeL"]["mean"],
                    marker="s",
                    label="full",
                    color="k",
                )
                plt.scatter(
                    avg_results["retain"]["retain_rougeL"]["mean"],
                    avg_results["retain"]["forget_rougeL"]["mean"],
                    marker="D",
                    label="retain",
                    color="k",
                )
                plt.xlabel("Retain ROUGE")
                plt.ylabel("Forget ROUGE")
                plt.legend(title="Model type")
                plt.savefig(f"{result_path}/rouge_plot{train_set*'_train'}.pdf")
                plt.close()

    os.makedirs(f"{fp}/results/experiment_results/", exist_ok=True)

    if args.experiment_type == "granularity":
        sizes = [int(config.split("_")[-1]) for config in combinations["data_config"]]
        sizes = sizes[1:]
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
                            "retain_model_utility_3"
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
                            "retain_model_utility_3"
                        ]["mean"]
                    if base_model == "retain":
                        forget_data = [0] * len(utility_data)
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
                    plt.plot(
                        [], [], color="gray", marker="o", alpha=0.3, ms=i / 20, ls=""
                    )[0]
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
                plt.xlim(left=-0.05)
                plt.savefig(
                    f"{fp}/results/experiment_results/{size}_percent_"
                    f"{ks_type}_{args.forget_length}.pdf"
                )
                plt.close()

    if args.experiment_type == "relationship":
        for data_name in tqdm(experiment_data_map.keys(), desc="Data split"):
            for ks_type in ["1", "2"]:
                entity_avg_results, entity_raw_results = results_getter(
                    data_name, subset_results=True
                )
                result_path = f"{fp}/results/experiment_results/{data_name}"
                os.makedirs(result_path, exist_ok=True)

                with open(
                    f"{result_path}/entity_subset_average_results.json", "w"
                ) as result_file:
                    json.dump(entity_avg_results, result_file, indent=2)

                with open(
                    f"{result_path}/entity_subset_raw_results.json", "w"
                ) as result_file:
                    json.dump(entity_raw_results, result_file, indent=2)

                non_entity_avg_results, non_entity_raw_results = results_getter(
                    data_name, subset_results=True, subset_non_entity_results=True
                )
                result_path = f"{fp}/results/experiment_results/{data_name}"
                os.makedirs(result_path, exist_ok=True)

                with open(
                    f"{result_path}/non_entity_subset_average_results.json", "w"
                ) as result_file:
                    json.dump(non_entity_avg_results, result_file, indent=2)

                with open(
                    f"{result_path}/non_entity_subset_raw_results.json", "w"
                ) as result_file:
                    json.dump(non_entity_raw_results, result_file, indent=2)

        for data_name in tqdm(experiment_data_map.keys(), desc="Plotting Experiment"):
            for ks_type in ["1", "2"]:

                subset_path = f"{fp}/results/experiment_results/{data_name}"
                base_path = f"{fp}/results/experiment_results/{data_name}"

                with open(
                    f"{subset_path}/entity_subset_average_results.json", "r"
                ) as result_file:
                    subset_results = json.load(result_file)

                with open(
                    f"{base_path}/non_entity_subset_average_results.json", "r"
                ) as result_file:
                    base_results = json.load(result_file)

                plt.figure()
                for forget_n, forget_method in enumerate(results_getter.forget_methods):
                    plot_relationship(
                        color=f"C{forget_n}",
                        marker="o",
                        forget_method=forget_method,
                        base_results=base_results,
                        subset_results=subset_results,
                        ks_type=ks_type,
                    )
                plot_relationship(
                    color="k",
                    marker="D",
                    forget_method="retain",
                    base_results=base_results,
                    subset_results=subset_results,
                    ks_type=ks_type,
                )
                plot_relationship(
                    color="k",
                    marker="s",
                    forget_method="full",
                    base_results=base_results,
                    subset_results=subset_results,
                    ks_type=ks_type,
                )

                colour_legend = plt.legend(title="Forget Type", framealpha=0.3)
                plt.gca().add_artist(colour_legend)

                face_color = ["k", "none"]
                sizes = [9, 6]
                h = [
                    plt.plot(
                        [],
                        [],
                        marker="o",
                        color="k",
                        markerfacecolor=face_col,
                        markersize=size,
                        ls="none",
                        alpha=0.5,
                    )[0]
                    for face_col, size in zip(face_color, sizes)
                ]

                labels = ["Non-entity questions", "Entity questions"]

                size_legend = plt.legend(
                    h,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.15),
                    ncol=2,
                    framealpha=0.3,
                )

                plt.yscale("symlog")
                plt.xlabel("Model Utility")
                plt.ylabel(f"Forget Quality ({ks_type} sided)")
                plt.xlim(left=-0.05)
                plt.tight_layout()
                plt.savefig(
                    f"{subset_path}/subset_result_plot_{ks_type}_"
                    f"{args.forget_length}.pdf"
                )
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
    parser.add_argument("--plot_truth_ratio_cdf", action="store_true")
    parser.add_argument(
        "--experiment_type",
        type=str,
        default=None,
        help="Type of experiment run, defaults to none for individual analysis",
    )
    parser.add_argument(
        "--forget_length",
        type=str,
        default="long",
        help="Length on the forget training.",
    )
    args = parser.parse_args()
    main(args)
