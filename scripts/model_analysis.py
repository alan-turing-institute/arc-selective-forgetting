import argparse
import json
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from arcsf.eval.metrics import ecdf

CONFIG_LOC = "configs/experiment"
OUTPUT_LOC = "output"

height = 10
width = height * 4
# plt.style.use("ggplot")
# plt.rcParams["axes.facecolor"] = "white"
# plt.rcParams["figure.figsize"] = (width, height)
# plt.rcParams["axes.linewidth"] = 1.6
# plt.rcParams["axes.spines.right"] = False
# plt.rcParams["axes.spines.top"] = False
# plt.rcParams["axes.spines.left"] = False
# plt.rcParams["axes.spines.bottom"] = False
# plt.rcParams["font.size"] = 16
# plt.rcParams["axes.grid"] = False


def get_key_with_property(d, func):
    return func(d, key=d.get)


def open_json_path(path):
    with open(path) as file:
        json_object = json.load(file)
    return json_object


def plot_truth_ratios(forget_types, args):
    train = args.train_length
    for split in ["retain", "forget"]:
        retain_eval = open_json_path(
            glob(f"{OUTPUT_LOC}/{args.experiment_name}/retain/*/eval_outputs.json")[0]
        )
        exp_config = yaml.safe_load(
            open(
                glob(
                    f"{OUTPUT_LOC}/{args.experiment_name}"
                    "/retain/*/experiment_config.yaml"
                )[0]
            )
        )
        retain_data, retain_y_values = ecdf(
            torch.tensor(retain_eval[f"{split}_truth_ratios"])
        )
        plt.plot(retain_data, retain_y_values, "-", label="retain", color="k")
        data_name = exp_config["config_names"]["data_config"]
        full_eval = open_json_path(
            glob(
                f"{OUTPUT_LOC}/{exp_config['full_model_name']}/full/*/"
                f"eval_outputs/{data_name}/"
                "eval_outputs.json"
            )[0]
        )
        full_data, full_y_values = ecdf(
            torch.tensor(full_eval[f"{split}_truth_ratios"])
        )
        plt.plot(full_data, full_y_values, "-.", label="full", color="k")

        for col_idx, forget_type in enumerate(forget_types):
            forget_runs = glob(f"{OUTPUT_LOC}/{args.experiment_name}/{forget_type}/*")
            all_forget_checkpoints = {
                forget_run.split("/")[-1]: None for forget_run in forget_runs
            }
            training_length = {
                forget_run.split("/")[-1]: None for forget_run in forget_runs
            }
            for forget_uuid in all_forget_checkpoints.keys():
                all_forget_checkpoints[forget_uuid] = glob(
                    f"{OUTPUT_LOC}/{args.experiment_name}/{forget_type}/{forget_uuid}/"
                    f"eval_checkpoints/*.json"
                )
                training_length[forget_uuid] = yaml.safe_load(
                    open(
                        f"{OUTPUT_LOC}/{args.experiment_name}"
                        f"/{forget_type}/{forget_uuid}/experiment_config.yaml"
                    )
                )["model_config"]["trainer_kwargs"]["num_train_epochs"]
            if train == "short":
                plot_key = get_key_with_property(training_length, min)
            elif train == "long":
                plot_key = get_key_with_property(training_length, max)

            forget_checkpoints = sorted(all_forget_checkpoints[plot_key])
            for alpha_idx, checkpoint in enumerate(forget_checkpoints[:-1]):
                forget_eval = open_json_path(checkpoint)
                forget_data, forget_y_values = ecdf(
                    torch.tensor(forget_eval[f"{split}_truth_ratios"])
                )
                plt.plot(
                    forget_data,
                    forget_y_values,
                    color=f"C{col_idx}",
                    alpha=(0.2 + ((alpha_idx / len(forget_checkpoints)) * 0.8)),
                )
            forget_eval = open_json_path(forget_checkpoints[-1])
            forget_data, forget_y_values = ecdf(
                torch.tensor(forget_eval[f"{split}_truth_ratios"])
            )
            plt.plot(
                forget_data,
                forget_y_values,
                label=f"{forget_type}",
                color=f"C{col_idx}",
                alpha=(0.2 + (((alpha_idx + 1) / len(forget_checkpoints)) * 0.8)),
            )

        plt.legend()
        plt.xlim(0, 1)
        plt.ylabel("Cumulative Probability")
        plt.xlabel("Truth Ratio")
        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_LOC}/{args.experiment_name}/{train}_iterative_cdf_{split}.pdf"
        )
        plt.close()


def plot_model_quality(forget_types, args):
    train = args.train_length
    retain_eval = open_json_path(
        glob(f"{OUTPUT_LOC}/{args.experiment_name}/retain/*/eval_outputs.json")[0]
    )
    exp_config = yaml.safe_load(
        open(
            glob(
                f"{OUTPUT_LOC}/{args.experiment_name}"
                f"/retain/*/experiment_config.yaml"
            )[0]
        )
    )
    data_name = exp_config["config_names"]["data_config"]
    full_eval = open_json_path(
        glob(
            f"{OUTPUT_LOC}/{exp_config['full_model_name']}/full/*/"
            f"eval_outputs/{data_name}/"
            "eval_outputs.json"
        )[0]
    )

    for ks_type in [1, 2]:
        plt.figure()
        for forget_index, forget_type in enumerate(forget_types):
            forget_runs = glob(f"{OUTPUT_LOC}/{args.experiment_name}/{forget_type}/*")
            all_forget_checkpoints = {
                forget_run.split("/")[-1]: None for forget_run in forget_runs
            }
            training_length = {
                forget_run.split("/")[-1]: None for forget_run in forget_runs
            }
            for forget_uuid in all_forget_checkpoints.keys():
                all_forget_checkpoints[forget_uuid] = glob(
                    f"{OUTPUT_LOC}/{args.experiment_name}/{forget_type}/{forget_uuid}/"
                    f"eval_checkpoints/*.json"
                )
                training_length[forget_uuid] = yaml.safe_load(
                    open(
                        f"{OUTPUT_LOC}/{args.experiment_name}"
                        f"/{forget_type}/{forget_uuid}/experiment_config.yaml"
                    )
                )["model_config"]["trainer_kwargs"]["num_train_epochs"]

            if train == "short":
                plot_key = get_key_with_property(training_length, min)
            elif train == "long":
                plot_key = get_key_with_property(training_length, max)

            forget_checkpoints = sorted(all_forget_checkpoints[plot_key])
            forget_quality = []
            model_utility = []
            for _, checkpoint in enumerate(forget_checkpoints[:-1]):
                forget_eval = open_json_path(checkpoint)
                forget_quality.append(forget_eval[f"forget_quality_{ks_type}"])
                model_utility.append(forget_eval["retain_model_utility_3"])
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

        plt.plot(
            retain_eval["retain_model_utility_3"],
            0,
            "D",
            color="k",
            label="retain",
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
            f"{OUTPUT_LOC}/{args.experiment_name}/{train}_results_{ks_type}.pdf"
        )
        plt.close()


def plot_retain_forget_utility(forget_types, args):
    train = args.train_length
    retain_eval = open_json_path(
        glob(f"{OUTPUT_LOC}/{args.experiment_name}/retain/*/eval_outputs.json")[0]
    )
    exp_config = yaml.safe_load(
        open(
            glob(
                f"{OUTPUT_LOC}/{args.experiment_name}"
                f"/retain/*/experiment_config.yaml"
            )[0]
        )
    )
    data_name = exp_config["config_names"]["data_config"]
    full_eval = open_json_path(
        glob(
            f"{OUTPUT_LOC}/{exp_config['full_model_name']}/full/*/"
            f"eval_outputs/{data_name}/"
            "eval_outputs.json"
        )[0]
    )
    plt.figure()
    for forget_index, forget_type in enumerate(forget_types):
        forget_runs = glob(f"{OUTPUT_LOC}/{args.experiment_name}/{forget_type}/*")
        all_forget_checkpoints = {
            forget_run.split("/")[-1]: None for forget_run in forget_runs
        }
        training_length = {
            forget_run.split("/")[-1]: None for forget_run in forget_runs
        }
        for forget_uuid in all_forget_checkpoints.keys():
            all_forget_checkpoints[forget_uuid] = glob(
                f"{OUTPUT_LOC}/{args.experiment_name}/{forget_type}/{forget_uuid}/"
                f"eval_checkpoints/*.json"
            )
            training_length[forget_uuid] = yaml.safe_load(
                open(
                    f"{OUTPUT_LOC}/{args.experiment_name}"
                    f"/{forget_type}/{forget_uuid}/experiment_config.yaml"
                )
            )["model_config"]["trainer_kwargs"]["num_train_epochs"]

        if train == "short":
            plot_key = get_key_with_property(training_length, min)
        elif train == "long":
            plot_key = get_key_with_property(training_length, max)

        forget_checkpoints = sorted(all_forget_checkpoints[plot_key])
        forget_quality = []
        model_utility = []
        for _, checkpoint in enumerate(forget_checkpoints[:-1]):
            forget_eval = open_json_path(checkpoint)
            forget_quality.append(forget_eval["forget_model_utility_3"])
            model_utility.append(forget_eval["retain_model_utility_3"])
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

    plt.plot(
        retain_eval["retain_model_utility_3"],
        retain_eval["forget_model_utility_3"],
        "D",
        color="k",
        label="retain",
    )

    plt.plot(
        full_eval["retain_model_utility_3"],
        full_eval["forget_model_utility_3"],
        "s",
        label="full",
        color="k",
    )

    plt.legend()
    plt.xlabel("Model Utility")
    plt.ylabel("Forget Utility")
    plt.savefig(f"{OUTPUT_LOC}/{args.experiment_name}/{train}_utility_results.pdf")
    plt.close()


def main(args):
    forget_types = ["difference", "kl", "ascent", "idk"]
    # forget_types = ["difference", "idk"]
    plot_truth_ratios(forget_types=forget_types, args=args)
    plot_model_quality(forget_types=forget_types, args=args)
    plot_retain_forget_utility(forget_types=forget_types, args=args)


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

    parser.add_argument("train_length", type=str)

    args = parser.parse_args()
    main(args)
