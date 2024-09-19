import argparse
import json
from glob import glob
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from arcsf.eval.metrics import ecdf

# Constants for file locations
CONFIG_LOC = "configs/experiment"
OUTPUT_LOC = "output"

# Plotting params
height = 10
width = height * 4


def get_key_with_property(d: dict, func: Callable):
    """
    returns the minimum or maximum train time, depending on the function passed

    Args:
        d: dictionary containing the save path of the experiment files
        func: min or max, depending on context

    Returns:
        save path of desired output (max train time or min train time)
    """
    return func(d, key=d.get)


def open_json_path(path: str):
    """
    Opens a json file given a path

    Args:
        path: path to json file

    Returns:
        json dictionary located at path.
    """
    with open(path) as file:
        json_object = json.load(file)
    return json_object


def plot_truth_ratios(forget_types, args):
    # this plots the truth ratio plots
    train = args.train_length
    # for retain and forget evals
    for split in ["retain", "forget"]:
        # load the retain eval
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
        # get the retain and full truth ratio CDFs and plot them
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

        # loop through forget types
        for col_idx, forget_type in enumerate(forget_types):
            # get the forget runs for this forget type
            forget_runs = glob(f"{OUTPUT_LOC}/{args.experiment_name}/{forget_type}/*")
            all_forget_checkpoints = {
                forget_run.split("/")[-1]: None for forget_run in forget_runs
            }
            training_length = {
                forget_run.split("/")[-1]: None for forget_run in forget_runs
            }
            # loop through and save the checkpoints and train length for each run
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
            # obtain the correct training lengths
            if train == "short":
                plot_key = get_key_with_property(training_length, min)
            elif train == "long":
                plot_key = get_key_with_property(training_length, max)
            # sort checkpoints
            forget_checkpoints = sorted(all_forget_checkpoints[plot_key])
            # plot the truth ratios at each checkpoint except the last
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
            # plot the last
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
        # save the figure
        plt.legend()
        plt.xlim(0, 1)
        plt.ylabel("Cumulative Probability")
        plt.xlabel("Truth Ratio")
        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_LOC}/{args.experiment_name}/{train}_iterative_cdf_{split}.pdf"
        )
        plt.close()


def plot_model_quality(forget_types: list[str], args: argparse.Namespace):
    # This plots the model utility vs the forget quality
    train = args.train_length
    # get the retain evaluation outputs
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
    # get the full model evalutation outputs
    data_name = exp_config["config_names"]["data_config"]
    full_eval = open_json_path(
        glob(
            f"{OUTPUT_LOC}/{exp_config['full_model_name']}/full/*/"
            f"eval_outputs/{data_name}/"
            "eval_outputs.json"
        )[0]
    )
    # for each ks type in our forget quality
    for ks_type in [1, 2]:
        # create a figure
        plt.figure()
        for forget_index, forget_type in enumerate(forget_types):
            # find the forget runs and create the checkpoint/train length dict
            forget_runs = glob(f"{OUTPUT_LOC}/{args.experiment_name}/{forget_type}/*")
            all_forget_checkpoints = {
                forget_run.split("/")[-1]: None for forget_run in forget_runs
            }
            training_length = {
                forget_run.split("/")[-1]: None for forget_run in forget_runs
            }
            # loop through the checkpoints and save the file names
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

            # pick the correct model key
            if train == "short":
                plot_key = get_key_with_property(training_length, min)
            elif train == "long":
                plot_key = get_key_with_property(training_length, max)

            forget_checkpoints = sorted(all_forget_checkpoints[plot_key])
            forget_quality = []
            model_utility = []

            # get results and full checpoints
            for _, checkpoint in enumerate(forget_checkpoints):
                forget_eval = open_json_path(checkpoint)
                forget_quality.append(forget_eval[f"forget_quality_{ks_type}"])
                model_utility.append(forget_eval["retain_model_utility_3"])
            # plot the points
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
            # plot the connecting lines
            alphas = np.linspace(0.1, 0.9, len(forget_quality))
            for index, alpha in enumerate(alphas[:-1]):
                plt.plot(
                    model_utility[index : index + 2],
                    forget_quality[index : index + 2],
                    "-",
                    alpha=alpha,
                    color=colour,
                )
        # plot retain and full values
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

        # save figure
        plt.legend()
        plt.xlabel("Model Utility")
        plt.ylabel(f"Forget Quality ({ks_type} sided)")
        plt.yscale("symlog")
        plt.savefig(
            f"{OUTPUT_LOC}/{args.experiment_name}/{train}_results_{ks_type}.pdf"
        )
        plt.close()


def plot_retain_forget_utility(forget_types: list[str], args: argparse.Namespace):
    """
    Plots the forget utility against the retain utility

    Args:
        forget_types: forget types to be plotted
        args: arguments from the argparser ("")
    """
    train = args.train_length
    # retrieve the retain eval outputs
    # currently glob works because there is one retain model per experiment
    retain_eval = open_json_path(
        glob(f"{OUTPUT_LOC}/{args.experiment_name}/retain/*/eval_outputs.json")[0]
    )
    # Get the experiment config from the output file
    exp_config = yaml.safe_load(
        open(
            glob(
                f"{OUTPUT_LOC}/{args.experiment_name}"
                f"/retain/*/experiment_config.yaml"
            )[0]
        )
    )
    # get the name of the dataset
    data_name = exp_config["config_names"]["data_config"]
    # use this along with experiment config to retreive full model eval outputs
    full_eval = open_json_path(
        glob(
            f"{OUTPUT_LOC}/{exp_config['full_model_name']}/full/*/"
            f"eval_outputs/{data_name}/"
            "eval_outputs.json"
        )[0]
    )
    # create a figure to plot on
    plt.figure()
    # loop over forget types
    for forget_index, forget_type in enumerate(forget_types):
        # get forget runs in this model
        forget_runs = glob(f"{OUTPUT_LOC}/{args.experiment_name}/{forget_type}/*")
        all_forget_checkpoints = {
            forget_run.split("/")[-1]: None for forget_run in forget_runs
        }
        training_length = {
            forget_run.split("/")[-1]: None for forget_run in forget_runs
        }
        # loop through the uuid filenames
        for forget_uuid in all_forget_checkpoints.keys():
            # gets a list of checkpoints from forget loop
            all_forget_checkpoints[forget_uuid] = glob(
                f"{OUTPUT_LOC}/{args.experiment_name}/{forget_type}/{forget_uuid}/"
                f"eval_checkpoints/*.json"
            )
            # gets the training (forget) length from the experiment config
            training_length[forget_uuid] = yaml.safe_load(
                open(
                    f"{OUTPUT_LOC}/{args.experiment_name}"
                    f"/{forget_type}/{forget_uuid}/experiment_config.yaml"
                )
            )["model_config"]["trainer_kwargs"]["num_train_epochs"]

        # get the correct run from the training lengths in configs
        if train == "short":
            plot_key = get_key_with_property(training_length, min)
        elif train == "long":
            plot_key = get_key_with_property(training_length, max)

        # sort the forget checkpoints so they plot in the correct order
        # create lists to store the values
        forget_checkpoints = sorted(all_forget_checkpoints[plot_key])
        forget_quality = []
        model_utility = []
        # loop through to create
        for _, checkpoint in enumerate(forget_checkpoints):
            forget_eval = open_json_path(checkpoint)
            forget_quality.append(forget_eval["forget_model_utility_3"])
            model_utility.append(forget_eval["retain_model_utility_3"])

        # color from matplotlib configs
        colour = f"C{forget_index}"
        # plot the points
        plt.scatter(
            model_utility,
            forget_quality,
            marker="o",
            s=[30 * (i + 1) for i in range(len(forget_quality))],
            alpha=0.5,
            label=forget_type,
            color=colour,
        )
        # plot the connecting lines
        alphas = np.linspace(0.1, 0.9, len(forget_quality))
        for index, alpha in enumerate(alphas[:-1]):
            plt.plot(
                model_utility[index : index + 2],
                forget_quality[index : index + 2],
                "-",
                alpha=alpha,
                color=colour,
            )

    # plot retain and full models
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
    # save figure
    plt.legend()
    plt.xlabel("Model Utility")
    plt.ylabel("Forget Utility")
    plt.savefig(f"{OUTPUT_LOC}/{args.experiment_name}/{train}_utility_results.pdf")
    plt.close()


def main(args):
    # Currently hardcoded the forget types
    forget_types = ["difference", "kl", "ascent", "idk"]
    forget_types = ["difference", "idk"]
    plot_truth_ratios(forget_types=forget_types, args=args)
    plot_model_quality(forget_types=forget_types, args=args)
    plot_retain_forget_utility(forget_types=forget_types, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "From a model path generates evaluation plots for an individual model."
        )
    )
    parser.add_argument(
        "experiment_name",
        type=str,
        help="Experiment name on which you want to plot",
    )

    parser.add_argument(
        "train_length",
        type=str,
        help="Train length to plot, currently works for 'long' or 'short'.",
    )

    args = parser.parse_args()
    main(args)
