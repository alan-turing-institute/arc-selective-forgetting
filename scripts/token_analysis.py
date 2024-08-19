import argparse
import json
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import SymLogNorm
from transformers import AutoTokenizer

CONFIG_LOC = "configs/experiment"
OUTPUT_LOC = "output"
height = 10
width = height * 3
plt.style.use("ggplot")
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.figsize"] = (width, height)
plt.rcParams["axes.linewidth"] = 1.6
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.left"] = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["font.size"] = 16
plt.rcParams["axes.grid"] = False


def open_json_path(path):
    with open(path) as file:
        json_object = json.load(file)
    return json_object


def main(args):
    args = args
    fp = args.experiment_path

    forget_types = ["idk", "difference", "ascent", "retain"]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    exp_cfg = yaml.safe_load(open(glob(f"{fp}/retain/*/experiment_config.yaml")[0]))
    data_name = exp_cfg["config_names"]["data_config"]
    full_model_path = exp_cfg["full_model_name"]
    loop_count = 0
    while True:
        row = random.randint(0, 50)
        data = []
        for forget_type in forget_types:
            token_losses = open_json_path(
                glob(f"{fp}/{forget_type}/*/forget_token_loss.json")[0]
            )
            data.append(np.array(token_losses["target_probs"][row]))
            tokens = token_losses["target_labels"][row]
            labels = [tokenizer.decode(token) for token in tokens]
        if len(labels) < 30:
            break
        loop_count += 1
        if loop_count > 30:
            raise ValueError("Short enough question could not be found")

    token_losses = open_json_path(
        glob(
            f"output/{full_model_path}/full/*/eval_outputs"
            f"/{data_name}/forget_token_loss.json"
        )[0]
    )
    data.append(np.array(token_losses["target_probs"][row]))
    tokens = token_losses["target_labels"][row]
    forget_types.append("full")
    labels = [tokenizer.decode(token) for token in tokens]
    row_token_positions = np.arange(len(data[0]))
    forget_type_positions = np.arange(len(forget_types))

    plt.figure(figsize=(16, 8))
    all_data = np.vstack(data)
    plt.imshow(all_data, norm=SymLogNorm(linthresh=1e-4), cmap="YlGn")
    plt.yticks(forget_type_positions, forget_types)
    plt.xticks(row_token_positions, labels, rotation=30)

    cbar = plt.colorbar(
        orientation="horizontal",
        location="top",
    )
    cbar.set_label("Correct Token Probability", labelpad=15)
    plt.tight_layout()
    plt.savefig(f"{fp}/random_row_token_probs.pdf")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "From an experiment path generates evaluation plots for every experiment."
        )
    )
    parser.add_argument(
        "experiment_path",
        type=str,
        help="Path to experiment.",
    )
    args = parser.parse_args()
    main(args)
