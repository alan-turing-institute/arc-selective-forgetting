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
MODEL_LOC = "configs/model"
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
    forget_types = ["idk", "difference", "retain"]
    exp_cfg = yaml.safe_load(open(glob(f"{fp}/retain/*/experiment_config.yaml")[0]))
    model_name = exp_cfg["config_names"]["model_config"]
    model_id = yaml.safe_load(open(f"{MODEL_LOC}/{model_name}/{model_name}.yaml"))
    tokenizer = AutoTokenizer.from_pretrained(model_id["model_id"])
    print(f"{fp}/retain/*/experiment_config.yaml")
    data_name = exp_cfg["config_names"]["data_config"]
    full_model_path = exp_cfg["full_model_name"]

    idk_vs_gt = {
        key: {"target": None, "idk": None} for key in ["retain", "full"] + forget_types
    }

    while True:
        retain_token_losses = open_json_path(
            glob(f"{fp}/retain/*/forget_token_loss.json")[0]
        )
        for q_type in ["target", "idk"]:
            idk_vs_gt["retain"][q_type] = np.mean(
                [
                    sum(probs) / len(probs)
                    for probs in retain_token_losses[f"{q_type}_probs"]
                ]
            )
        row = random.randint(0, len(retain_token_losses["target_losses"]))
        data = []
        for forget_type in forget_types:
            token_losses = open_json_path(
                glob(f"{fp}/{forget_type}/*/forget_token_loss.json")[0]
            )
            for q_type in ["target", "idk"]:
                idk_vs_gt[forget_type][q_type] = np.mean(
                    [
                        sum(probs) / len(probs)
                        for probs in token_losses[f"{q_type}_probs"]
                    ]
                )
            data.append(np.array(token_losses["target_probs"][row]))
            tokens = token_losses["target_labels"][row]
            labels = [tokenizer.decode(token) for token in tokens]
        if len(labels) < 30:
            break

    token_losses = open_json_path(
        glob(
            f"output/{full_model_path}/full/*/eval_outputs"
            f"/{data_name}/forget_token_loss.json"
        )[0]
    )
    for q_type in ["target", "idk"]:
        idk_vs_gt["full"][q_type] = np.mean(
            [sum(probs) / len(probs) for probs in token_losses[f"{q_type}_probs"]]
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

    print(json.dumps(idk_vs_gt, indent=2))

    prob_map = np.zeros((len(idk_vs_gt), 2))
    for idx, (_, values) in enumerate(idk_vs_gt.items()):
        prob_map[idx, 0] = values["target"]
        prob_map[idx, 1] = values["idk"]

    fig, ax = plt.subplots(figsize=(6, 8))
    im = plt.imshow(prob_map, cmap="YlGn")
    plt.yticks(forget_type_positions, idk_vs_gt.keys())
    plt.xticks([0, 1], ["Target", "IDK"])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    cbar = plt.colorbar(
        im,
        fraction=0.1,
        orientation="vertical",
        location="right",
    )
    cbar.set_label("Answer Probability", labelpad=20, rotation=270)
    plt.tight_layout()
    plt.savefig(f"{fp}/mean_answer_probs.pdf")
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
