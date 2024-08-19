import argparse
import json
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import yaml
from transformers import AutoTokenizer

CONFIG_LOC = "configs/experiment"
OUTPUT_LOC = "output"


def open_json_path(path):
    with open(path) as file:
        json_object = json.load(file)
    return json_object


def main(args):
    args = args
    fp = args.experiment_path

    forget_types = ["idk", "difference", "kl", "ascent", "retain"]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    exp_cfg = yaml.safe_load(open(glob(f"{fp}/retain/*/experiment_config.yaml")[0]))
    data_name = exp_cfg["config_names"]["data_config"]
    full_model_path = exp_cfg["full_model_name"]
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

    token_losses = open_json_path(
        glob(f"output/{full_model_path}/full/*/forget_token_loss.json")[0]
    )
    data.append(np.array(token_losses["target_probs"][row]))
    tokens = token_losses["target_labels"][row]
    forget_types.append("full")
    labels = [tokenizer.decode(token) for token in tokens]
    row_token_positions = np.arange(len(data[0]))

    plt.figure(figsize=(10, 5))
    for data_name, data in zip(forget_types, data):
        plt.plot(row_token_positions, data, label=data_name)

    plt.xticks(row_token_positions, labels, rotation=35)
    plt.legend()
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
