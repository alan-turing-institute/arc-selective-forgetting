import argparse
import os
import random

import numpy as np
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from arcsf.data.data_module import BlankQAFormatter, EvalQADataset, get_data
from arcsf.eval.utils import all_eval
from arcsf.utils import get_device

if __name__ == "__main__":
    # currently this is our random seed
    parser = argparse.ArgumentParser(
        description=(
            "Runs quantitative evaluation, comparing output logits of model against"
            " targets. It currently saves the truth ratio and input-wise losses."
        )
    )
    batch_size = 1
    parser.add_argument("model_dir", type=str, help="Relative path to model directory.")
    parser.add_argument(
        "--data_split",
        "-s",
        default="forget",
        type=str,
        help="Split of data to evaluate on",
    )
    parser.add_argument(
        "--random_seed",
        "-r",
        default=None,
        type=int,
        help="Random seed for script",
    )

    args = parser.parse_args()
    model_dir = args.directory

    if args.random_seed:
        rand = args.random_seed
    else:
        rand = random.randint(-10000, 10000)

    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

    # experiment_config

    model.config.pad_token_id = tokenizer.eos_token_id

    experiment_config = yaml.safe_load(open(model_dir + "/experiment_config.yaml"))
    forget_data, retain_data = get_data(
        "tofu", **experiment_config["data_config"], random_seed=rand
    )
    splits = {
        "retain": retain_data,
        "forget": forget_data,
    }

    device = get_device()
    print(f"Pytorch device: {device}")
    qa_formatter = BlankQAFormatter()
    dataset = EvalQADataset(
        splits[args.data_split],
        tokenizer,
        qa_formatter,
        "standard",
        quantitative_eval=True,
        qualitative_eval=True,
        device=get_device(),
        n_perturbed=2,
        random_seed=rand,
        padding=False,
    )

    # pass all to the function
    outputs = all_eval(
        model,
        dataset,
        batch_size,
        device,
        tokenizer,
        max_new_tokens=50,
    )
    save_dir = f"{model_dir}/eval/{args.data_split}/"
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(save_dir + "/truth_ratios.txt", outputs["truth_ratios"])
    np.savetxt(save_dir + "/all_losses.txt", outputs["all_losses"])
    np.savetxt(save_dir + "/rouge1_scores.txt", outputs["rouge1_recall"])
    np.savetxt(save_dir + "/rougeL_scores.txt", outputs["rougeL_recall"])
