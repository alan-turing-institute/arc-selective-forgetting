import argparse
import os

import numpy as np
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from arcsf.data.data_module import BlankQAFormatter, EvalQADataset, get_data
from arcsf.eval.utils import all_eval
from arcsf.utils import get_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs quantitative evaluation, comparing output logits of model against"
            " targets. It currently saves the truth ratio and input-wise losses."
        )
    )
    batch_size = 5
    parser.add_argument("model_dir", type=str, help="Relative path to model directory.")
    parser.add_argument(
        "--data_split",
        "-s",
        default="forget",
        type=str,
        help="Split of data to evaluate on",
    )

    args = parser.parse_args()
    model_dir = args.model_dir

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, padding_side="right", padding=True
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    experiment_config = yaml.safe_load(open(model_dir + "/experiment_config.yaml"))
    random_seed = experiment_config["seed"]
    forget_data, retain_data = get_data(
        experiment_config["data_config"]["dataset_name"],
        **experiment_config["data_config"]["data_kwargs"],
        random_seed=random_seed,
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
        device=get_device(),
        n_perturbed=2,
        random_seed=random_seed,
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
