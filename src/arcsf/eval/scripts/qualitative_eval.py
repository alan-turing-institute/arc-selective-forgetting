import argparse
import random

import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from arcsf.data.data_module import EvalQADataset, get_data, qa_formatter_blank
from arcsf.eval.utils import qualitative_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs qualitative evaluation, comparing outputs of model"
            " against target strings."
        )
    )
    parser.add_argument(
        "model_path", type=str, help="Relative path to model directory."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Relative path (from args.model_path) to the experiment config file.",
    )
    parser.add_argument(
        "--data_split",
        "-s",
        default="forget",
        type=str,
        help="Split of data to evaluate on",
    )
    parser.add_argument(
        "--n_inputs",
        "-n",
        default=10,
        type=int,
        help="Max number of inputs to sample",
    )

    parser.add_argument(
        "--random_seed",
        "-r",
        default=None,
        type=int,
        help="Random seed for script",
    )

    args = parser.parse_args()
    model_dir = args.model_path

    if args.random_seed:
        rand = args.random_seed
    else:
        rand = random.randint(-10000, 10000)

    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model.config.pad_token_id = tokenizer.eos_token_id

    experiment_config = yaml.safe_load(open(f"{model_dir}/{args.config_path}"))
    exp_name = experiment_config["experiment_name"]
    print(f"experiment name: {exp_name}")

    forget_data, retain_data = get_data(
        "tofu", random_seed=rand, **experiment_config["data_config"]
    )
    splits = {
        "retain": retain_data,
        "forget": forget_data,
    }

    qa_formatter = qa_formatter_blank
    dataset = EvalQADataset(
        splits[args.data_split],
        tokenizer,
        qa_formatter,
        "standard",
        qualitative_eval=True,
        quantitative_eval=False,
    )
    # pass all to the function
    qualitative_eval(
        model,
        tokenizer,
        dataset,
        n_inputs=args.n_inputs,
        random_seed=rand,
        max_new_tokens=50,
    )
