import argparse

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from arcsf.data.data_module import BlankQAFormatter, EvalQADataset, get_data
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

    args = parser.parse_args()
    model_dir = args.model_path

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.config.pad_token_id = tokenizer.eos_token_id

    experiment_config = yaml.safe_load(open(f"{model_dir}/experiment_config.yaml"))
    random_seed = experiment_config["seed"]
    exp_name = experiment_config["experiment_name"]
    print(f"experiment name: {exp_name}")

    forget_data, retain_data = get_data(
        experiment_config["data_config"]["dataset_name"],
        random_seed=random_seed,
        **experiment_config["data_config"]["data_kwargs"],
    )
    splits = {
        "retain": retain_data,
        "forget": forget_data,
    }

    qa_formatter = BlankQAFormatter()
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
        random_seed=random_seed,
        max_new_tokens=50,
    )
